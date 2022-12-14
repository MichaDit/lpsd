/*
 * Mex file that implements the core DFT part of the LPSD algorithm.
 *
 * M Hewitson 2008-01-15 (original code)
 * Artem Basalaev 2022-11-11 (modification of DFT for CSD case, adding LCSD code)
 *
 * $Id$
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "ltpda_dft.h"
#include "version.h"
#include "c_sources/polyreg.c"

#define DEBUG 0

/*
 * Rounding function
 *
 */
int myround(double x)
{
  return ((int)(floor(x + 0.5)));
}

/*
 * Short routine to compute the DFT at a single frequency
 *
 */
void dft(double *Pr_r, double *Pr_i, double *Vr_r,  double *Vr_i, long int *Navs, // outputs
         double *x1data, double* x2data, long int nData, long int segLen, //input data, its length, length of segment
         double *Cr, double *Ci, double olap, int order, // DFT coefficients, overlap percentage, detrending order
         bool csd) //whether to do CSD (true) or PSD (false)
{
  long int istart;
  double shift, start;
  double *px1, *px2, *cr, *ci;
  double rxsum1, ixsum1;
  double rxsum2, ixsum2;
  double Xr_r, Xr_i, Mr_r, Mr_i, XM_diff_i, XM_diff_r, M2_r, M2_i, Qr_i, Qr_r;
  double p1, p2, *x1, *x2, *a1, *a2;
  long int jj, ii;

  /* Compute the number of averages we want here */
  double ovfact = 1. / (1. - olap / 100.);
  double davg = ((double)((nData - segLen)) * ovfact) / segLen + 1;
  long int navg = myround(davg);

  /* Compute steps between segments */
  if (navg == 1)
    shift = 1;
  else
    shift = (double)(nData - segLen) / (double)(navg - 1);

  if (shift < 1)
    shift = 1;

  /*   mexPrintf("Seglen: %d\t | Shift: %f\t | navs: %d\n", segLen, shift, navg);*/

  /* allocate vectors */
  x1 = (double *)calloc(segLen, sizeof(double));    /* detrending output */
  a1 = (double *)calloc(order + 1, sizeof(double)); /* detrending coefficients */
  x2 = (double *)calloc(segLen, sizeof(double));    /* detrending output */
  a2 = (double *)calloc(order + 1, sizeof(double)); /* detrending coefficients */

  /* Loop over segments */
  start = 0.0;
  Xr_r = 0.0;
  Qr_r = 0.0;
  Mr_r = 0.0;
  M2_r = 0.0;
  Xr_i = 0.0;
  Qr_i = 0.0;
  Mr_i = 0.0;
  M2_i = 0.0;
  XM_diff_r = 0.0;
  XM_diff_i = 0.0;

  for (ii = 0; ii < navg; ii++)
  {
    /* compute start index */
    istart = myround(start);
    start += shift;

    /* pointer to start of this segment */
    px1 = &(x1data[istart]);
    px2 = &(x2data[istart]);

    /* pointer to DFT coeffs */
    cr = &(Cr[0]);
    ci = &(Ci[0]);

    detrend(order, px1, segLen, x1, a1);
    if (csd){
        detrend(order, px2, segLen, x2, a2);
    }
    else{ //doing PSD, x1=x2, hence already detrended input data
        memcpy(x2, x1, segLen * sizeof(double));
    }

    /* Go over all samples in this segment */
    rxsum1 = ixsum1 = rxsum2 = ixsum2 = 0.0;
    for (jj = 0; jj < segLen; jj++)
    {
      // dot(C[segment], x1[segment])
      p1 = x1[jj];
      rxsum1 += (*cr) * p1; /* cos term */
      ixsum1 += (*ci) * p1; /* sin term */

      // dot(C[segment], x2[segment])
      p2 = x2[jj];
      rxsum2 += (*cr) * p2; /* cos term */
      ixsum2 += (*ci) * p2; /* sin term */

      /* increment pointers */
      cr++;
      ci++;
    }

    /* Welford's algorithm to update mean and variance */
    if (ii == 0)
    {
      //M = dot(C[segment], x1[segment]) *  conj(dot(C[segment],x2[segment]))
      // conjugate x2
      ixsum2 = -1.0 * ixsum2;
      // multiply real and imaginary parts
      // (x+yi)(u+vi) = (xu-yv)+(xv+yu)i
      Mr_r = (rxsum1 * rxsum2 - ixsum2 * ixsum1);
      Mr_i = (rxsum1 * ixsum2 + ixsum1 * rxsum2);
    }
    else
    {
      // for Xr same calculation as above
      ixsum2 = -1.0 * ixsum2;
      // multiply real and imaginary parts
      // (x+yi)(u+vi) = (xu-yv)+(xv+yu)i
      Xr_r = (rxsum1 * rxsum2 - ixsum2 * ixsum1);
      Xr_i = (rxsum1 * ixsum2 + ixsum1 * rxsum2);

      Qr_r = Xr_r - Mr_r;
      Qr_i = Xr_i - Mr_i;

      Mr_r += Qr_r / ii;
      Mr_i += Qr_i / ii;

      // M2 += Qr * (Xr - Mr)
      XM_diff_r = Xr_r - Mr_r;
      XM_diff_i = Xr_i - Mr_i;
      M2_r = Qr_r*XM_diff_r - Qr_i*XM_diff_i;
      M2_i = Qr_r*XM_diff_i + Qr_i*XM_diff_r;
    }
  }

  /* clean up */
  free(x1);
  free(x2);
  free(a1);
  free(a2);

  /* Outputs */
  *Pr_r = Mr_r;
  *Pr_i = Mr_i;
  if (navg == 1)
  {
    *Vr_r = Mr_r * Mr_r - Mr_i * Mr_i;
    *Vr_i = 2 * Mr_i * Mr_r;
  }
  else
  {
    *Vr_r = M2_r / (navg - 1);
    *Vr_i = M2_i / (navg - 1);
  }
  *Navs = navg;
}


void detrend(int order, double *px, int segLen, double *x, double *a){
    /* Detrend segment */
    switch (order)
    {
    case -1:
      /* no detrending */
      memcpy(x, px, segLen * sizeof(double));
      break;
    case 0:
      /* mean removal */
      polyreg0(px, segLen, x, a);
      break;
    case 1:
      /* linear detrending */
      polyreg1(px, segLen, x, a);
      break;
    case 2:
      /* 2nd order detrending */
      polyreg2(px, segLen, x, a);
      break;
    case 3:
      /* 3rd order detrending */
      polyreg3(px, segLen, x, a);
      break;
    case 4:
      /* 4th order detrending */
      polyreg4(px, segLen, x, a);
      break;
    case 5:
      /* 5th order detrending */
      polyreg5(px, segLen, x, a);
      break;
    case 6:
      /* 6th order detrending */
      polyreg6(px, segLen, x, a);
      break;
    case 7:
      /* 7th order detrending */
      polyreg7(px, segLen, x, a);
      break;
    case 8:
      /* 8th order detrending */
      polyreg8(px, segLen, x, a);
      break;
    case 9:
      /* 9th order detrending */
      polyreg9(px, segLen, x, a);
      break;
    case 10:
      /* 10th order detrending */
      polyreg10(px, segLen, x, a);
      break;
    }
}
