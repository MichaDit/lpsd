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
 * Compute LPSD
 *
 */
void lcsd_c(double* Sxx, double* S, double* ENBW, double *devxx, double *dev, double *asd, // outputs
                double* x1data, double* x2data, long int nData, //input data, its length
                double olap, int order,  //overlap percentage, detrending order,
                int nf, double fs,       //number of frequency bins, sampling frequency
                double* Cr, double* Ci, //DFT coefficients,
                long int* segLen, double *S1, double *S2, //segment lengths, window norms
                bool csd //whether to do CSD (true) or PSD (false)
                )
{
    double Pr, Vr, A2ns, B2ns, S12;
    long int nsegs, ii, jj, shift;

    Pr = 0;
    Vr = 0;
    nsegs = 0;
    shift = 0;

    for (ii = 0; ii < nf; ii++) {
        double *Cr_bin;
        double *Ci_bin;
        Cr_bin = (double *)calloc(segLen[ii], sizeof(double));
        Ci_bin = (double *)calloc(segLen[ii], sizeof(double));

        for(jj=0; jj<segLen[ii]; jj++){
            Cr_bin[jj] = Cr[shift+jj];
            Ci_bin[jj] = Ci[shift+jj];
        }

        dft(
            &Pr,
            &Vr,
            &nsegs,
            x1data,
            x2data,
            nData,
            segLen[ii],
            Cr_bin,
            Ci_bin,
            olap,
            order,
            csd
        );

        A2ns = 2.0 * Pr;
        B2ns = 4.0 * Vr / (double)(nsegs);
        S12 = S1[ii] * S1[ii];
        ENBW[ii] = fs * S2[ii] / S12;
        // Scale PS / PSD
        Sxx[ii] = A2ns / fs / S2[ii];
        S[ii] = A2ns / S12;
        // Scale sqrt(variance)
        devxx[ii] = sqrt(B2ns / pow(fs, 2) / pow(S2[ii], 2));
        dev[ii] = sqrt(B2ns / pow(S12, 2));
        asd[ii] = sqrt(Sxx[ii]);

        shift += segLen[ii];
        free(Cr_bin);
        free(Ci_bin);
    }
}

/*
 * Short routine to compute the DFT at a single frequency
 *
 */
void dft(double *Pr, double *Vr, long int *Navs,
         double *x1data, double* x2data, long int nData, long int segLen,
         double *Cr, double *Ci, double olap, int order, bool csd)
{
  long int istart;
  double shift, start;
  double *px1, *px2, *cr, *ci;
  double rxsum, ixsum;
  double Xr, Mr, M2, Qr;
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
  Xr = 0.0;
  Qr = 0.0;
  Mr = 0.0;
  M2 = 0.0;

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
    rxsum = ixsum = 0.0;
    for (jj = 0; jj < segLen; jj++)
    {
      p1 = x1[jj];
      p2 = x2[jj];
      rxsum += (*cr) * p1; /* cos term */
      ixsum += (*ci) * p2; /* sin term */

      /* increment pointers */
      cr++;
      ci++;
    }

    /* Welford's algorithm to update mean and variance */
    if (ii == 0)
    {
      Mr = (rxsum * rxsum + ixsum * ixsum);
    }
    else
    {
      Xr = (rxsum * rxsum + ixsum * ixsum);
      Qr = Xr - Mr;
      Mr += Qr / ii;
      M2 += Qr * (Xr - Mr);
    }
  }

  /* clean up */
  free(x1);
  free(x2);
  free(a1);
  free(a2);

  /* Outputs */
  *Pr = Mr;
  if (navg == 1)
  {
    *Vr = Mr * Mr;
  }
  else
  {
    *Vr = M2 / (navg - 1);
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
