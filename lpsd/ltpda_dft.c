/*
 * Mex file that implements the core DFT part of the LPSD algorithm.
 *
 * M Hewitson  15-01-08
 *
 * $Id$
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <mex.h>

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
void dft(double *Pr, double *Vr, long int *Navs,
         double *xdata, long int nData, long int segLen, double *Cr, double *Ci, double olap, int order)
{
  long int istart;
  double shift, start;
  double *px, *cr, *ci;
  double rxsum, ixsum;
  double Xr, Mr, M2, Qr;
  double p, *x, *a;
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
  x = (double *)calloc(segLen, sizeof(double));    /* detrending output */
  a = (double *)calloc(order + 1, sizeof(double)); /* detrending coefficients */

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
    px = &(xdata[istart]);

    /* pointer to DFT coeffs */
    cr = &(Cr[0]);
    ci = &(Ci[0]);

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

    /* Go over all samples in this segment */
    rxsum = ixsum = 0.0;
    for (jj = 0; jj < segLen; jj++)
    {
      p = x[jj];
      rxsum += (*cr) * p; /* cos term */
      ixsum += (*ci) * p; /* sin term */

      /* increment pointers */
      cr++;
      ci++;
    }
    /*mexPrintf("   xsum=(%g +i %g), ysum=(%g + i%g)\n", rxsum, ixsum, rysum, iysum);*/

    /* Average the cross-power
     * Rsum += rxsum*rxsum + ixsum*ixsum;
     */

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

  /* mexPrintf("     start: %f \t istart: %d | %d \n", start, istart, nData-istart);*/
  /*mexPrintf(" Rsum=%g, MR=%g \n", Rsum,MR); */

  /* clean up */
  free(x);
  free(a);

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


/*
 * Fast linear detrending routine
 *
 */
void remove_linear_drift(double *segm, double *data, int nfft)
{
  int i;
  long double sx, sy, stt, sty, a, b, xm;

  sx = (long double)nfft * (long double)(nfft - 1) / 2.0L;
  xm = (long double)(nfft - 1) / 2.0L;
  stt = ((long double)nfft * (long double)nfft * (long double)nfft - (long double)nfft) / 12.0L;

  sy = sty = 0;
  for (i = 0; i < nfft; i++)
  {
    sy += data[i];
    sty += (i - xm) * data[i];
  }
  b = sty / stt;
  a = (sy - sx * b) / nfft;
  for (i = 0; i < nfft; i++)
  {
    segm[i] = data[i] - (a + b * i);
  }
}
