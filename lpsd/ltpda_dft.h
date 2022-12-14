/*
 * Header for ltpda_dft.c
 *
 * $Id$
 */

void  print_usage(char *version);

// for compiling as DLL on Windows, uncomment part of the line below to make the function callable
/*__declspec(dllexport) */void dft(double *Pr_r, double *Pr_i, double *Vr_r,  double *Vr_i, long int *Navs, // outputs
         double *x1data, double* x2data, long int nData, long int segLen, //input data, its length, length of segment
         double *Cr, double *Ci, double olap, int order, // DFT coefficients, overlap percentage, detrending order
         bool csd); //whether to do CSD (true) or PSD (false)

void detrend(int order, double *px, int segLen, double *x, double *a);
