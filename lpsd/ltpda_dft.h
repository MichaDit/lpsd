/*
 * Header for ltpda_dft.c
 *
 * $Id$
 */

void  print_usage(char *version);

void dft(double *Mr, double *Vr, long int *Navs,
        double *xdata1, double* xdata2, long int nData, long int segLen,
        double *Cr, double *Ci, double olap, int order, bool csd);

void lpcd_c(double* Sxx, double* S, double* ENBW, double *devxx, double *dev, double *asd, // outputs
                double* x1data, double* x2data, long int nData, //input data, its length
                double olap, int order,  //overlap percentage, detrending order,
                int nf, double fs,       //number of frequency bins, sampling frequency
                double* Cr, double* Ci, //DFT coefficients,
                long int* segLen, double *S1, double *S2, //segment lengths, window norms
                bool csd //whether to do CSD (true) or PSD (false)
                );

void detrend(int order, double *px, int segLen, double *x, double *a);
