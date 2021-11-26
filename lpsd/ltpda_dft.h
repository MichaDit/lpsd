/*
 * Header for ltpda_dft.c
 *
 * $Id$
 */

void  print_usage(char *version);

void dft(double *Mr, double *Vr, long int *Navs,
        double *xdata, long int nData, long int segLen, double *Cr, double *Ci, double olap, int order);

void xdft(double *Mr, double *Mi, double *XX, double *YY, double *M2, long int *Navs,
        double *xdata, double *ydata, long int nData, long int segLen, double *Cr, double *Ci, double olap, int order);

/*void xdft(double *XBARr, double *XBARi, double *S2, double *XYr, double *XYi, double *XX, double *YY, long int *Navs,
 *    double *xdata, double *ydata, long int nData, long int segLen,
 *    double *Cr, double *Ci, double olap, int order);
 */

void remove_linear_drift(double *segm, double *data, int nfft);
