/* Copyright: Centre for Reservoir Geophysics, Imperial College London, 2010 */
/* All rights reserved.                                                      */
/* first calculate ref, then do FX on ref, then clauclate x by ref again and start the BP again.*/
#include "su.h"
#include "segy.h"
#include "header.h"
#include <signal.h>
#include <omp.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/************************* self documentation ********************************/
char *sdoc[] = {
"                                                                            ",
" BP  --         Reflectivity inversion by basis pursuit                        ",
"                                                                            ",
" BP < infile > outfile [optional parameters]                  ",
"                                                                            ",
" Required Parameters:                                                       ",
"   wfile=                     wavelet file                                  ",
" Optional Parameters:                                                       ",
"   N=10                       represent thickness of the bed                ",
"   lamda=2.3                  weight factor of basis pursuit                ",
"   CGAccuracy=0.0001          Accuracy of cgsolver                          ",
"   FeaTol=0.1                 thresh value of PrimalFeas and DualFeas       ",
"   PDGapTol=0.1               thresh value of PDGap                         ",
"   MaxBPIter=30               max iteration number of basis pursuit         ",
"   MOF=1                      whether to use LS solution to initialize      ",

NULL};

/*
 * Credits: Ruo Wang, CRG,ICL  AUG/2011
 *
 * Reference:
 *
 * Revision 1.0 
 */

/*********************** end self doc ****************************************/
#define LOOKFAC	2	/* Look ahead factor for npfaro	  */
#define PFA_MAX	720720	/* Largest allowed nfft	          */

/* Function Prototypes */
static void cgsolver(float **a, float *rhs,
                     int nrow, int ncol, int niter,
                     float error, float *x);
void Synthetic(int nrow, int ncol, float **wedge,
               float *c, float *x);
void Analysis(int nrow, int ncol, float **wedge, 
              float *c, float *x);
void Ref_wedge(int nta, int ncol, int N, 
               float **wedge_ref);
void Seis_wedge(int nta, int nw, int nr, int ncol, 
                float *wavelet, float **wedge_ref, 
                float **wedge_seis);
void BP(float **data, float **allxx, int nrow, 
        int ncol,int T0,float lamda, float **wedge,
        float **wedge_ref, float **allcoef2d, 
        int ntra, float CGAccuracy, float FeaTol,
        float PDGapTol, float delta, float gamma, 
        int MaxBPIter); 
void MOF_fun(float **data, float **allxx, 
             int nrow, int ncol, int ntra, 
             float gamma2, float **wedge, 
             int MaxCGIter, float CGAccuracy);
float norm2(float *data, int n);
void normalize(float *trace, int dim, float factor);
void normalize_2d(float **trace, int dim1, int dim2, 
                  float factor, int verbose);
float maxamp(float **trace, int dim1, int dim2, int verbose);
void convmtx(float **t, float *v, int nv, int n);

int cuda_mtrx_multiply(int nrow1, int ncol1, float **A, int nrow2, int ncol2, float **B, float **C); 
int cuda_mtrx_vector_multiply(int nrow1, int ncol1, float **A, int nrow2, float *B, float *C);
static void cuda_cgsolver(float **a, float *rhs, int nrow, int ncol,
                     int niter, float error, float *x);

segy tr, wtr, rtr;

int
main(int argc, char **argv)
{   
    
    int nta, ntra;
   
    int T0;
    int ncol;
    float lamda;

    float dt;  
    int itr, it, i, n, N;
    int MOF, FX_flag, FXiter;
    float **data, **reflectivity;
    float **wedge_ref, **wedge_seis;
    int in_norm, out_norm;
    float max_amp=1.0;
 
    float *wavelet;
    int wave_sign;
    int nw, nr;
    float maxvalue;

    float CGAccuracy, FeaTol, PDGapTol;
    float delta, gamma, gamma2;
    int  MaxBPIter, MaxCGIter;

    char *wfile;
    FILE *wfp;
    FILE *headerfp;

    float taper;	/* length of taper			*/
    float fmin;		/* minimum frequency to process in Hz	*/
    float fmax;		/* maximum frequency to process in Hz	*/
    float twlen;       	/* time window length 			*/
    int ntrw;	       	/* number of traces in window		*/
    int ntrf;		/* number of traces for filter		*/
   
    float **allxx, **allcoef2d;

    FILE *timefp;
    time_t t1, t2;
 
    initargs(argc,argv);
    requestdoc(1);

    fprintf(stderr, "Start recording time...!\n");
    timefp = fopen("tRecord","w+");
    time(&t1);
    fprintf(timefp,"Main begin time,%lld\n",(long long)t1);
    fflush(timefp);
    fprintf(stderr,"tRecord file made.\n");
    fprintf(stderr,"Main begin time,%lld\n",(long long)t1);

    if (!gettr(&tr))  err("can't get first trace");
    nta = tr.ns;
    dt = (float) tr.dt/1000000.0;
    warn("nta=%d dt=%g",nta,dt);

    ntra = gettra(&tr,0);
    warn("ntra = %d", ntra);
    erewind(stdin);

    /******begin of the parameter of FX*****************/	
    if (!getparfloat("taper", &taper)) taper=.1;
    if (taper==0.0) taper=.004; 
    if (!getparfloat("fmin", &fmin)) fmin= 6.;

    if (!getparfloat("fmax", &fmax)) fmax=.6/(2*dt);
    if (!getparfloat("twlen", &twlen)) twlen=(float)(nta-1)*dt;
    if (twlen<.3) {
        twlen=.3;
    }
    if (!getparint("ntrw", &ntrw)) ntrw=10;
    if (!ntrw) {
        ntrw = 10;
    }

    if (!getparint("ntrf", &ntrf)) ntrf=4;
    /******end of the parameter of FX*******************/	
  
    if (!getparint("wave_sign", &wave_sign))        wave_sign = 1;    
    if (!getparint("N", &N))                        N = 10;
    if (!getparint("MOF", &MOF))                    MOF = 0;
    if (!getparint("FX_flag", &FX_flag))            FX_flag = 0;
    if (!getparint("FXiter", &FXiter))              FXiter = 1;

    if (!getparint("in_norm", &in_norm))            in_norm = 1;
    if (!getparint("out_norm", &out_norm))          out_norm = 1;

    if (!getparfloat("lamda", &lamda))              lamda = 2.3;

    if (!getparfloat("CGAccuracy", &CGAccuracy))    CGAccuracy = 0.0001;
    if (!getparfloat("FeaTol", &FeaTol))            FeaTol = 0.1;
    if (!getparfloat("PDGapTol", &PDGapTol))        PDGapTol = 0.1;
    if (!getparfloat("delta", &delta))              delta = 1e-4;
    if (!getparfloat("gamma", &gamma))              gamma = 1;
    if (!getparint("MaxBPIter", &MaxBPIter))        MaxBPIter = 30;
    if (!getparint("MaxCGIter", &MaxCGIter))        MaxCGIter = 200;


    if (!getparstring("wfile",&wfile))
        err("wavelet file name must be specified!");

    gamma2=gamma*gamma;
    
    ncol=0;/*initial the number of matrix col*/ 
    for(n=1; n<=N; n++)
        ncol += nta-n;
    ncol= ncol*2;
    ncol+=nta;
    
    warn("the value of ncol is : %d", ncol);

    allxx = alloc2float(ncol, ntra);    
    data = alloc2float(nta,ntra);
    allcoef2d = alloc2float(nta,ntra);
    reflectivity = alloc2float(nta,ntra);/* the capacity of ref and intrace are the same*/
    wedge_ref = alloc2float(ncol, nta);
    wedge_seis = alloc2float(ncol, nta);

    headerfp = etmpfile();
    wfp = efopen(wfile,"r");
    fgettr(wfp,&wtr);/*read wfp into wtr*/
    nw = wtr.ns;
    nr = nta - nw + 1;
    warn("nw=%d, nr=%d", nw, nr);
    wavelet = alloc1float(nw);


/***************************Input**************************************/

    memcpy( (void *)wavelet, (const void*)wtr.data, nw*FSIZE);
    normalize(wavelet, nw, 1.0);

    maxvalue = 0.0;
    T0 = 0;
    for (it=nw/3; it<nw*2/3; it++){
        if(fabs(wavelet[it])>maxvalue) {
            maxvalue = fabs(wavelet[it]);
            T0 = it;
        }
    }
    warn("T0=%d", T0);
    if(wavelet[T0]<0.0) for(it=0; it<nw; it++) wavelet[it] *= -1.0;
    for(it=0; it<nw; it++) wavelet[it] *= wave_sign;

    for (itr=0; itr<ntra; itr++) {
        gettr(&tr);
        efwrite(&tr, HDRBYTES, 1, headerfp);
        memcpy( (void *)data[itr], (const void*)tr.data, nta*FSIZE);
    }
    erewind(headerfp);
    erewind(stdin);

/***************************Processing************************************/

    if(in_norm) {
        warn("Normalize the input traces");
        
        max_amp=maxamp(data, ntra, nta, 0);
        normalize_2d(data, ntra, nta, 1.0,1);
    }

    /*time(&t1);
    warn("Main begin time: %lld", (long long)t1);*/

    /**build the wedge reflector matrix**/
    Ref_wedge(nta, ncol, N, wedge_ref);
    
    /**build the wedge seismic response matrix**/
    t2 = time(&t2);
    fprintf(timefp, "Seis_wedge model build starts,%lld, %lld, %lld\n",(long long)t2,(long long)(t2-t1),(long long)((t2-t1)/60));
    fflush(timefp);

    Seis_wedge(nta, nw, nr, ncol, wavelet,
               wedge_ref, wedge_seis);

    t2 = time(&t2);
    fprintf(timefp, "Seis_wedge model build ends, %lld, %lld, %lld\n",(long long)t2,(long long)(t2-t1),(long long)((t2-t1)/60));
    fflush(timefp);

    
    /**initialization with the least squares solution**/
    t2 = time(&t2);
    fprintf(timefp, "MOF initial model calculation starts,%lld, %lld, %lld\n",(long long)t2,(long long)(t2-t1),(long long)((t2-t1)/60));
    fflush(timefp);

    MOF_fun(data, allxx, nta, ncol, ntra, gamma2,
            wedge_seis, MaxCGIter, CGAccuracy);

    t2 = time(&t2);
    fprintf(timefp, "MOF initial model calculation ends, %lld, %lld, %lld\n",(long long)t2,(long long)(t2-t1),(long long)((t2-t1)/60));
    fflush(timefp);

    /**Basis Pursuit**/
    t2 = time(&t2);
    fprintf(timefp, "Basis Pursuit process starts,%lld, %lld, %lld\n",(long long)t2,(long long)(t2-t1),(long long)((t2-t1)/60));
    fflush(timefp);
 
    BP(data, allxx, nta, ncol, T0,
       lamda, wedge_seis, wedge_ref, 
       allcoef2d, ntra, CGAccuracy, 
       FeaTol, PDGapTol, delta, gamma, 
       MaxBPIter);

    t2 = time(&t2);
    fprintf(timefp, "Basis Pursuit process ends, %lld, %lld, %lld\n",(long long)t2,(long long)(t2-t1),(long long)((t2-t1)/60));
    fflush(timefp);

    for (itr=0; itr<ntra; itr++) {
        for (i=0; i<T0; i++)
            reflectivity[itr][i] = 0.0;
        for (i=0; i<nta; i++){
            if((i+T0)<nta)
            reflectivity[itr][i+T0] = allcoef2d[itr][i];
        }
    }
    
    /*time(&t2);
    warn("Main end time: %lld, %lld, %lld", (long long)t2, (long long)(t2-t1), (long long)((t2-t1)/60));*/

    if (out_norm){
        warn("The Output Reflectivity Normalized");
        normalize_2d(reflectivity, ntra, nta, 1.0,1);

    }
   
/**************************Output*********************************************/
   
    for(itr=0; itr<ntra; itr++){
        for(it=0; it<nta; it++)
            tr.data[it]=reflectivity[itr][it];
        tr.ns = nta;
        tr.dt = dt*1000000;
        puttr(&tr);
    }

    erewind(stdin);
    efclose(wfp);
    free1float(wavelet);
    free2float(allcoef2d);
    free2float(allxx);
    free2float(wedge_ref);
    free2float(wedge_seis);
    free2float(data);
    free2float(reflectivity);

    t2 = time(&t2);
    fprintf(timefp, "Main end time, %lld, %lld, %lld\n",(long long)t2,(long long)(t2-t1),(long long)((t2-t1)/60));
    fflush(timefp);
    fprintf(stderr,"Main end time, %lld, %lld, %lld\n",(long long)t2,(long long)(t2-t1),(long long)((t2-t1)/60));

    return(CWP_Exit());
}

/*******************************Functions***********************************/

void Ref_wedge(int nta, int ncol, int N, float **wedge_ref)
{
 int i, j, n, m;
 for(i=0; i<nta; i++)
     for(j=0; j<ncol; j++)
         wedge_ref[i][j]=0.0;

 for(n=1; n<=N; n++){
     for(m=1; m<=nta-n; m++){
        wedge_ref[m-1][m-1+(n-1)*(nta-n+1)]=1.0;
        wedge_ref[m+n-1][m-1+(n-1)*(nta-n+1)]=1.0;
     }
 }

 for(n=1; n<=N; n++){
     for(m=1; m<=nta-n; m++){
        wedge_ref[m-1][m-1+(n-1)*(nta-n+1)+(ncol-nta)/2]=1.0;
        wedge_ref[m+n-1][m-1+(n-1)*(nta-n+1)+(ncol-nta)/2]=-1.0;
     }
 }
 for(i=0; i<nta; i++){
     wedge_ref[i][i+ncol-nta]=1.0;

 }

 warn("wedge model reflectivity matrix building finished!");
}


void Seis_wedge(int nta, int nw, int nr, int ncol, float *wavelet, float **wedge_ref, float **wedge_seis)
{
 int i, j, k;
 float **W;/*convolution matrix of wavelet*/

 /*wedge_seis1=alloc2float(ncol, nta+nw);*/
 W=alloc2float(nr, nta);

 convmtx(W, wavelet, nw, nr);
 for(i=0; i<nta; i++)
     for(j=0; j<ncol; j++)
         wedge_seis[i][j]=0.0;/*initial the value of wedge matrix*/
 
 /*for(i=0; i<ncol; i++){
     for(j=0; j<nta; j++){
         for(k=0; k<nw; k++){
             if((j+k)<nta)wedge_seis[j+k][i] += wavelet[k]*wedge_ref[j][i];
         }
     }    
 }*/

 cuda_mtrx_multiply(nta, nr, W, nr, ncol, wedge_ref, wedge_seis);
 /*for(i=0; i<nta; i++)
     for(j=0; j<nr; j++)
         for(k=0; k<ncol; k++)
             wedge_seis[i][k]+=W[i][j]*wedge_ref[j][k];*/

 warn("wedge model seismic response matrix building finished!"); 

 /*free2float(wedge_seis1);*/
}

/* calculate x=[A -A]*c */
void Synthetic(int nrow, int ncol, float **wedge, float *c, float *x)
{
 int i, j;
 float *c1;
 
 c1 = alloc1float(ncol);

 for(i=0; i<ncol; i++)
     c1[i]=0.0;
 for(i=0; i<nrow; i++)
     x[i]=0.0;
 for(i=0; i<ncol; i++)
     c1[i]=(c[i]-c[i+ncol]);

 for(i=0; i<nrow; i++)x[i]=0.0;
 
 cuda_mtrx_vector_multiply(nrow, ncol, wedge, ncol, c1, x);
 /*for (i = 0; i < nrow; i++){
     for (j = 0; j < ncol; j++){
         x[i] += wedge[i][j]*c1[j];
     }
 }*/
 /*for(i=0; i<nrow; i++)
     warn("x[%d]=%f", i, x[i]);exit(0);*/

 free1float(c1);
}

/* calculate c=[A -A]T *x */
void Analysis(int nrow, int ncol, float **wedge, float *c, float *x)
{
 int i, j;
 float **wedgeT, *c1;

 wedgeT = alloc2float(nrow,ncol);
 c1=alloc1float(ncol);

 for(i=0; i<ncol; i++){
     for(j=0; j<nrow; j++){
         wedgeT[i][j]=0.0;
     }
 }
 for(i=0; i<ncol; i++)c1[i]=0.0;
 for(i=0; i<ncol*2; i++)c[i]=0.0;

 for(i=0; i<ncol; i++)
     for(j=0; j<nrow; j++)
         wedgeT[i][j]=wedge[j][i];

 cuda_mtrx_vector_multiply(ncol, nrow, wedgeT, nrow, x, c1);
 /*for (i = 0; i < ncol; i++)
     for (j = 0; j < nrow; j++)
         c1[i] += wedgeT[i][j]*x[j];*/
 for(i=0; i<ncol; i++){
     c[i]=c1[i];
     if(c1[i]!=0.0)c[i+ncol]=-c1[i];
 }
 
 free2float(wedgeT);
 free1float(c1);
}

void MOF_fun(float **data, float **allxx,
             int nrow, int ncol, int ntra, 
             float gamma2, float **wedge, 
             int MaxCGIter, float CGAccuracy)
{
 int i, j, k, itr;
 float **wedgeT, **A_2;
 int nt;
 float *x01, *xx, *tmp_data;

 nt=nrow;

 A_2 = alloc2float(nt, nt);
 wedgeT = alloc2float(nt, ncol);
 x01 = alloc1float(nt);
 xx = alloc1float(ncol);
 tmp_data = alloc1float(nt);
          
 for(i=0; i<ncol; i++){
     for(j=0; j<nt; j++){
         wedgeT[i][j]=wedge[j][i];
     }
 }
 
 for(i=0; i<nt; i++){
     for(j=0; j<nt; j++){
         A_2[i][j]=0.0;
     }
 }
 
 cuda_mtrx_multiply(nt, ncol, wedge, ncol, nt, wedgeT, A_2);
 /*for(i=0; i<nt; i++){
     for(j=0; j<nt; j++){
         for(k=0; k<ncol; k++){
             A_2[i][j]+=wedge[i][k]*wedgeT[k][j];
         }
     }
 }*/
 /*for(i=0; i<nt; i++)
     for(j=0; j<nt; j++)
         if(A_2[i][j]!=0)warn("A_2[%d][%d]=%f", i, j, A_2[i][j]);exit(0);*/
 
 warn("Calculate initial value from the input trace!");
 for(itr=0; itr<ntra; itr++){
     for(i=0; i<nt; i++)
         tmp_data[i]=data[itr][i];
  
     for(i=0; i<nt; i++)x01[i]=0.0;
     for(i=0; i<ncol; i++)xx[i]=0.0;
     
     cgsolver(A_2, tmp_data, nt, nt, MaxCGIter, CGAccuracy, x01);
     /*for(i=0; i<nt; i++)
         warn("x01[%d]=%f", i, x01[i]);exit(0);*/


     cuda_mtrx_vector_multiply(ncol, nt, wedgeT, nt, x01, xx);
     /*for(i=0; i<ncol; i++){
         for(j=0; j<nt; j++){
             xx[i]+=wedgeT[i][j]*x01[j];
         }
     }*/
     for(i=0; i<ncol; i++)
         allxx[itr][i]=xx[i];
 }

 free2float(wedgeT);
 free2float(A_2);
 free1float(x01);
 free1float(xx);
 free1float(tmp_data);
}

void BP(float **data, float **allxx, int nrow, 
        int ncol, int T0, float lamda, 
        float **wedge,float **wedge_ref, 
        float **allcoef2d, int ntra, 
        float CGAccuracy, float FeaTol, 
        float PDGapTol, float delta, 
        float gamma, int MaxBPIter)  
{

 int itr;
 int i, j, k, bpitr;
 int nt;
 int MaxCGIter;
 float **c;

 float maxap, mp;
 float *xx;
 float *sg, *ap, *ap2;
 float sigma=0.99;

 float zx, sumx, Obj;
 float mu, mu0, minmu; 
 float gamma2, delta2;
 float PrimalStep, PrimalFeas, DualFeas, PDGap;

 float stepx, stepz;
 int flagx, flagz, index, indez;

 float *tmp_data;
 float *x0, *y0, *z0, *x, *y, *z;
 float *v1, *v2, *v3;
 float *dx, *dy, *dz;
 float *syn0, *ana0, *ana1, *ana4;

 float *f;
 float **A, **A2, **A2T, **A3;
 
 float *H, *H2, *h;
 float *r, *p, *w;
 float *xold;

 float *stex, *stez;
 float **allx, **ally, **allz, *allmu;

 MaxCGIter=200;
 gamma2=gamma*gamma;
 delta2=delta*delta;
 nt=nrow;

/*******allocate the space*****************/
 sg = alloc1float(ncol*2);
 ap2 = alloc1float(ncol*2); 
 ap = alloc1float(ncol); 
 xx = alloc1float(ncol);
 c = alloc2float(ncol, ntra);
 allx = alloc2float(ncol*2, ntra);
 allz = alloc2float(ncol*2, ntra);
 ally = alloc2float(nt, ntra);
 allmu = alloc1float(ntra);
 r = alloc1float(nt);
 p = alloc1float(nt);
 w = alloc1float(nt);
 H = alloc1float(ncol*2);
 H2 = alloc1float(ncol*2);
 h =alloc1float(nt);
 f = alloc1float(ncol*4+nt);
 ana0 = alloc1float(ncol*2);
 ana1 = alloc1float(ncol*2);
 ana4 = alloc1float(ncol*2);
 syn0 = alloc1float(nt);
 tmp_data = alloc1float(nt);
 v1 = alloc1float(ncol*2);
 v2 = alloc1float(nt);
 v3 = alloc1float(ncol*2);
 x = alloc1float(2*ncol);
 y = alloc1float(nt);
 z = alloc1float(2*ncol);
 x0 = alloc1float(2*ncol);
 y0 = alloc1float(nt);
 z0 = alloc1float(2*ncol);
 dx = alloc1float(2*ncol);
 dy = alloc1float(nt);
 dz = alloc1float(2*ncol);
 xold = alloc1float(2*ncol);
 stex = alloc1float(2*ncol);
 stez = alloc1float(2*ncol);
 A = alloc2float(nt, nt);
 A2 = alloc2float(ncol*2, nt);
 A2T = alloc2float(nt, ncol*2);
 A3 = alloc2float(nt, ncol*2);

/*****************************************/

 warn("BP Interior method begins!");

/********build the input matrix A2, A2T**********/

 for(i=0; i<nrow; i++){
     for(j=0; j<2*ncol; j++){
         A2[i][j]=0.0;
     }
 }

 for(i=0; i<nrow; i++){
     for(j=0; j<ncol; j++){
         A2[i][j]=wedge[i][j];
         if(wedge[i][j]==0.0)A2[i][j+ncol]=wedge[i][j];
         A2[i][j+ncol]=-wedge[i][j];
     }
 }

 for(i=0; i<ncol*2; i++){
     for(j=0; j<nrow; j++){
         A2T[i][j]=0.0;
         A2T[i][j]=A2[j][i];
     }
 }


 for(itr=0; itr<ntra; itr++){
     for(i=0; i<ncol*2; i++){
         allx[itr][i]=0.0;
         allz[itr][i]=0.0;
     }
 }     

 for(itr=0; itr<ntra; itr++){
     for(i=0; i<nt; i++){
         ally[itr][i]=0.0;
     }
 }

/**********barrier iterations*****************************/

 for(bpitr=0; bpitr<MaxBPIter; bpitr++){
     
     /*warn("BP-Interior iterations!");
     warn("Barrier Iteration number: %d", bpitr+1);        */
         
     for (itr=0; itr<ntra; itr++) {
         
         PrimalStep=0;
         /*warn("Trace number: %d", itr+1);*/
         for(i=0; i<nt; i++){
             tmp_data[i]=data[itr][i];
         }
         for(i=0; i<ncol; i++)
             xx[i]=allxx[itr][i];
 
         /***************initialize*****************/

         for(i=0; i<2*ncol; i++){
             x[i]=0.0;
             z[i]=0.0;
             x0[i]=0.0;
             z0[i]=0.0;
         }
         for(i=0; i<nt; i++){
             y[i]=0.0;
             y0[i]=0.0;
         }
         if(bpitr==0){

             /***initialize x,y,z, make sure x and z are interior, using MOF***/
             for(i=0; i<ncol; i++){
                 if(xx[i]>0)x[i]=xx[i];
                 if(xx[i]<0)x[i+ncol]=-xx[i];
             }
             for(i=0; i<2*ncol; i++){
                 sg[i]=0.0;
                 if(fabs(x[i])>0)sg[i]=x[i]/fabs(x[i]);
             }
             Synthetic(nrow, ncol, wedge, sg, y);
             Analysis(nrow, ncol, wedge, ap2, y);

             for(i=0; i<ncol; i++)ap[i]=ap2[i];

             maxap=0.0;
             for(i=0; i<ncol; i++){
                 if(fabs(ap[i])>maxap)maxap=fabs(ap[i]);
             }
             mp=1.1*maxap;
             for(i=0; i<ncol; i++)ap[i] /= mp;
             for(i=0; i<nt; i++)y[i] /= mp;

             for(i=0; i<ncol; i++){
                 z[i] = 1-ap[i];
                 z[i+ncol] = 1+ap[i];
             }

             for(i=0; i<ncol*2; i++){
                 x[i] += 0.1;
                 z[i] +=gamma2*x[i];
             }

             for(i=0; i<ncol*2; i++){
                 x0[i]=x[i];
                 z0[i]=z[i];
             }
             for(i=0; i<nt; i++)y0[i] = y[i];

             /*****initialize the barrier parameter mu******/
             mu=mu0=0.01;
           
             for(i=0; i<ncol*2; i++)f[i]=z[i]*x[i];
             Synthetic(nrow, ncol, wedge, x, syn0);
             Analysis(nrow, ncol, wedge, ana0, y);
             for(i=ncol*2; i<ncol*2+nt; i++)f[i]=tmp_data[i-ncol*2]-syn0[i-ncol*2]-delta*y[i-ncol*2];
             for(i=ncol*2+nt; i<ncol*4+nt; i++)f[i]=ana0[i-ncol*2-nt]-z[i-ncol*2-nt]+gamma*x[i-ncol*2-nt]+1;
             mu=mu0*norm2(f, ncol*4+nt)/sqrt(ncol*2);
         }
         else{
             for(i=0; i<ncol*2; i++){
                 x[i]=allx[itr][i];
                 z[i]=allz[itr][i];
             }
             for(i=0; i<nt; i++){
                 y[i]=ally[itr][i];
             }
             mu=allmu[itr];
         }

/*************Stop initialization**********************************************/        
 
         /*initialize the intermediate variable*/
         for(i=0; i<ncol*2; i++){
             ana1[i]=0.0;
             ana4[i]=0.0;
         }
         
         /*Calculate the residules*/
         for(i=0; i<ncol*2; i++){
             v1[i]=0.0;
             v1[i]=mu-z[i]*x[i];
         }
         
         Synthetic(nrow, ncol, wedge, x, syn0);
         
         Analysis(nrow, ncol, wedge, ana1, y);

         for(i=0; i<nt; i++){
             v2[i]=0.0;
             v2[i]=tmp_data[i]-syn0[i]-delta2*y[i];
         }
 
         for(i=0; i<ncol*2; i++){
             v3[i]=0.0;
             v3[i]=-ana1[i]-z[i]+lamda+gamma2*x[i];
         }
         
         /*test convergence*/
        
         PrimalFeas=0.0;
         DualFeas=0.0;
         PDGap=0.0; 
         PrimalFeas=norm2(v2, nt)/(1+norm2(x, ncol*2));
         DualFeas=norm2(v3, ncol*2)/(1+norm2(y, nt));
         zx=0.0;
         sumx=0.0;
         Obj=0.0;
         for(i=0; i<ncol*2; i++){
             zx+=z[i]*x[i];
             sumx+=x[i];
             Obj+=fabs(x[i]);
         }
         PDGap=zx/(1+norm2(x, ncol*2)*norm2(z, ncol*2)); 
         /*warn("Pimalfeas=%g, Dualfeas=%g, PDGap=%g, Obj=%g", PrimalFeas, DualFeas, PDGap, Obj);*/
         
         /*solve Newton's direction by CG*/
                 
         /*******build the rhs matrix***********/

         for(i=0; i<ncol*2; i++){
             H[i]=0.0;
             H2[i]=0.0;
             H[i]=1/(z[i]/x[i]+gamma2);
             H2[i]=H[i]*(v1[i]/x[i]-v3[i]);
         }

         for(i=0; i<ncol*2; i++){
             for(j=0; j<nt; j++){
                 A3[i][j]=0.0; 
                 A3[i][j]=H[i]*A2T[i][j];
             }
         }

         for(i=0; i<nt; i++){
             for(j=0; j<nt; j++)
                 A[i][j]=0.0;
         }

 /*time_t t3, t4;
 time(&t3);
 warn("Seis_wedge build begin time: %lld", (long long)t3);*/
         cuda_mtrx_multiply(nt, ncol*2, A2, ncol*2, nt, A3, A);
         /*for(i=0; i<nt; i++){
             for(j=0; j<nt; j++){
                 for(k=0; k<ncol*2; k++){
                     A[i][j]+=A2[i][k]*A3[k][j];
                 }
             }
         }
 time(&t4);
 warn("Seis_wedge end time: %lld, %lld, %lld", (long long)t4, (long long)(t4-t3), (long long)((t4-t3)/60));exit(0);*/
        
         for(i=0; i<nt; i++)
             A[i][i]+=delta2;

         Synthetic(nrow, ncol, wedge, H2, syn0);
     
         for(i=0; i<nt; i++){
             dy[i]=0.0;
             h[i]=0.0;
             h[i]=v2[i]-syn0[i];/*h is the rhs*/
         }
        
         cgsolver(A, h, nt, nt, MaxCGIter, CGAccuracy, dy);
         Analysis(nrow, ncol, wedge, ana4, dy);
         for(i=0; i<2*ncol; i++){ 
             dx[i]=0.0;
             dz[i]=0.0;
         }
         for(i=0; i<2*ncol; i++){
             dx[i]=H[i]*(v1[i]/x[i]-v3[i])+H[i]*ana4[i];
             dz[i]=v1[i]/x[i]-dx[i]*z[i]/x[i];
         }
    
         /*calculate maximum steps*/

         stepx=stepz=1e20;
         flagx=flagz=0;
         index=indez=0;

         for(i=0; i< ncol*2; i++){
             if(dx[i]<0)index=1;
             if(dz[i]<0)indez=1;
         }
         for(i=0; i<ncol*2; i++){
             stex[i]=0.0;
             stez[i]=0.0;
         }

         if(index){
             for(i=0; i<ncol*2; i++){
                 if(dx[i]<0){
                     stex[flagx]=x[i]/(-dx[i]);
                     flagx++;/* at last, there are flag elements in the column stex*/
                 }    
             }
             stepx=stex[0];
             for(i=0; i<flagx; i++){
                 if(stex[i]<stepx)stepx=stex[i];
             }
         }
         if(indez){
             for(i=0; i<ncol*2; i++){
                 if(dz[i]<0){
                     stez[flagz]=z[i]/(-dz[i]);
                     flagz++;/* at last, there are flag elements in the column stex*/
                 }    
             }
             stepz=stez[0];
             for(i=0; i<flagz; i++){
                 if(stez[i]<stepz)stepz=stez[i];
             }
         }

         stepx=(sigma*stepx>1?1:sigma*stepx);
         stepz=(sigma*stepz>1?1:sigma*stepz);
         
         /*Update x,y,z,mu*/

         for(i=0; i<ncol*2; i++){
             xold[i]=0.0;
             xold[i]=x[i];
             x[i]+=stepx*dx[i];
             z[i]+=stepz*dz[i];
         }
         for(i=0; i<nt; i++){
             y[i]+=stepz*dy[i];
         }
         minmu=stepx;
         minmu=(minmu>stepz?stepz:minmu);
         minmu=(minmu>sigma?sigma:minmu);
         mu=(1-minmu)*mu;
         PrimalStep=norm2(dx, ncol*2)*stepx/norm2(xold, ncol*2);
         /*warn("mu=%g, PrimalStep=%g", mu, PrimalStep);*/

         /***the initial value of next iteration***/

         allmu[itr]=mu;
         for(i=0; i<ncol*2; i++){
             allx[itr][i]=x[i];
             allz[itr][i]=z[i];
         }
         for(i=0; i<nt; i++){
             ally[itr][i]=y[i];
         }

     }/*one iteration of all traces is finished*/
     
 }/*all iterations of all traces are ended*/
     
 for(itr=0; itr<ntra; itr++){ 
     for(i=0; i<ncol; i++){
         c[itr][i]=0.0;
         c[itr][i]=allx[itr][i]-allx[itr][i+ncol];
     }
 } 
 for(itr=0; itr<ntra; itr++)
     for(i=0; i<nt; i++)
         allcoef2d[itr][i]=0.0; 
 
 for(itr=0; itr<ntra; itr++){
     for(i=0; i<nrow; i++){
         for(j=0; j<ncol; j++){
             allcoef2d[itr][i]+=wedge_ref[i][j]*c[itr][j];
         }
     }
 }
 
 free1float(sg);  
 free1float(ap);  
 free1float(ap2);  
 free1float(xx);  
 free2float(allx);
 free2float(ally);
 free2float(allz);
 free1float(allmu);
 free2float(c);
 free1float(r);
 free1float(p);
 free1float(w);
 free1float(H);
 free1float(H2);
 free1float(h);
 free1float(f);
 free1float(ana0);
 free1float(ana1);
 free1float(ana4);
 free1float(syn0);
 free1float(tmp_data); 
 free1float(v1);
 free1float(v2);
 free1float(v3);
 free1float(x);
 free1float(y);
 free1float(z);
 free1float(x0);
 free1float(y0);
 free1float(z0);
 free1float(dx);
 free1float(dy);
 free1float(dz); 
 free1float(xold);
 free1float(stex);
 free1float(stez);
 free2float(A);
 free2float(A2);
 free2float(A2T);
 free2float(A3);
}

static void cgsolver(float **a, float *rhs, int nrow, int ncol,
                     int niter, float error, float *x)
/***************************************************************************^M
 * least-squares linear system solver using CG^M
 * ****************************************************************************^M
 * a        array[][] of the matrix^M
 * rhs      array[] of the righ hand side^M
 * nrow     number of rows^M
 * ncol     number of columns^M
 * niter    number of CG iterations^M
 * error    thresholding relative error^M
 * x        least-squares solution^M
 * ***************************************************************************/
{
    float *s, *r, *p, *q, *rn;
    int i, j, icount;
    float alpha, beta;
    float rsum, qsum, rnsum, r0sum;


    /* allocate spaces */
    s = alloc1float(nrow);
    r = alloc1float(ncol);
    p = alloc1float(ncol);
    q = alloc1float(nrow);
    rn = alloc1float(ncol);

    /* initialize the vectors */

    /* x = 0 */
    for(i=0; i<ncol; i++) x[i] = 0.;

    /* s = rhs */
    for(i=0; i<nrow; i++) s[i] = rhs[i];
     
    /* p = r = A^T s*/
    for(i=0; i<ncol; i++) p[i] = 0.;
    for(j=0; j<nrow; j++)
       for(i=0; i<ncol; i++)
          p[i] += a[j][i]*s[j];
    for(i=0; i<ncol; i++) r[i] = p[i];

    for(i=0, rsum=0.; i<ncol; i++) rsum += r[i]*r[i];
    r0sum = rsum*error;
    /*warn("r0sum=%f", r0sum);exit(0);*/

    /* start iteration */
    for(icount=0; icount < niter && rsum>r0sum; icount++)
    {
       /* q = A p */
       /*cuda_mtrx_vector_multiply(nrow, ncol, a, ncol, p, q);*/
       for(i=0; i<nrow; i++)
       {
          q[i] = 0.;
          for(j=0; j<ncol; j++)
             q[i] += a[i][j]*p[j];
       }
       /*for(i=0; i<nrow; i++)
           for(j=0; j<ncol; j++)
               warn("a[%d][%d]=%f", i, j, a[i][j]);exit(0);*/

       for(i=0, qsum=0.; i<nrow; i++) qsum += q[i]*q[i];

       /* alpha = r.r/q.q */
       alpha = rsum/qsum;

       /* x = x + alpha*p, s = s - alpha*q */
       for(i=0; i<ncol; i++) x[i] += alpha*p[i];
       for(i=0; i<nrow; i++) s[i] -= alpha*q[i];

       /* rn = A^T s */
       for(i=0; i<ncol; i++) rn[i] = 0.;
       for(j=0; j<nrow; j++)
          for(i=0; i<ncol; i++)
             rn[i] += a[j][i]*s[j];

       for(i=0, rnsum=0.; i<ncol; i++) rnsum += rn[i]*rn[i];

       /* beta = rn.rn/r.r */
       beta = rnsum/rsum;
       /*warn("beta=%f", beta);exit(0);*/

       /* p = rn + beta*p */
       for(i=0; i<ncol; i++) p[i] = rn[i] + beta*p[i];
       /*for(i=0; i<ncol; i++)
           warn("p[%d]=%f", i, p[i]);exit(0);*/

       /* r = rn */
       for(i=0; i<ncol; i++) r[i] = rn[i];
       rsum = rnsum;
       /*warn("rsum=%f", rsum);exit(0);*/
    }

    /* free the spaces */
    free1float(s);
    free1float(r);
    free1float(p);
    free1float(q);
    free1float(rn);

}

/*calculate the 2-norm of a certain column vector*/
float norm2(float *data, int n)
{
    int i;
    float sum=0.0;
    for(i=0; i<n; i++)
        sum+=data[i]*data[i];
    sum=sqrt(sum);
    return (sum);
}


void normalize(float *trace, int dim, float factor)
{
    float maxvalue=0.0;
    int i;

    for (i=0; i<dim; i++)
        if(fabs(trace[i])>maxvalue) maxvalue = fabs(trace[i]);
    if(!maxvalue) maxvalue=1.0;
    for (i=0; i<dim; i++)
        trace[i] /= maxvalue;
    for (i=0; i<dim; i++)
        trace[i] *= factor;
}

void normalize_2d(float **trace, int dim1, int dim2, float factor, int verbose)
{
    float maxvalue=0.0;
    int i, j;

    for (i=0; i<dim1; i++)
    for (j=0; j<dim2; j++)
        if(fabs(trace[i][j])>maxvalue) maxvalue = fabs(trace[i][j]);
    if(verbose)
    warn("maxvalue=%f", maxvalue);
    if(!maxvalue) maxvalue=1.0;
    for (i=0; i<dim1; i++)
    for (j=0; j<dim2; j++)
        trace[i][j] /= maxvalue;
    for (i=0; i<dim1; i++)
    for (j=0; j<dim2; j++)
        trace[i][j] *= factor;

}

float maxamp(float **trace, int dim1, int dim2, int verbose)
{
    float maxvalue=0.0;
    int i, j;

    for (i=0; i<dim1; i++)
    for (j=0; j<dim2; j++)
        if(fabs(trace[i][j])>maxvalue) maxvalue = fabs(trace[i][j]);
    if(verbose)
    warn("maxvalue=%f", maxvalue);
    if(!maxvalue) maxvalue=1.0;
    return maxvalue;
}

int cuda_mtrx_multiply(int nrow1, int ncol1, float **A, int nrow2, int ncol2, float **B, float **C)
{
    float alpha=1.0;
    float beta=0.0;

    int i, j;
    float *h_A, *h_B, *h_C;
    float *d_a,*d_b,*d_c;/*1d matrix in the device*/

    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    h_A=alloc1float(nrow1*ncol1);
    h_B=alloc1float(nrow2*ncol2);
    h_C=alloc1float(nrow1*ncol2);

    for(i=0; i<nrow1; i++)
        for(j=0; j<ncol1; j++)
            h_A[i*ncol1+j]=A[i][j];/*i*ncols+j*/
    for(i=0; i<nrow2; i++)
        for(j=0; j<ncol2; j++)
            h_B[i*ncol2+j]=B[i][j];
    for(i=0; i<nrow1; i++)
        for(j=0; j<ncol2; j++)
            h_C[i*ncol2+j]=0.0;

    cudaStat=cudaMalloc((void**)&d_a,nrow1*ncol1*sizeof(float));
    if(cudaStat!=cudaSuccess){
        printf("device memory allocation failed");
        return EXIT_FAILURE;
    }
    cudaStat=cudaMalloc((void**)&d_b,nrow2*ncol2*sizeof(float));
    if(cudaStat!=cudaSuccess){
        printf("device memory allocation failed");
        return EXIT_FAILURE;
    }
    cudaStat=cudaMalloc((void**)&d_c,nrow1*ncol2*sizeof(float));
    if(cudaStat!=cudaSuccess){
        printf("device memory allocation failed");
        return EXIT_FAILURE;
    }

    stat=cublasCreate(&handle);
    if(stat!=CUBLAS_STATUS_SUCCESS){
        printf("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    cudaMemcpy(d_a,h_A,nrow1*ncol1*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_B,nrow2*ncol2*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(d_c,0,nrow1*ncol2*sizeof(float));

    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,ncol2,nrow1,nrow2,&alpha,d_b,ncol2,d_a,ncol1,&beta,d_c,ncol2);
    cudaMemcpy(h_C,d_c,nrow1*ncol2*sizeof(float),cudaMemcpyDeviceToHost);

    for(i=0; i<nrow1; i++)
        for(j=0; j<ncol2; j++)
            C[i][j]=h_C[i*ncol2+j];

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);

    free1float(h_A);
    free1float(h_B);
    free1float(h_C);

    return 1;
}

int cuda_mtrx_vector_multiply(int nrow1, int ncol1, float **A, int nrow2, float *B, float *C)
{
    float alpha=1.0;
    float beta=0.0;

    int i, j;
    float *h_A, *h_B, *h_C;
    float *d_a,*d_b,*d_c;/*1d matrix in the device*/

    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    h_A=alloc1float(nrow1*ncol1);
    h_B=alloc1float(nrow2);
    h_C=alloc1float(nrow1);

    for(i=0; i<nrow1; i++)
        for(j=0; j<ncol1; j++)
            h_A[i*ncol1+j]=A[i][j];/*i*ncols+j*/
    for(i=0; i<nrow2; i++)
            h_B[i]=B[i];
    for(i=0; i<nrow1; i++)
            h_C[i]=0.0;
    /*for(i=0; i<nrow1*ncol1; i++)
        if(h_A[i]!=0)warn("h_A[%d]=%f", i, h_A[i]);exit(0);*/

    cudaStat=cudaMalloc((void**)&d_a,nrow1*ncol1*sizeof(float));
    if(cudaStat!=cudaSuccess){
        printf("device memory allocation failed");
        return EXIT_FAILURE;
    }
    cudaStat=cudaMalloc((void**)&d_b,nrow2*sizeof(float));
    if(cudaStat!=cudaSuccess){
        printf("device memory allocation failed");
        return EXIT_FAILURE;
    }
    cudaStat=cudaMalloc((void**)&d_c,nrow1*sizeof(float));
    if(cudaStat!=cudaSuccess){
        printf("device memory allocation failed");
        return EXIT_FAILURE;
    }

    stat=cublasCreate(&handle);
    if(stat!=CUBLAS_STATUS_SUCCESS){
        printf("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    cudaMemcpy(d_a,h_A,nrow1*ncol1*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_B,nrow2*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(d_c,0,nrow1*sizeof(float));

    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,nrow1,nrow2,&alpha,d_b,1,d_a,ncol1,&beta,d_c,1);
     /*float *tmpt;
       tmpt=alloc1float(nrow1);
       cudaMemcpy(tmpt,d_c,nrow1*sizeof(float),cudaMemcpyDeviceToHost);
       for(i=0; i<nrow1; i++)
           if(tmpt[i]!=0)warn("tmpt[%d]=%f", i, tmpt[i]);exit(0);*/

    cudaMemcpy(h_C,d_c,nrow1*sizeof(float),cudaMemcpyDeviceToHost);

    for(i=0; i<nrow1; i++)
        C[i]=h_C[i];

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);

    free1float(h_A);
    free1float(h_B);
    free1float(h_C);

    return 1;
}

/*convmtx(c,n)returns the convolution matrix for vector c*/
/*If c is a column vector and x is a column vector of length n*/
/*then convmtx(c,n)*x is the same as conv(c,x)*/
/*A convolution matrix is a matrix, formed from a vector, 
whose inner product with another vector is the convolution 
of the two vectors.

A = convmtx(c,n) where c is a length m column vector returns 
a matrix A of size (m+n-1)-by-n. The product of A and another 
column vector x of length n is the convolution of c with x.*/
void convmtx(float **t, float *v, int nv, int n)
{
     int i, j, m;
     float *c, *r, *x, *k;
     int *cidx, *ridx;
     int **t_flag, **cidx1, **ridx1;

     m=nv+n-1;

     c=alloc1float(nv+n-1);
     r=alloc1float(n);
     x=alloc1float(n-1+m);
     k=alloc1float(n-1);
     cidx=alloc1int(m);
     ridx=alloc1int(n);
     t_flag=alloc2int(n,m);
     cidx1=alloc2int(n,m);
     ridx1=alloc2int(n,m);

     for(i=0; i<m; i++)
         c[i]=0.0;
     for(i=0; i<n; i++)
         r[i]=0.0;
     for(i=0; i<nv; i++){
         c[i]=v[i];
     }
     for(i=0; i<n-1; i++)
         k[i]=r[n-i-1];
     for(i=0; i<n-1; i++)
         x[i]=k[i];
     for(i=n-1; i<n-1+m; i++)
         x[i]=c[i-n+1];
     for(i=0; i<m; i++)
         cidx[i]=i;
     for(i=0; i<n; i++)
         ridx[i]=n-i;
     for(i=0; i<m; i++){
         for(j=0; j<n; j++){
             cidx1[i][j]=cidx[i];
             /*warn("cidx1[%d][%d]=%d", i, j, cidx1[i][j]);*/
             /*warn("cidx[%d]=%d", i, cidx[i]);*/
         }
     }/*exit(0);*/
     /*exit(0);
     for(i=0; i<m; i++){
         for(j=0; j<n; j++){
   warn("c[%d][%d]=%d", i, j, cidx1[i][j]);
         }exit(0);
     }exit(0);*/
     for(i=0; i<m; i++){
         for(j=0; j<n; j++){
             ridx1[i][j]=ridx[j];
             /*warn("ridx1[%d][%d]=%d", i, j, ridx1[i][j]);*/
         }
     }/*exit(0);*/
     for(i=0; i<m; i++){
         for(j=0; j<n; j++){
             t_flag[i][j]=cidx1[i][j]+ridx1[i][j];
             /*warn("ridx1[%d][%d]=%d", i, j, t_flag[i][j]);*/
         }
     }
     for(i=0; i<m; i++)
         for(j=0; j<n; j++)
             t[i][j]=x[t_flag[i][j]-1];

     free1float(c);
     free1float(r);
     free1float(k);
     free1float(x);
     free1int(cidx);
     free1int(ridx);
     free2int(cidx1);
     free2int(ridx1);
     free2int(t_flag);

}

static void cuda_cgsolver(float **a, float *rhs, int nrow, int ncol,
                     int niter, float error, float *x)
/***************************************************************************^M
 * least-squares linear system solver using CG^M
 * ****************************************************************************^M
 * a        array[][] of the matrix^M
 * rhs      array[] of the righ hand side^M
 * nrow     number of rows^M
 * ncol     number of columns^M
 * niter    number of CG iterations^M
 * error    thresholding relative error^M
 * x        least-squares solution^M
 * ***************************************************************************/
{
    float alpha1=1.0;/*parameters for Sgemv*/
    float beta1=0.0;

    float *h_A;
    /*b=rhs*/
    float *d_a,*d_b,*d_x;/*1d matrix in the device*/
    
    float *s, *r, *p, *q, *rn;
    int i, j, icount;
    float alpha, beta, alpha_minus;
    float rsum, qsum, rnsum, r0sum;

    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    h_A=alloc1float(nrow*ncol);
 
    /* original allocate spaces */
    /*s = alloc1float(nrow);
    r = alloc1float(ncol);
    p = alloc1float(ncol);
    q = alloc1float(nrow);
    rn = alloc1float(ncol);*/

    /*transfer matrix a into 1D vector*/
    for(i=0; i<nrow*ncol; i++)
        h_A[i]=0.0;
    for(i=0; i<nrow; i++)
        for(j=0; j<ncol; j++)
            h_A[i*ncol+j]=a[i][j];/*i*ncols+j*/

    cudaStat=cudaMalloc((void**)&d_a,nrow*ncol*sizeof(float));
    /*if(cudaStat!=cudaSuccess){
        printf("device memory allocation failed");
        return EXIT_FAILURE;
    }*/
    cudaStat=cudaMalloc((void**)&d_b,nrow*sizeof(float));
    /*if(cudaStat!=cudaSuccess){
        printf("device memory allocation failed");
        return EXIT_FAILURE;
    }*/
    cudaStat=cudaMalloc((void**)&d_x,ncol*sizeof(float));
    /*if(cudaStat!=cudaSuccess){
        printf("device memory allocation failed");
        return EXIT_FAILURE;
    }*/

    stat=cublasCreate(&handle);
    /*if(stat!=CUBLAS_STATUS_SUCCESS){
        printf("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }*/

    /*transfer data from host to device*/
    cudaMemcpy(d_a,h_A,nrow*ncol*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,rhs,nrow*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(d_x,0,ncol*sizeof(float));

    /*allocate cuda memory for parameters in cg process*/
    cudaStat=cudaMalloc((void**)&s,nrow*sizeof(float));
    /*if(cudaStat!=cudaSuccess){
        printf("device memory allocation failed");
        return EXIT_FAILURE;
    }*/
    cudaStat=cudaMalloc((void**)&r,ncol*sizeof(float));
    /*if(cudaStat!=cudaSuccess){
        printf("device memory allocation failed");
        return EXIT_FAILURE;
    }*/
    cudaStat=cudaMalloc((void**)&p,ncol*sizeof(float));
    /*if(cudaStat!=cudaSuccess){
        printf("device memory allocation failed");
        return EXIT_FAILURE;
    }*/
    cudaStat=cudaMalloc((void**)&q,nrow*sizeof(float));
    /*if(cudaStat!=cudaSuccess){
        printf("device memory allocation failed");
        return EXIT_FAILURE;
    }*/
    cudaStat=cudaMalloc((void**)&rn,ncol*sizeof(float));
    /*if(cudaStat!=cudaSuccess){
        printf("device memory allocation failed");
        return EXIT_FAILURE;
    }*/

    /* initialize the vectors */
    
    /* x = 0 */
    for(i=0; i<ncol; i++) x[i] = 0.;/*the vector used to store the result in host*/

    /* s = rhs */
    /*for(i=0; i<nrow; i++) s[i] = rhs[i];*/
    cublasScopy(handle,nrow,d_b,1,s,1);

    /* p = r = A^T s*/
    /*for(i=0; i<ncol; i++) p[i] = 0.;*/
    cudaMemset(p,0,ncol*sizeof(float));

    /*for(j=0; j<nrow; j++)
       for(i=0; i<ncol; i++)
          p[i] += a[j][i]*s[j];*/
    cublasSgemv(handle,CUBLAS_OP_T,nrow,ncol,&alpha1,d_a,ncol,s,1,&beta1,p,1);

    /*for(i=0; i<ncol; i++) r[i] = p[i];*/
    cublasScopy(handle,ncol,p,1,r,1);

    /*for(i=0, rsum=0.; i<ncol; i++) rsum += r[i]*r[i];*/
    rsum=0.0;
    cublasSdot(handle,ncol,r,1,r,1,&rsum);
    r0sum = rsum*error;

    /* start iteration */
    for(icount=0; icount < niter && rsum>r0sum; icount++)
    {
       /* q = A p */
       /*for(i=0; i<nrow; i++)
       {
          q[i] = 0.;
          for(j=0; j<ncol; j++)
             q[i] += a[i][j]*p[j];
       }*/
       cudaMemset(q,0,ncol*sizeof(float));
       cublasSgemv(handle,CUBLAS_OP_N,nrow,ncol,&alpha1,d_a,ncol,p,1,&beta1,q,1);

       /*for(i=0, qsum=0.; i<nrow; i++) qsum += q[i]*q[i];*/
       cublasSdot(handle,nrow,q,1,q,1,&qsum);

       /* alpha = r.r/q.q */
       alpha = rsum/qsum;
       alpha_minus=-alpha;    
 
       /* x = x + alpha*p, s = s - alpha*q */
       /*for(i=0; i<ncol; i++) x[i] += alpha*p[i];
       for(i=0; i<nrow; i++) s[i] -= alpha*q[i];*/
       cublasSaxpy(handle,ncol,&alpha,p,1,x,1);
       cublasSaxpy(handle,nrow,&alpha_minus,q,1,s,1);

       /* rn = A^T s */
       /*for(i=0; i<ncol; i++) rn[i] = 0.;
       for(j=0; j<nrow; j++)
          for(i=0; i<ncol; i++)
             rn[i] += a[j][i]*s[j];*/
       cudaMemset(rn,0,ncol*sizeof(float));
       cublasSgemv(handle,CUBLAS_OP_T,nrow,ncol,&alpha1,d_a,ncol,s,1,&beta1,rn,1);

       /*for(i=0, rnsum=0.; i<ncol; i++) rnsum += rn[i]*rn[i];*/
       cublasSdot(handle,ncol,rn,1,rn,1,&rsum);

       /* beta = rn.rn/r.r */
       beta = rnsum/rsum;

       /* p = rn + beta*p */
       /*for(i=0; i<ncol; i++) p[i] = rn[i] + beta*p[i];*/
       cublasSscal(handle,ncol,&beta,p,1);
       cublasSaxpy(handle,ncol,&alpha1,rn,1,p,1);

       /* r = rn */
       /*for(i=0; i<ncol; i++) r[i] = rn[i];*/
       cublasScopy(handle,ncol,rn,1,r,1);
       rsum = rnsum;
    }
    cudaMemcpy(x,d_x,ncol*sizeof(float),cudaMemcpyDeviceToHost);

    /* free the spaces of cuda program*/
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_x);
    cublasDestroy(handle);
    cudaFree(s);
    cudaFree(r);
    cudaFree(p);
    cudaFree(q);
    cudaFree(rn);

    free1float(h_A);

    /*original free*/
    /*free1float(s);
    free1float(r);
    free1float(p);
    free1float(q);
    free1float(rn);*/

}
