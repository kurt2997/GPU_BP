/* Copyright: Ruo Wang, Centre for Reservoir Geophysics, Imperial College London, 2013 */
/* All rights reserved.                                                      */

#include <su.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <signal.h>
#include <header.h>
#include <segy.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int cuda_mtrx_multiply(int nrow1, int ncol1, float **A, int nrow2, int ncol2, float **B, float **C);

int cuda_mtrx_multiply(int nrow1, int ncol1, float **A, int nrow2, int ncol2, float **B, float **C)
{
    float alpha=1.0;
    float beta=0.0;
    int nrow1, ncol1, nrow2, ncol2;

    int i, j;
    float *h_A, *h_B, *h_C;
    float *d_a,*d_b,*d_c;/*1d matrix in the device*/

    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    nrow1=3;
    ncol1=2;
    nrow2=2;
    ncol2=2;

    stat=cublasInit();
    if(stat!=CUBLAS_STATUS_SUCCESS){
        printf("CUBLAS init error!");
        return EXIT_FAILURE;
    }


    h_A=alloc1float(nrow1*ncol1);
    h_B=alloc1float(nrow2*ncol2);
    h_C=alloc1float(nrow1*ncol2);

    for(i=0; i<nrow1; i++)
        for(j=0; j<ncol1; j++)
            A[i][j]=i+1;
    for(i=0; i<nrow2; i++)
        for(j=0; j<ncol2; j++)
            B[i][j]=1.0;
    for(i=0; i<nrow1; i++)
        for(j=0; j<ncol2; j++)
            C[i][j]=0.0;

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
    for(i=0; i<nrow1; i++)
        for(j=0; j<ncol2; j++)
            printf("C[%d][%d]=%f\n", i, j, C[i][j]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);

    free1float(h_A);
    free1float(h_B);
    free1float(h_C);

    stat=cublasShutdown();
    if(stat!=CUBLAS_STATUS_SUCCESS){
        printf("Shutdown error\n");
        return EXIT_FAILURE;
    }

    return 1;
}

