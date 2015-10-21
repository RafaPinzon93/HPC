/*
Rafael Pinzón Rivera 1088313004
*/

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#define BLOCKSIZE 256
#define TILE_WIDTH 256

typedef int dataType;

void conv(const dataType *A, const dataType *B, dataType* out, int N, int nMask) {
  int N_start_point;
  int i ;
  for ( i = 0; i < N; ++i)
      N_start_point = i - (nMask/2);
      out[i] = 0;
        for (int j = 0; j < nMask; ++j)
            if (N_start_point + j >= 0 && (N_start_point + j) < N )
                out[i] += A[i] * B[j];
}

__global__ void Convolution1D_Basic(dataType *N, dataType *M, dataType *P, int Mask_Width, int Width){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    dataType Pvalue = 0.0;
    int N_start_point = i - (Mask_Width/2);
    for (int j = 0; j < Mask_Width; ++j)
    {
        if (N_start_point + j >= 0 && (N_start_point + j) < Width)
        {
            Pvalue += N[N_start_point + j] * M[j];
            // if (N[N_start_point + j] != 0 && M[j] != 0)
                // printf("%d ", Pvalue);
        }
    }
    if (Pvalue != 0) 
        printf("\nPValue[%d] = %d\n", i, Pvalue);
    P[i] = Pvalue;
}

int InvoqueKernel(dataType *A, dataType *B, dataType *C, int n, int pMask_Width, int Option){
    /* 
        Función que invoca un método en el kernel (GPU) 
        Parametros:
            A: Vector de enteros.
            B: Vector de enteros a operar con A.
            C: Vector de enteros, para guardar la solución.
            Option: Saber que función va a llamar.
    */
    dataType *d_A, *d_B, *d_C;
    int sizeN = n*sizeof(dataType);
    int MaskSize = pMask_Width*sizeof(dataType);
    // int Mask_Width = 2*sizeN + 1;
    //Reserve memory in the device
    cudaMalloc((void **)&d_A, sizeN);
    cudaMalloc((void **)&d_B, MaskSize);
    cudaMalloc((void **)&d_C, sizeN);
    //Copy the data to the device memory
    cudaMemcpy(d_A, A, sizeN, cudaMemcpyHostToDevice);

    int dimGrid1D = ceil((float)sizeN/(float)MaskSize);

    dim3 dimBlock3D(TILE_WIDTH, TILE_WIDTH, 1);
    int dimGridX = (int)ceil(sizeN/(dataType)dimBlock3D.x);
    int dimGridY = (int)ceil(sizeN/(dataType)dimBlock3D.y);
    dim3 dimGrid3D(dimGridX, dimGridY, 1);

    switch (Option){
        case 1:
            cudaMemcpy(d_B, B, MaskSize, cudaMemcpyHostToDevice);
            Convolution1D_Basic<<< dimGrid1D, TILE_WIDTH >>>(d_A, d_B, d_C, pMask_Width, n);
            cudaDeviceSynchronize();
            break;
        case 2:
            break;
        case 3:
            break;
    }
    cudaMemcpy(C, d_C, sizeN, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}


int main(){
    int j;
    int SIZES[] = {7};
    int Mask_Width = 5;
    for (j = 0; j < sizeof(SIZES)/sizeof(SIZES[0]); ++j)
    {
        dataType *A=(dataType *) malloc(SIZES[j]*sizeof(dataType));
        dataType *hA=(dataType *) malloc(SIZES[j]*sizeof(dataType));
        dataType *B=(dataType *) malloc(Mask_Width*sizeof(dataType));
        dataType *hB=(dataType *) malloc(Mask_Width*sizeof(dataType));
        dataType *C=(dataType *) malloc(SIZES[j]*sizeof(dataType));
        dataType *result=(dataType *) malloc(SIZES[j]*sizeof(dataType));
        clock_t inicioCPU, inicioGPU,finCPU, finGPU;
        printf("A ");
        for (int iVector = 0; iVector < SIZES[j]; ++iVector)
        {
            A[iVector]=1;
            hA[iVector]=A[iVector];
            printf("%d ", A[iVector]);
        }
        printf("\nB ");
        for (int iVector = 0; iVector < Mask_Width; ++iVector)
        {
            B[iVector]=1;
            hB[iVector]=B[iVector];
            printf("%d ", B[iVector]);
        }
        printf("\n");


        // Ejecuto por GPU
        inicioGPU=clock();
        InvoqueKernel(A, B, C, SIZES[j], Mask_Width, 1);
        finGPU = clock();


        // Ejecuto por CPU
        inicioCPU=clock();
        conv(hA, hB, result, SIZES[j], Mask_Width);
        finCPU=clock();


        printf("Size %d\n", SIZES[j]);
        printf("\nResult ");
        for (int iVector = 0; iVector < SIZES[j]; ++iVector)
        {
            printf("%d ", result[iVector]);
        }
        printf("\n");
        printf("\nC ");
        for (int iVector = 0; iVector < SIZES[j]; ++iVector)
        {
            printf("%d ", C[iVector]);
        }
        printf("\n");
        printf("El tiempo GPU es: %f\n",(double)(finGPU - inicioGPU) / CLOCKS_PER_SEC);
        printf("El tiempo CPU es: %f\n",(double)(finCPU - inicioCPU) / CLOCKS_PER_SEC);
        free(A);
        free(B);
    }
    return 0;
}
