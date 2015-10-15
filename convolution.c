/*
Rafael Pinzón Rivera 1088313004
*/

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define BLOCKSIZE 1024

__global__ void Convolution1D_Basic(int *N, int *M, int *P, int Mask_Width, int Width){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float Pvalue = 0;
    int N_start_point = i - (Mask_Width/2);
    for (int j = 0; j < Mask_Width; ++j)
    {
        if (N_start_point + j >= 0 && N_start_point + j < Width)
        {
            Pvalue += N[N_start_point + j] * M[j]
        }
    }
    P[i] = Pvalue
}

int InvoqueKernel(int *A, int *B, int *C, int n, int Option){
    /* 
        Función que invoca un método en el kernel (GPU) 
        Parametros:
            A: Vector de enteros.
            B: Vector de enteros a operar con A.
            C: Vector de enteros, para guardar la solución.
            Option: Saber que función va a llamar.
    */
    int *d_A, *d_B, *d_C;
    int Mask_Width = 2*n + 1;
    //Reservo Memoria en el dispositivo
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_C, size);
    //Copio los datos al device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    // Ejecuto el Kernel (del dispositivo)
    int dimGrid = ceil(SIZE/BLOCKSIZE);
    printf(" DimGrid %d\n", dimGrid);
    switch (Option){
        case 1:
            cudaMalloc((void **)&d_B, size);
            cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
            sumaCurrency<<< 1, BLOCKSIZE >>>(d_A, d_C);
            break;
        case 2:
            break;
        case 3:
            break;
    }
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}


int main(){
    int j;
    int SIZES[] = {4};
    for (j = 0; j < sizeof(SIZES)/sizeof(SIZES[0]); ++j)
    {
        int *A=(int *) malloc(SIZES[j]*sizeof(int));
        int *B=(int *) malloc(SIZES[j]*sizeof(int));
        int *C=(int *) malloc(SIZES[j]*sizeof(int));
        // clock_t inicioCPU, inicioGPU,finCPU, finGPU;
        int i;
        for(i=0;i< SIZES[j]; i++){
            A[i]=1;
            // A[i]=rand()%21;
            B[i]=i;
            // B[i]=rand()%21;
        }
        // Ejecuto por GPU
        inicioGPU=clock();
        InvoqueKernel(A, B, C, SIZES[j]);
        finGPU = clock();
        // Ejecuto por CPU
        inicioCPU=clock();
        // vectorAddCPU(A, B, C, SIZES[j]);
        finCPU=clock();
        printf("Size %d\n", SIZES[j]);
        for (int iVector = 0; iVector < SIZES[j]; ++iVector)
        {
            printf("%d\n", C[i]);
        }
        // printf("El tiempo GPU es: %f\n",(double)(finGPU - inicioGPU) / CLOCKS_PER_SEC);
        // printf("El tiempo CPU es: %f\n",(double)(finCPU - inicioCPU) / CLOCKS_PER_SEC);
        free(A);
        free(B);
    }
    return 0;
}
