/*
Rafael Pinzón Rivera 1088313004
*/

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#define USECPSEC 1000000ULL

#define BLOCKSIZE 256
#define TILE_WIDTH 256
#define MAX_MASK_WIDTH 10

typedef int dataType;

unsigned long long dtime_usec(unsigned long long prev){
  timeval tv1;
  gettimeofday(&tv1,0);
  return ((tv1.tv_sec * USECPSEC)+tv1.tv_usec) - prev;
  // return ((tv1.tv_sec *1000)+tv1.tv_usec/1000) - prev;
}

__constant__ dataType gM[MAX_MASK_WIDTH];

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
    // if (Pvalue != 0) 
        // printf("\nPValue[%d] = %d\n", i, Pvalue);
    P[i] = Pvalue;
}

__global__ void Convolution1D_Constant(dataType *N, dataType *P, int Mask_Width, int Width){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    dataType Pvalue = 0.0;
    int N_start_point = i - (Mask_Width/2);
    for (int j = 0; j < Mask_Width; ++j)
    {
        if (N_start_point + j >= 0 && (N_start_point + j) < Width)
        {
            Pvalue += N[N_start_point + j] * gM[j];
            // if (N[N_start_point + j] != 0 && M[j] != 0)
                // printf("%d ", Pvalue);
        }
    }
    // if (Pvalue != 0) 
        // printf("\nPValue[%d] = %d\n", i, Pvalue);
    P[i] = Pvalue;
}

__global__ void Convolution1D_Tiled(dataType *N, dataType *P, int Mask_Width, int Width){
    __shared__ dataType N_ds[TILE_WIDTH + MAX_MASK_WIDTH -1];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int midMask = Mask_Width/2;

    int halo_index_left = (blockIdx.x - 1) *blockDim.x + threadIdx.x;
    if (threadIdx.x >= blockDim.x - midMask){
        N_ds[threadIdx.x - (blockDim.x - midMask)] = (halo_index_left < 0) ? 0 : N[halo_index_left];
    }

    N_ds[midMask + threadIdx.x] = N[blockIdx.x * blockDim.x + threadIdx.x];

    int halo_index_right = (blockIdx.x + 1) *blockDim.x + threadIdx.x;
    if (threadIdx.x >= blockDim.x - midMask){
        N_ds[threadIdx.x + blockDim.x + midMask] = (halo_index_right < 0) ? 0 : N[halo_index_right];
    }

    dataType Pvalue = 0.0;
    for (int j = 0; j < Mask_Width; ++j)
    {
        Pvalue += N_ds[threadIdx.x + j] * gM[j];
    }
    // if (Pvalue != 0) 
        // printf("\nPValue[%d] = %d\n", i, Pvalue);
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
    unsigned long long gpu_time = dtime_usec(0);


    switch (Option){
        case 1:
            cudaMemcpy(d_B, B, MaskSize, cudaMemcpyHostToDevice);
            Convolution1D_Basic<<< dimGrid1D, TILE_WIDTH >>>(d_A, d_B, d_C, pMask_Width, n);
            cudaDeviceSynchronize();
            gpu_time = dtime_usec(gpu_time);
            printf("Finished 1. Basic.  Results match. gpu time: %lldus\n", gpu_time);
            break;
        case 2:
            cudaMemcpyToSymbol(gM, B, MaskSize);
            Convolution1D_Constant<<< dimGrid1D, TILE_WIDTH >>>(d_A, d_C, pMask_Width, n);
            gpu_time = dtime_usec(gpu_time);
            printf("Finished 2. Constant.  Results match. gpu time: %lldus\n", gpu_time);

            break;
        case 3:
            cudaMemcpyToSymbol(gM, B, MaskSize);
            Convolution1D_Constant<<< dimGrid1D, TILE_WIDTH >>>(d_A, d_C, pMask_Width, n);
            gpu_time = dtime_usec(gpu_time);
            printf("Finished 3. Constant Tiled.  Results match. gpu time: %lldus\n", gpu_time);
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
    int SIZES[] = {51, 101, 501, 1001, 5001};
    int Mask_Width = 5;
    for (j = 0; j < sizeof(SIZES)/sizeof(SIZES[0]); ++j)
    {
        printf("Size %d\n", SIZES[j]);
        dataType *A=(dataType *) malloc(SIZES[j]*sizeof(dataType));
        dataType *hA=(dataType *) malloc(SIZES[j]*sizeof(dataType));
        dataType *B=(dataType *) malloc(Mask_Width*sizeof(dataType)); 
        dataType *hB=(dataType *) malloc(Mask_Width*sizeof(dataType));
        dataType *C=(dataType *) malloc(SIZES[j]*sizeof(dataType));
        dataType *result=(dataType *) malloc(SIZES[j]*sizeof(dataType));
        // clock_t inicioCPU, inicioGPU,finCPU, finGPU;
        // printf("A ");
        for (int iVector = 0; iVector < SIZES[j]; ++iVector)
        {
            A[iVector]=1;
            hA[iVector]=A[iVector];

            // printf("%f ", A[iVector]);
        }
        // printf("\nB ");
        for (int iVector = 0; iVector < Mask_Width; ++iVector)
        {
            B[iVector]=1;
            hB[iVector]=B[iVector];

            // printf("%f ", B[iVector]);
        }
        printf("\n");


        // Ejecuto por GPU
        // inicioGPU=clock();
        InvoqueKernel(A, B, C, SIZES[j], Mask_Width, 1);
        // finGPU = clock();
        InvoqueKernel(A, B, C, SIZES[j], Mask_Width, 2);
        InvoqueKernel(A, B, C, SIZES[j], Mask_Width, 3);


        // printf("\nResult ");

        free(A);
        free(B);
    }
    return 0;
}
