/*
Rafael Pinzón Rivera 1088313004
*/

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define SIZE 2000
#define BLOCKSIZE 1

__global__ void vecAdd(int *A, int *B, int *C, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//int i = threadIdx.x;

			//blockIdx.x;
  if (i < n){
		C[i] = A[i] + B[i];
	    // printf("%d. %d + %d = %d\n",i, A[i], B[i], C[i]);
  }
}

__global__ void scan(float *g_odata, float *g_idata, int n)

	__shared__ float temp[]; // allocated on invocation
	 int thid = threadIdx.x;
	int1 pout = 0, pin = 1;
	// Load input into shared memory.
	 // This is exclusive scan, so shift right by one
	 // and set first element to 0
	temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;
	__syncthreads();
	for (int offset = 1; offset < n; offset *= 2)
	{
	  pout = 1 - pout; // swap double buffer indices
	  pin = 1 - pout;
	  if (thid >= offset)
	    temp[pout*n+thid] += temp[pin*n+thid - offset];
	  else
	    temp[pout*n+thid] = temp[pin*n+thid];
	  __syncthreads();
	}
	g_odata[thid] = temp[pout*n+thid1]; // write output
}

__global__ void sumaCurrency(int *g_idata, int *g_odata){
	extern __shared__ int sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s=1; s < blockDim.x; s *= 2) {
		if (tid % (2*s) == 0)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
		// write result for this block to global mem
	if
	(tid == 0) g_odata[blockIdx.x] = sdata[0];
}


int vectorAddGPU( int *A, int *B, int *C, int n){
	int size = n*sizeof(int);
	int *d_A, *d_B, *d_C;
	//Reservo Memoria en el dispositivo
	cudaMalloc((void **)&d_A, size);
	cudaMalloc((void **)&d_B, size);
	cudaMalloc((void **)&d_C, size);
	//Copio los datos al device
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	// Ejecuto el Kernel (del dispositivo)
	int dimGrid = ceil(SIZE/BLOCKSIZE);
	printf(" DimGrid %d\n", dimGrid);
	vecAdd<<< dimGrid, BLOCKSIZE >>>(d_A, d_B, d_C, n);
	// vecAdd<<< HILOS, BLOCKES >>>
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}
int vectorAddGPUCurrency( int *A, int *B, int *C, int n){
	int size = n*sizeof(int);
	int *d_A, *d_B, *d_C;
	//Reservo Memoria en el dispositivo
	cudaMalloc((void **)&d_A, size);
	cudaMalloc((void **)&d_B, size);
	cudaMalloc((void **)&d_C, size);
	//Copio los datos al device
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	// Ejecuto el Kernel (del dispositivo)
	int dimGrid = ceil(SIZE/BLOCKSIZE);
	printf(" DimGrid %d\n", dimGrid);
	sumaCurrency<<< 1, BLOCKSIZE >>>(d_A, d_C);
	// scan<<< 1, BLOCKSIZE >>>(d_A, d_C, 4);
	// vecAdd<<< HILOS, BLOCKES >>>
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}

int vectorAddCPU( int *A, int *B, int *C, int n){
	int i;
	for(i=0;i< n; i++){
		C[i]=A[i]+B[i];
		//printf("%d. %d+", i, A[i]);
    //printf("%d=",B[i]);
		//printf("%d\n",C[i]);
	}
	return 0;
}

int main(){
	int j;
	int SIZES[] = {512, 1024, 3000, 5000, 1000000, 5000000};
	for (j = 0; j < sizeof(SIZES)/sizeof(SIZES[0]); ++j)
	{
		int *A=(int *) malloc(SIZES[j]*sizeof(int));
		int *B=(int *) malloc(SIZES[j]*sizeof(int));
		int *C=(int *) malloc(SIZES[j]*sizeof(int));
		clock_t inicioCPU, inicioGPU,finCPU, finGPU;
		int i;
		for(i=0;i< SIZES[j]; i++){
			A[i]=rand()%21;
			B[i]=rand()%21;
			// A[i]=srand(time(NULL));
			// B[i]=srand(time(NULL));

		}
		// Ejecuto por GPU
		inicioGPU=clock();
		vectorAddGPU(A, B, C, SIZES[j]);
		finGPU = clock();
		// Ejecuto por CPU
		inicioCPU=clock();
		vectorAddCPU(A, B, C, SIZES[j]);
		finCPU=clock();
		printf("Size %d\n", SIZES[j]);
		printf("El tiempo GPU es: %f\n",(double)(finGPU - inicioGPU) / CLOCKS_PER_SEC);
		printf("El tiempo CPU es: %f\n",(double)(finCPU - inicioCPU) / CLOCKS_PER_SEC);
		free(A);
		free(B);
	}
	return 0;
}
