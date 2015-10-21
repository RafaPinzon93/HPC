/*
Rafael Pinzón Rivera 1088313004
*/

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#define SIZE 1024

__global__ void vecAdd(int *A, int *B, int *C, int n){
	int i = threadIdx.x;
			//blockIdx.x*blockDim + threadIdx.x;
  if (i < n){
		C[i] = A[i] + B[i];
	    printf("%d. %d + %d = %d\n",i, A[i], B[i], C[i]);
  }
}
int vectorAdd( int *A, int *B, int *C, int n){
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
	vecAdd<<< 1, n >>>(d_A, d_B, d_C, n);
	// vecAdd<<< n, 1 >>>(d_A, d_B, d_C, n);
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}

int main(){
	int *A=(int *) malloc(SIZE*sizeof(int));
	
	int *B=(int *) malloc(SIZE*sizeof(int));
	
	int *C=(int *) malloc(SIZE*sizeof(int));
	time_t inicio,fin;
	inicio=time(NULL);
	int i;
	for(i=0;i< SIZE; i++){
		A[i]=rand()%21;
		B[i]=rand()%21;
		// A[i]=srand(time(NULL));
		// B[i]=srand(time(NULL));
	
	}
	vectorAdd(A, B, C, SIZE);
	fin=time(NULL);
	printf("El tiempo es: %f\n",difftime(fin,inicio));
	free(A);
	free(B);
	return 0;
}