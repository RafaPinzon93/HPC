#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define SIZEx 3.0
#define SIZEy 3.0
#define BLOCKSIZE 16.0
#define TILE_WIDTH 4

__global__ void vecAdd(int *A, int *B, int *C, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int i = threadIdx.x;

            //blockIdx.x;
    if (i < n){
        C[i] = A[i] + B[i];
        // printf("%d. %d + %d = %d\n",i, A[i], B[i], C[i]);
    }
}

__global__ void MatrixSumKernel(int *d_M, int *d_N, int *d_P, int Width){
    int Row = blockIdx.x * blockDim.x + threadIdx.x;
    int Col = blockIdx.y * blockDim.y + threadIdx.y;
    if ((Row <= Width) && (Col <= Width)){
        d_P[Row*Width + Col] = d_M[Row*Width +  Col] + d_N[Row*Width +  Col];
        // printf("%d\n", d_P[Row*Width + Col]);
    }
  
}

__global__ void MatrixMulKernel(int *d_M, int *d_N, int *d_P, int Width){
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < Width) && (Col < Width)){
        int Pvalue = 0;
        int k;
        for (k =0; k<Width; ++k){
            Pvalue += d_M[Row*Width+k]*d_N[k*Width+Col];
        }
        d_P[Row*Width+Col] = Pvalue;
    }
}

__global__ void MatrixMulKernelSec(int *d_M, int *d_N, int *d_P, int pWidth){

    int lBx = blockIdx.x; int lBy = blockIdx.y;
    int lTx = threadIdx.x; int lTy = threadIdx.y;

    __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];
    // Identify the row and column of the d_P element to work on

    // int lRow = lBy * TILE_WIDTH + lTy;
    int lRow = lTx + lBx * blockDim.x;
    // int lCol = lBx * TILE_WIDTH + lTx;
    int lCol = lTy + lBy * blockDim.y;

    int Pvalue = 0;

    // Loop over the d_M and d_N tiles required to compute d_P element
    printf("%d / %d = %d\n", pWidth, TILE_WIDTH, pWidth/TILE_WIDTH);
    int m = 0;
    for (m = 0; m <= pWidth-1/TILE_WIDTH+1; ++m)
    {
        
        // Coolaborative loading of d_M and d_N tiles into shared memory
        if (lRow < pWidth && m*TILE_WIDTH+lTx < pWidth)
            Mds[lTy][lTx] = d_M[lRow*pWidth + m*TILE_WIDTH+lTx];
        else
            Mds[lTy][lTx] = 0;
        if (lCol < pWidth && m*TILE_WIDTH+lTy < pWidth)
            Nds[lTy][lTx] = d_N[(m*TILE_WIDTH+lTy)*pWidth+lCol];
        else
            Nds[lTy][lTx] = 0;
        Mds[lTy][lTx] = d_M[(lRow*pWidth) + (m*TILE_WIDTH) + lTy];
        Nds[lTy][lTx] = d_N[((m*TILE_WIDTH) + lTx)*pWidth + lCol];
        // printf("%d * %d\n", Mds[lTy][lT], Nds[lTy][lTx] );
        __syncthreads();
        int k;
        printf("Baba\n");
        for (k = 0; k < TILE_WIDTH; k++)
        {
            printf("%d * %d \n", Mds[lTx][k], Nds[k][lTy]);
            Pvalue += Mds[lTy][k] * Nds[k][lTx];
            // Pvalue += Mds[k][lTx] * Nds[lTy][k];
        }
        __syncthreads();
    }
    // 
    if (lRow < pWidth && lCol < pWidth)
        d_P[lRow*pWidth + lCol] = Pvalue;
       // C[Row*numCColumns+Col] = Pvalue;
    // d_P[lRow*pWidth + lCol] = Pvalue;

}

void MatrixSum(int *d_M, int *d_N, int *d_P, int Width){
    int i; 
  int n = sqrt(Width);
    for (i = 0; i < n; ++i)
    {
        /* code */
        int j;
        for (j = 0; j < n; ++j)
        {
            /* code */
            d_P[i*n + j] = d_M[i*n +  j] + d_N[i*n +  j];
        }
    }
  
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
    int dimGrid = ceil(SIZEx/BLOCKSIZE);
    printf(" DimGrid %d\n", dimGrid);
    vecAdd<<< dimGrid, BLOCKSIZE >>>(d_A, d_B, d_C, n);
    // vecAdd<<< DIMGRID, HILOS >>>
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
  
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

int matrixAddGPU( int *A, int *B, int *C, int n){
    int size = n*sizeof(int);
    int *d_A, *d_B, *d_C; n=sqrt(n);
    //Reservo Memoria en el dispositivo
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    //Copio los datos al device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    // Ejecuto el Kernel (del dispositivo)
    //int dimGrid = ceil(SIZE/BLOCKSIZE);
  
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 dimGrid(ceil(SIZEx/BLOCKSIZE),ceil(SIZEy/BLOCKSIZE) , 1);
  
    printf(" DimGrid %f\n", ceil(SIZEx/BLOCKSIZE));
    MatrixSumKernel<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, n);
    // vecAdd<<< DIMGRID, HILOS >>>
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

int MatrixMulGPU( int *A, int *B, int *C, int n){
    int size = n*sizeof(int);
    int *d_A, *d_B, *d_C; n=sqrt(n);
    //Reservo Memoria en el dispositivo
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    //Copio los datos al device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    // Ejecuto el Kernel (del dispositivo)
    //int dimGrid = ceil(SIZE/BLOCKSIZE);
  
    // dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    // dim3 dimGrid(ceil(n/BLOCKSIZE),ceil(n/BLOCKSIZE) , 1);
    dim3 dimGrid(ceil(n/TILE_WIDTH),ceil(n/TILE_WIDTH) , 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  
    printf(" DimGrid %f\n", ceil(SIZEx/BLOCKSIZE));
    MatrixMulKernelSec<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, n);
    // MatrixMulKernel<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, n);
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

    }
    return 0;
}

void ImprimirMatriz(int *pMatrix,int pRows, int pCols){
    int lRow, lColumn;
    for (lRow = 0; lRow < pRows; ++lRow)
    {
        for (lColumn = 0; lColumn < pCols; ++lColumn)
        {
            printf("%d ", pMatrix[lRow * pRows + lColumn]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(){
    int lTest;
    int lArrayTests[] = {4};
    int lNumberOfTests = sizeof(lArrayTests)/sizeof(lArrayTests[0]);
    for (lTest = 0; lTest < lNumberOfTests ; ++lTest)
    {
        int *A=(int *) malloc(lArrayTests[lTest]*sizeof(int));
        int *B=(int *) malloc(lArrayTests[lTest]*sizeof(int));
        int *C=(int *) malloc(lArrayTests[lTest]*sizeof(int));
        
        int Rows = sqrt(lArrayTests[lTest]);
        int Cols = sqrt(lArrayTests[lTest]);
        
        clock_t inicioCPU, inicioGPU,finCPU, finGPU;
        
        int lRandomNumberA;
        int lRandomNumberB;
        int i;

        printf("Tamano matriz %d\n", lArrayTests[lTest]);
        for(i=0 ; i< lArrayTests[lTest]; i++){
            lRandomNumberA = rand()%21;
            A[i] = lRandomNumberA;
            // printf("Random Number A %d A[%d]\n", lRandomNumberA, A[i]);
            lRandomNumberB = rand()%21;
            B[i]=lRandomNumberB;
            // printf("Random Number B %d B[%d]\n", lRandomNumberB, B[i]);
            // A[i]=srand(time(NULL));
            // B[i]=srand(time(NULL));

        }
        // Ejecuto por GPU
        inicioGPU=clock();
        // matrixAddGPU(A, B, C, lArrayTests[lTest]);
        MatrixMulGPU(A, B, C, lArrayTests[lTest]);
        finGPU = clock();
        // Ejecuto por CPU
        inicioCPU=clock();
        // MatrixSum(A, B, C, lArrayTests[lTest]);
        finCPU=clock();

        printf("Size %d\n", lArrayTests[lTest]);
        printf("El tiempo GPU es: %f\n",(double)(finGPU - inicioGPU) / CLOCKS_PER_SEC);
        printf("El tiempo CPU es: %f\n",(double)(finCPU - inicioCPU) / CLOCKS_PER_SEC);
    
        ImprimirMatriz(A, Rows, Cols);
        ImprimirMatriz(B, Rows, Cols);
        ImprimirMatriz(C, Rows, Cols);

    
        free(A);
        free(B);
    }
    return 0;
}
