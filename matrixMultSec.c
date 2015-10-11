#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define SIZEx 2.0
#define SIZEy 2.0
#define BLOCKSIZE 64.0
#define TILE_WIDTH 32

typedef struct udtMatrix {
        int Rows;
      int Cols;
}udtMatrix;

typedef struct udtTest {
        udtMatrix A;
      udtMatrix B;
}udtTest;

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
    __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

    int lBx = blockIdx.x; int lBy = blockIdx.y;
    int lTx = threadIdx.x; int lTy = threadIdx.y;

    // Identify the row and column of the d_P element to work on

     int lRow = lBy * TILE_WIDTH + lTy;
    //int lRow = lTx + lBx * blockDim.x;
     int lCol = lBx * TILE_WIDTH + lTx;
    //int lCol = lTy + lBy * blockDim.y;

    float Pvalue = 0;

    // Loop over the d_M and d_N tiles required to compute d_P element
    //printf("%d\n", pWidth/TILE_WIDTH);
    int lTamanho=(pWidth/TILE_WIDTH);
    // printf("%d\n", lTamanho);
    if(lTamanho==0) lTamanho = 1;
    int m;
    for (m = 0; m < lTamanho; ++m)
    {
       
        // Coolaborative loading of d_M and d_N tiles into shared memory
        if (lRow < pWidth && m*TILE_WIDTH+lTx < pWidth)
            Mds[lTy][lTx] = d_M[(lRow*pWidth) + (m*TILE_WIDTH) + lTx];
        else
            Mds[lTy][lTx] = 0;
        if (lCol < pWidth && m*TILE_WIDTH+lTy < pWidth)
            Nds[lTy][lTx] = d_N[((m*TILE_WIDTH) + lTy)*pWidth + lCol];
        else
            Nds[lTy][lTx] = 0;
        //printf("%d * %d\n", Mds[lTy][lTx], Nds[lTy][lTx] );
        __syncthreads();
        int k;
        for (k = 0; k < TILE_WIDTH; ++k)
        {
             //printf("%d * %d \n", Mds[k][lTx], Nds[lTy][k]);
             Pvalue += Mds[lTy][k] * Nds[k][lTx];
            //Pvalue += Mds[k][lTx] * Nds[lTy][k];
        }
        __syncthreads();
    }
    if (lRow < pWidth && lCol < pWidth)
        d_P[lRow*pWidth + lCol] = Pvalue;

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

int MatrixMulGPU( int *A, int *B, int *C, int n, int pTiled = 0){
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
 
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(ceil(n/TILE_WIDTH),ceil(n/TILE_WIDTH) , 1);
 
    //printf(" DimGrid %f\n", ceil(SIZEx/BLOCKSIZE));
    // printf("%d\n", n);
    if (pTiled){
        printf("Tiled\n");
        MatrixMulKernelSec<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, n);
    }
    else{
        printf("Not Tiled\n");
        MatrixMulKernel<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, n); 
    }
    // MatrixMulKernel<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, n);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}


int MatrixMulCPU(int *prA, int *prB, int *prC, int prN){
    int lRadio = sqrt(prN);
    int lRowValue = 0;
    for (int i = 0; i < lRadio; ++i)
    {
        for (int j = 0; j < lRadio; ++j)
        {
            for (int k = 0; k < lRadio; ++k)
            {   
                lRowValue += prA[i*lRadio + k] * prB[k*lRadio + j];
            }
            prC[i*lRadio + j] = lRowValue;
            lRowValue = 0;
        }
    }
    return 0;
}

int vectorAddCPU( int *A, int *B, int *C, int n){
    int i;
    for(i=0;i< n; i++){
        C[i]=A[i]+B[i];
    }
    return 0;
}

void ImprimirMatriz(int *pMatrix, int pRows, int pCols){
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
    udtMatrix A;
    udtMatrix B;
    A.Rows = 2;
    A.Cols = 2;
    B.Rows = 2;
    B.Cols = 2;
    
    udtTest test;
    test.A = A;
    test.B = B;
  
    udtTest lArrayTests[] = {test};
    
      
    int lNumberOfTests = sizeof(lArrayTests)/sizeof(lArrayTests[0]);
    printf("%d", lNumberOfTests);
      
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)
    
      for ( lTest = 0; lTest < lNumberOfTests ; ++lTest)
    {
        numARows = lArrayTests[lTest].A.Rows;
        numAColumns = lArrayTests[lTest].A.Cols;
        numBRows = lArrayTests[lTest].B.Rows;
        numBColumns = lArrayTests[lTest].B.Cols;
        numCRows = lArrayTests[lTest].A.Rows;
        numCColumns = lArrayTests[lTest].B.Cols;
     
        int *A=(int *) malloc(lArrayTests[lTest].A.Rows*lArrayTests[lTest].A.Cols*sizeof(int));
        int *B=(int *) malloc(lArrayTests[lTest].B.Rows*lArrayTests[lTest].B.Cols*sizeof(int));
        int *C=(int *) malloc(lArrayTests[lTest].A.Rows*lArrayTests[lTest].B.Cols*sizeof(int));
       
        //int Rows = sqrt(lArrayTests[lTest]);
        //int Cols = sqrt(lArrayTests[lTest]);
       
        clock_t inicioCPU, inicioGPU, inicioGPU_Tiled,finCPU, finGPU, finGPU_Tiled;
       
        int lRandomNumberA;
        int lRandomNumberB;
        int i;

        //printf("Tamano matriz %d\n", lArrayTests[lTest]);
        for(i=0 ; i< lArrayTests[lTest].A.Rows*lArrayTests[lTest].A.Cols; i++){
            lRandomNumberA = rand()%21;
            A[i] = lRandomNumberA;
            // printf("Random Number A %d A[%d]\n", lRandomNumberA, A[i]);
            lRandomNumberB = rand()%21;
            B[i]=lRandomNumberB;
            // printf("Random Number B %d B[%d]\n", lRandomNumberB, B[i]);
            // A[i]=srand(time(NULL));
            // B[i]=srand(time(NULL));

        }

         // Ejecuto por CPU
        inicioCPU=clock();
        MatrixMulCPU(A, B, C, lArrayTests[lTest].A.Rows*lArrayTests[lTest].A.Cols);
        finCPU=clock();
        // Ejecuto por GPU
        inicioGPU=clock();
        MatrixMulGPU(A, B, C, lArrayTests[lTest].A.Rows*lArrayTests[lTest].A.Cols, 0);
        finGPU = clock();
        // Ejecuto por GPU con Tiles
        inicioGPU_Tiled = clock();
        MatrixMulGPU(A, B, C, lArrayTests[lTest].A.Rows*lArrayTests[lTest].A.Cols, 1);
        finGPU_Tiled = clock();
       

        //printf("Size %d\n", lArrayTests[lTest]);
        printf("El tiempo CPU es: %f\n",(double)(finCPU - inicioCPU) / CLOCKS_PER_SEC);
        printf("El tiempo GPU es: %f\n",(double)(finGPU - inicioGPU) / CLOCKS_PER_SEC);
        printf("El tiempo GPU Tiled es: %f\n",(double)(finGPU_Tiled - inicioGPU_Tiled) / CLOCKS_PER_SEC);
   
        ImprimirMatriz(A, lArrayTests[lTest].A.Rows, lArrayTests[lTest].A.Cols);
        ImprimirMatriz(B, lArrayTests[lTest].A.Rows, lArrayTests[lTest].A.Cols);
        ImprimirMatriz(C, lArrayTests[lTest].A.Rows, lArrayTests[lTest].A.Cols);

   
        free(A);
        free(B);
    }
    printf("Tile Width %d", TILE_WIDTH);
    return 0;
}
