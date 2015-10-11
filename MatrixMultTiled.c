#include "malloc.h"
#include "time.h"
#include "stdio.h"
#include "cuda.h"
#include "stdlib.h"
#define ARow 2
#define ACBR 2
#define BCol 2
#define TILE_WIDTH 32
#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )
# define BLOCK_DIM 30

static void HandleError( cudaError_t err, const char *file, int line )
{
  if (err != cudaSuccess)
  {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
           file, line );
    exit( EXIT_FAILURE );
  }
}

__global__ void MatrixMulKernelTile(float* A, float* B, float* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {

    float CValue = 0;
		
  	int lBx = blockIdx.x; int lBy = blockIdx.y;
    int lTx = threadIdx.x; int lTy = threadIdx.y;
  	
    int Row = lBy*TILE_WIDTH + lTy;
    int Col = lBx*TILE_WIDTH + lTx;

    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    for (int k = 0; k < (TILE_WIDTH + ACols -1)/TILE_WIDTH; k++) {

         if (k*TILE_WIDTH + lTx < ACols && Row < ARows)   
           As[lTy][lTx] = A[Row*ACols + k*TILE_WIDTH + lTx];
         else                                                   
           As[lTy][lTx] = 0.0;

         if (k*TILE_WIDTH + lTy < BRows && Col < BCols)   
           Bs[lTy][lTx] = B[(k*TILE_WIDTH + lTy)*BCols + Col];
         else                                                   
           Bs[lTy][lTx] = 0.0;
      	 printf("%d * %d", As[lTy][lTx], Bs[lTy][lTx] );

         __syncthreads();

      for (int n = 0; n < TILE_WIDTH; ++n) {
        printf("%d * %d", As[lTy][lTx], Bs[lTy][lTx] );
           CValue += As[lTy][n] * Bs[n][lTx];
      }
         __syncthreads();
    }

    if (Row < CRows && Col < CCols) 
      C[Row*CCols+Col] = CValue;
  		//C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
}




__global__ void MatrixMulKernel(float *d_M, float *d_N, float *d_P, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols){
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < ARows) && (Col < BCols)){
        int Pvalue = 0;
        int k;
        for (k =0; k<ACols; ++k){
            Pvalue += d_M[Row*ACols+k]*d_N[k*BCol+Col];
        }
        d_P[Row*CCols+Col] = Pvalue;
    }
}


void MatrixMulCPU(float *d_M, float *d_N, float *d_P){
  int i;
  for (i=0;i<ARow;i++){
     for (int j=0;j<BCol;j++){
      for (int k=0;k<ACBR;k++){
      	d_P[i*BCol + j] += d_M[i*ACBR+k]*d_N[k*BCol+j];
      }
     }
   }
  
}




void initialize(float *vec, int rows, int cols)
{
  int i;printf("Reach\n");

 	srand(time(NULL));
  for ( i = 0; i< rows*cols; i++)
  {
  	vec[i] = 1.0;//rand() % (1+10-0) + 0;
  }
}

void printLast(float *c)
{
   int i;
  for ( i = ARow*BCol-5; i < ARow*BCol ; i++) // ROW * ROW
  { 
  	printf("%d = %f\n",i,c[i]);
  } 
}

main () 
{
  float *A,*B,*C;
  float *d_a,*d_b,*d_c;
  clock_t inicioCPU, inicioGPU,finCPU, finGPU;

  A = NULL;
  B = NULL;
  C = NULL;
  A = (float *) malloc ( sizeof(float) * ARow*ACBR);
  B = (float *) malloc ( sizeof(float) * ACBR*BCol);
  C = (float *) malloc ( sizeof(float) * ARow*BCol);
  initialize(A, ARow, ACBR);
  initialize(B, ACBR, BCol);
  //Allocate the memory on the GPU
  printf("Reach\n");

  HANDLE_ERROR ( cudaMalloc((void **)&d_a , ARow*ACBR*sizeof(float) ) );
  HANDLE_ERROR ( cudaMalloc((void **)&d_b , ACBR*BCol*sizeof(float) ) );
  HANDLE_ERROR ( cudaMalloc((void **)&d_c , ARow*BCol*sizeof(float) ) );
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH,1);
  dim3 dimGrid((int)ceil((float)BCol/(float)dimBlock.x),(int)ceil((float)ARow/(float)dimBlock.y),1);

 //Copy Host array to Device array
  HANDLE_ERROR (cudaMemcpy (d_a , A , ARow*ACBR*sizeof(float) , cudaMemcpyHostToDevice));
  HANDLE_ERROR (cudaMemcpy (d_b , B , ACBR*BCol*sizeof(float) , cudaMemcpyHostToDevice));


  //Make a call to GPU kernel
  //matrix_Multiplication_Tiles <<< dimGrid, dimBlock  >>> (dev_a , dev_b , dev_c ) ;
  inicioGPU=clock();
	MatrixMulKernelTile <<< dimGrid,dimBlock >>> (d_a,d_b,d_c, ARow,ACBR,ACBR,BCol,ARow,BCol);
  //MatrixMulKernel <<< dimGrid,dimBlock >>> (d_a,d_b,d_c, ARow,ACBR,ACBR,BCol,ARow,BCol);
  finGPU = clock();
  
  
  inicioCPU=clock();
  MatrixMulCPU(A, B, C);
  finCPU=clock();

  //Copy back to Host array from Device array
  //HANDLE_ERROR (cudaMemcpy(C , d_c , ARow*BCol*sizeof(float) , cudaMemcpyDeviceToHost));

  printLast(C);
  //time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  //printf("Se ha demorado %f segundos.\n",time_spent);
  printf("El tiempo GPU es: %f\n",(double)(finGPU - inicioGPU) / CLOCKS_PER_SEC);
  printf("El tiempo CPU es: %f\n",(double)(finCPU - inicioCPU) / CLOCKS_PER_SEC);
  free(A);
  free(B);
  free(C);
}
