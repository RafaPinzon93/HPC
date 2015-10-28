/*
Rafael Pinzón Rivera 1088313004
*/

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <iostream>
#include <highgui.h>
#include <cv.h>


#define USECPSEC 1000000ULL

#define BLOCKSIZE 256
#define TILE_WIDTH 32
#define MAX_MASK_WIDTH 9

using namespace std;
using namespace cv;

typedef unsigned char dataType;
typedef char dataType2;

unsigned long long dtime_usec(unsigned long long prev){
  timeval tv1;
  gettimeofday(&tv1,0);
  return ((tv1.tv_sec * USECPSEC)+tv1.tv_usec) - prev;
  // return ((tv1.tv_sec *1000)+tv1.tv_usec/1000) - prev;
}

__constant__ dataType2 gM[MAX_MASK_WIDTH];

__device__ dataType rgbLimit(int v){
  if(v>255)
    return 255;
  else if(v<0)
    return 0;
    
  return v;
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

__global__ void Convolution2D_Basic(dataType *Img_in, dataType2 *M, dataType *Img_out,int Mask_Width,int rowImg,int colImg){
  
  unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
  unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

  int N_start_point_i = row - (Mask_Width/2);
  int N_start_point_j = col - (Mask_Width/2);

    int Pvalue=0;
    for (int ii= 0;ii<Mask_Width;ii++) {
      for (int jj= 0;jj<Mask_Width;jj++) {
        if ((N_start_point_i+ii >= 0 && N_start_point_i + ii < colImg)&& (N_start_point_j+jj >= 0 && N_start_point_j + jj < rowImg)) {
          Pvalue+=Img_in[(N_start_point_i+ii)*rowImg+(N_start_point_j+jj)]*M[ii*Mask_Width+jj];
        }

      }
  }
 //if(row*rowImg+col<rowImg*colImg)
    Img_out[row*rowImg+col] = rgbLimit(Pvalue);
}

__global__ void Convolution2D_Constant(dataType *Img_in, dataType *Img_out,int Mask_Width,int rowImg,int colImg){
  
  unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
  unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

  int N_start_point_i = row - (Mask_Width/2);
  int N_start_point_j = col - (Mask_Width/2);

    int Pvalue=0;
    for (int ii= 0;ii<Mask_Width;ii++) {
      for (int jj= 0;jj<Mask_Width;jj++) {
        if ((N_start_point_i+ii >= 0 && N_start_point_i + ii < colImg)&& (N_start_point_j+jj >= 0 && N_start_point_j + jj < rowImg)) {
          Pvalue+=Img_in[(N_start_point_i+ii)*rowImg+(N_start_point_j+jj)]*gM[ii*Mask_Width+jj];
        }

      }
  }
 //if(row*rowImg+col<rowImg*colImg)
    Img_out[row*rowImg+col] = rgbLimit(Pvalue);
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
    P[i] = Pvalue;
}

int InvoqueKernel(dataType *pImgIn, dataType2 *pMask, dataType *pImg_out, int pRows, int pCols, int pMaskSize, int Option){
    /* 
        Función que invoca un método en el kernel (GPU) 
        Parametros:
            A: Vector de enteros.
            B: Vector de enteros a operar con A.
            C: Vector de enteros, para guardar la solución.
            Option: Saber que función va a llamar.
    */

    // int scale = 1;
    // int delta = 0;
    // int ddepth = CV_8UC1;
    Mat gray_image;

    int sizeImage = sizeof(dataType)*pRows*pCols;
    dataType *d_img_in, *d_img_out;
    dataType2 *d_Mask;
    cudaMalloc((void**)&d_img_in,sizeImage);
    cudaMalloc((void**)&d_img_out,sizeImage);
    cudaMalloc((void**)&d_Mask, pMaskSize);
    cudaMemcpy(d_img_in, pImgIn, sizeImage, cudaMemcpyHostToDevice);

    dim3 dimBlock3D(TILE_WIDTH, TILE_WIDTH, 1);
    int dimGridX = (int)ceil(pRows/(float)dimBlock3D.x);
    int dimGridY = (int)ceil(pCols/(float)dimBlock3D.y);
    dim3 dimGrid3D(dimGridX, dimGridY, 1);
    unsigned long long gpu_time = dtime_usec(0);



    switch (Option){
        case 1:
            cudaMemcpy(d_Mask, pMask, pMaskSize, cudaMemcpyHostToDevice);
            Convolution2D_Basic<<<dimGrid3D,dimBlock3D>>>(d_img_in, d_Mask, d_img_out, 3, pRows, pCols);
            cudaDeviceSynchronize();
            cudaMemcpy(pImg_out, d_img_out, sizeImage, cudaMemcpyDeviceToHost);
            gray_image.create(pCols, pRows, CV_8UC1);
            gray_image.data = pImg_out;
            imwrite("./outputs/1088313004.png",gray_image);
            gpu_time = dtime_usec(gpu_time);
            printf("Finished 1. Basic.  Results match. gpu time: %lldus\n", gpu_time);
            break;
        case 2:
            cudaMemcpyToSymbol(gM, pMask, pMaskSize);
            Convolution2D_Constant<<<dimGrid3D,dimBlock3D>>>(d_img_in, d_img_out, 3, pRows, pCols);
            cudaDeviceSynchronize();
            cudaMemcpy(pImg_out, d_img_out, sizeImage, cudaMemcpyDeviceToHost);
            gray_image.create(pCols, pRows, CV_8UC1);
            gray_image.data = pImg_out;
            imwrite("./outputs/1088313004.png",gray_image);
            gpu_time = dtime_usec(gpu_time);
            printf("Finished 2. Constant.  Results match. gpu time: %lldus\n", gpu_time);
            break;
            
        case 3:
            gpu_time = dtime_usec(gpu_time);
            break;

    }
    cudaMemcpy(pImg_out, d_img_out, sizeImage, cudaMemcpyDeviceToHost);
    cudaFree(d_img_in);
    cudaFree(d_img_out);
    return 0;
}


int main(){
    char imageSource[20];
    for (int i = 1; i <= 1; ++i)
    {
        Mat image;
        sprintf(imageSource, "inputs/img%d.jpg", i);
        printf("%s\n", imageSource);
        image = imread(imageSource, 0);
        Size sizeImage = image.size();

        int row=sizeImage.width;
        int col=sizeImage.height;
        int sizeMask= sizeof(dataType2)*9;
        int size = sizeof( dataType)*row*col;
        dataType *img_in=( dataType*)malloc(size);
        dataType *img_out=(dataType*)malloc(size);
        dataType2 Mask[9] = {-1,0,1,-2,0,2,-1,0,1};

        img_in = image.data;

        InvoqueKernel(img_in, Mask, img_out, row, col, sizeMask, 1);
        InvoqueKernel(img_in, Mask, img_out, row, col, sizeMask, 2);
    }
    return 0;
}