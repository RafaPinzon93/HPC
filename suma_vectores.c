#include <stdio.h>
//#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#define SIZE 100

int main(){
	int *A=(int *) malloc(SIZE*sizeof(int));
	
	int *B=(int *) malloc(SIZE*sizeof(int));
	
	int *C=(int *) malloc(SIZE*sizeof(int));
	time_t inicio,fin;
	inicio=time(NULL);
	int i;
	for(i=0;i< sizeof(A); i++){
		A[i]=rand()% 20;
		B[i]=rand()% 20;
	
	}
	
	for(i=0;i< sizeof(A); i++){
		C[i]=A[i]+B[i];
		printf("%i+",A[i]);
		printf("%i=",B[i]);
		printf("%i\n",C[i]);
	}
	//sleep(5);
	fin=time(NULL);
	printf("El tiempo es: %f\n",difftime(fin,inicio));
	free(A);
	free(B);
	return 0;
	
}