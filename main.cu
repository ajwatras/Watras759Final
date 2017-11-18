#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "FOV_models.cpp"

__global__ void simpleKernel(int * d_a){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	d_a[id] = id;
}

int main(int argc, char* argv[]){
	Matrix A,B;
	A.x_dim = 2;
	A.y_dim = 2;
	B.x_dim = 2;
	B.y_dim = 2;
	double a_vals[4];
	double b_vals[4];
	a_vals[0] = 1;
	a_vals[1] = 0;
	a_vals[2] = 0;
	a_vals[3] = 1;
	for (int i = 0; i < 4; i++){
		b_vals[i] = i;
	}
	A.elements = a_vals;
	B.elements = b_vals;
	Matrix C1 = MatMult(A,B);
	Matrix C2 = MatMult(B,B);
	printf("%d, %d \n",B.x_dim, B.y_dim);

	for (int i = 0; i < 2; i++){
		for (int j = 0; j < 2; j++){
			printf("%f ", C1.elements[2*i+j]);
		}
		printf("\n");
	}
	printf("\n");
	for (int i = 0; i < 2; i++){
		for (int j = 0; j < 2; j++){
			printf("%f ", C2.elements[2*i+j]);
		}
		printf("\n");
	}

	Matrix C3 = rotateMat(0,0,1);
	printf("\n");
	for (int i = 0; i < C3.y_dim; i++){
		for (int j = 0; j < C3.x_dim; j++){
			printf("%f ", C3.elements[C3.y_dim*i+j]);
		}
		printf("\n");
	}

}

int main2(int argc, char* argv[]){

	int num_threads=0,num_blocks=1,N=0,*d_a;
	int *h_a = (int *)malloc(sizeof(int)*N);
	if (argc == 1){
		printf("Usage: ./main N");
		return 1;
	}
	else 
		N = atoi(argv[1]);


	cudaMalloc((void **) &d_a,sizeof(int)*N);

	if (N > 1024){
		num_blocks = N/1024;
		num_threads = 1024;
	}
	else {
		num_blocks = 1;
		num_threads = N;
	}

	simpleKernel<<<num_blocks,num_threads>>>(d_a);
	cudaMemcpy(h_a,d_a,N*sizeof(int),cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++){
		printf("(%d,%d,%d): %d\n",N,num_blocks,num_threads,h_a[i]);
	}
	return 1;
}