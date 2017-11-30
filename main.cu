#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "FOV_models.cpp"

//__global__ void simpleKernel(int * d_a){
//	int id = threadIdx.x + blockDim.x * blockIdx.x;
//	
//	d_a[id] = id;
//}

int main(int argc, char* argv[]){

double FOV_rads[2] = {1,1};
Matrix R, t;

R.x_dim = 3;
R.y_dim = 3;
t.x_dim = 1;
t.y_dim = 3;

t.elements = (double *)malloc(sizeof(double)*3);
R.elements = (double *)malloc(sizeof(double)*9);
R.elements[0] = 1;
R.elements[4] = 1;
R.elements[8] = 1;


Matrix out = FOVcone(FOV_rads,R,t,1);

for (int i = 0; i < 3; i++){
	for (int j = 0; j < 4; j++){
		printf("%f ",out.elements[4*i +j]);
	}
	printf("\n");

}

}

/*
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
*/