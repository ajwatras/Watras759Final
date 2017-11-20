#include "omp.h"

struct Matrix{
	int x_dim;
	int y_dim;
	double *elements;
};

Matrix MatMult(Matrix A, Matrix B){
	Matrix C;
	C.x_dim = A.x_dim;
	C.y_dim = B.y_dim;
	//double elem[C.x_dim * C.y_dim];
	double *elem = (double *)malloc(sizeof(double)*C.x_dim*C.y_dim);
	int n = A.y_dim;


	for (int i = 0; i < C.x_dim; i++){
		for (int j = 0; j < C.y_dim; j++){
			for (int k = 0; k < n; k++){
				elem[C.y_dim * j + i] += A.elements[k*A.y_dim + i] * B.elements[j*B.y_dim + k];
			}
		}
	}
	C.elements = elem;
	return C;
}

Matrix rotateMat(double x_ang, double y_ang, double z_ang){
	//Generate the Rotation matrix from a set of rotation angles
	Matrix R,Rx,Ry,Rz;
	#pragma omp parallel sections
	{
		// Generate Rotation around the x axis. 
		#pragma omp section
		{ 
			Rx.x_dim = 3;
			Rx.y_dim = 3;
			double *x_elements = (double *)malloc(sizeof(double)*Rx.x_dim*Rx.y_dim);

			x_elements[0] = 1;
			x_elements[4] = cos(x_ang);
			x_elements[5] = -sin(x_ang);
			x_elements[7] = sin(x_ang);
			x_elements[8] = cos(x_ang);

			Rx.elements = x_elements;
		}
		//Generate rotation around the Y axis.
		#pragma omp section
		{
			Ry.x_dim = 3;
			Ry.y_dim = 3;
			double *y_elements = (double *)malloc(sizeof(double)*Ry.x_dim*Ry.y_dim);

			y_elements[0] = cos(y_ang);
			y_elements[2] = sin(y_ang);
			y_elements[4] = 1;
			y_elements[6] = -sin(y_ang);
			y_elements[8] = cos(y_ang);

			Ry.elements = y_elements;
		}

		// Generate rotation around the Z axis.
		#pragma omp section
		{
			Rz.x_dim = 3;
			Rz.y_dim = 3;
			double *z_elements = (double *)malloc(sizeof(double)*Rz.x_dim*Rz.y_dim);

			z_elements[0] = cos(z_ang);
			z_elements[1] = -sin(z_ang);
			z_elements[3] = sin(z_ang);
			z_elements[4] = cos(z_ang);
			z_elements[8] = 1;

			Rz.elements = z_elements;
		}
	}

	// Combine axis rotations into single matrix. 
	R = MatMult(Rx,Ry);
	R = MatMult(R,Rz);

	free(Rx.elements);
	free(Ry.elements);
	free(Rz.elements);

	return R;

}

Matrix FOVcone(double *FOV_rads, Matrix R, Matrix t, double scale){
	//incomplete
	Matrix out, ray;

	out.x_dim = 4;
	out.y_dim = 3;
	double *elems = (double *) malloc(sizeof(double)*12);

	Matrix direction;
	direction.x_dim = 1;
	direction.y_dim = 3;
	double *dir_elems = (double *) malloc(sizeof(double)*3);

	dir_elems[0] = sin(FOV_rads[0]/2);
	dir_elems[1] = sin(FOV_rads[1]/2);
	dir_elems[2] = 1;

	direction.elements = dir_elems;

	ray = MatMult(R,direction);


	return out;


}

int rayPlaneIntersect(double *plane, double *ray_normal, double *ray_translation, double *point){
	
	//incomplete
	return 0;

}

double arrayArea(){
	//incomplete
	return 0;
}

void combinePoly(){
	//incomplete
}