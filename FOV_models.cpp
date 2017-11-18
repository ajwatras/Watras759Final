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
	Matrix R;

	Matrix Rx; 
	Rx.x_dim = 3;
	Rx.y_dim = 3;
	double *x_elements = (double *)malloc(sizeof(double)*Rx.x_dim*Rx.y_dim);

	x_elements[0] = 1;
	x_elements[4] = cos(x_ang);
	x_elements[5] = -sin(x_ang);
	x_elements[7] = sin(x_ang);
	x_elements[8] = cos(x_ang);

	Rx.elements = x_elements;

	Matrix Ry; 
	Ry.x_dim = 3;
	Ry.y_dim = 3;
	double *y_elements = (double *)malloc(sizeof(double)*Ry.x_dim*Ry.y_dim);

	y_elements[0] = cos(y_ang);
	y_elements[2] = sin(y_ang);
	y_elements[4] = 1;
	y_elements[6] = -sin(y_ang);
	y_elements[8] = cos(y_ang);

	Ry.elements = y_elements;

	Matrix Rz; 
	Rz.x_dim = 3;
	Rz.y_dim = 3;
	double *z_elements = (double *)malloc(sizeof(double)*Rz.x_dim*Rz.y_dim);

	z_elements[0] = cos(z_ang);
	z_elements[1] = -sin(z_ang);
	z_elements[3] = sin(z_ang);
	z_elements[4] = cos(z_ang);
	z_elements[8] = 1;

	Rz.elements = z_elements;

	R = MatMult(Rx,Ry);
	R = MatMult(R,Rz);
	
	return R;

}

Matrix FOVcone(double *FOV_rads, Matrix R, Matrix t, double scale){
	
}