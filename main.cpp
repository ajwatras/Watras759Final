#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <iostream>

#include "FOV_models.cpp"

int main(int argc, char* argv[]){

double FOV_rads[2] = {1,1};
Matrix t;

Matrix R = rotateMat(.5,0,0);

t.x_dim = 1;
t.y_dim = 3;
t.elements = (double *)malloc(sizeof(double)*3);


double plane[4] = {0,0,1,-1};
double ray_normal[3] = {0,0,1};
double ray_translation[3] = {0, 0, 0};
double *point = (double *)malloc(sizeof(double)*3);

rayPlaneIntersect(plane, ray_normal, ray_translation, point);

printf("[ %f %f %f ]\n",point[0],point[1],point[2]);

polygon test_p = FOVproject(FOV_rads, plane,R,t);

std::cout << boost::geometry::wkt(test_p) << std::endl;


Matrix out = FOVcone(FOV_rads,R,t,1);

for (int i = 0; i < 4; i++){
	for (int j = 0; j < 3; j++){
		printf("%f ",out.elements[3*i +j]);
	}
	printf("\n");

}



}
