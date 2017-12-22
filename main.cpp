#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <omp.h>

#include "stopwatch.hpp"
#include "FOV_models.cpp"
/*
int main(){
	double FOV_rads[2] = {.9449, .7273};
	double plane_of_stitching[4] = {0,0,1,0};
	Matrix camera_R,camera_t;
	double thresh = 25;
	camera_R = rotateMat(0,0,0);
	double t_elements[3] = {0,0-1};
	camera_t.x_dim = 3;
	camera_t.y_dim = 1;
	camera_t.elements = t_elements;
	polygon poly;

	poly = FOVproject(FOV_rads, plane_of_stitching, camera_R, camera_t);
	std::cout << "test"<< std::endl;

}*/
int main(int argc, char* argv[]){
//Process user inputs
if (argc != 2){
	std::cout << "Usage: ./main N" << std::endl;
	std::cout << "N should be the desired number of threads" << std::endl;
	return 0;
}

int N = atoi(argv[1]);
omp_set_num_threads(N);


//Initialize Variables
stopwatch<std::milli, float> sw;
double FOV_rads[2] = {.9449, .7273};
double plane_of_stitching[4] = {0,0,1,0};
double scene_depth = 16.5;
double thresh = 25;

int num_cameras = 5;
Matrix *camera_R = (Matrix *)malloc(sizeof(Matrix)*num_cameras);
Matrix *camera_t = (Matrix *)malloc(sizeof(Matrix)*num_cameras);
double Rz_bounds[3] = {-M_PI/2,M_PI/2,M_PI};
double R_bounds[3] = {-M_PI/3,M_PI/6, M_PI/3};
double T_bounds[3] = {2.5, 5,7.5};
int num_R = (R_bounds[2] - R_bounds[0])/R_bounds[1] + 1;
int num_T = ((T_bounds[2] - T_bounds[0])/T_bounds[1] + 1);
int num_Rz = ((Rz_bounds[2] - Rz_bounds[0])/Rz_bounds[1] + 1);
double *R_vals = (double *)malloc(sizeof(double)*num_R);
int cc = 0;
int num_pos = ((Rz_bounds[2] - Rz_bounds[0])/Rz_bounds[1] + 1)*((T_bounds[2] - T_bounds[0])/T_bounds[1] + 1);

int *pose = (int *)malloc(sizeof(int)*num_cameras);
double *Rz = (double *)malloc(sizeof(double)*num_cameras);
double *t = (double *)malloc(sizeof(double)*num_cameras);
double area, max_area;
max_area = 0;
polygon poly;
double *cam1 = (double *)malloc(sizeof(double)*3);
double *cam2 = (double *)malloc(sizeof(double)*3);
double *cam3 = (double *)malloc(sizeof(double)*3);
double *cam4 = (double *)malloc(sizeof(double)*3);
double *cam5 = (double *)malloc(sizeof(double)*3);

std::vector<bool> v(num_pos);										//Needed in order to run permutations. 
std::fill(v.begin(), v.begin() + num_cameras, true);

for (int i = 0; i < num_R; i++){
	R_vals[i] = R_bounds[0] + R_bounds[1] * i;
	//std::cout << R_vals[i] << " ";
}
//std::cout << std::endl;


sw.start();
//Initialize allowable camera positions.
double *positionlist = (double *)malloc(sizeof(double)*2*num_pos);
cc = 0;
#pragma omp parallel for collapse(2)
for (int i = 0; i < num_Rz; i++){
	for (int j = 0; j < num_T; j++){
		positionlist[cc] = Rz_bounds[0] + i*Rz_bounds[1];
		positionlist[cc+1] = T_bounds[0] + j*T_bounds[1];
		//std::cout << cc <<": [" << positionlist[cc] <<"," << positionlist[cc+1] <<"] " << std::endl;
		cc+=2;
		
	}
}

//Check every possible set of poses
do {
	cc = 0;
	//std::cout << "Pose: ";
	#pragma omp parallel for
    for (int i = 0; i < num_pos; ++i) {
        if (v[i]) {
            pose[cc] = i;
            //std::cout << pose[cc] <<", ";
            cc++;
        }
    }
    //std::cout << std::endl;
    #pragma omp parallel for
    for (int j = 0; j < num_cameras; j++){
    	//std::cout << pose[j] << " ";
    	// for the given pose, create Rz and t;
    	Rz[j] = positionlist[2*pose[j]];
    	t[j] = positionlist[2*pose[j]+1];

    	camera_t[j].x_dim = 3;
    	camera_t[j].y_dim = 1;
    	camera_t[j].elements = (double *)malloc(sizeof(double)*3);
    	camera_t[j].elements[0] = positionlist[2*pose[j]+1]*sin(Rz[j]);
    	camera_t[j].elements[1] = positionlist[2*pose[j]+1]*sin(Rz[j]);
    	camera_t[j].elements[2] = -scene_depth;
    }
    //std::cout << std::endl;
    //For each possible camera rotation, check the array area.
    #pragma omp parallel for collapse(5)
    for (int i1 = 0; i1 < num_R; i1++ ){
    	for (int i2 = 0; i2 < num_R; i2++ ){
    		for (int i3 = 0; i3 < num_R; i3++ ){
    			for (int i4 = 0; i4 < num_R; i4++ ){
    				for (int i5 = 0; i5 < num_R; i5++ ){  					
						//Generate rotation Matrices
    					camera_R[0] = rotateMat(R_vals[i1]*cos(Rz[0]),R_vals[i1]*-sin(Rz[0]),Rz[0]);
    					camera_R[1] = rotateMat(R_vals[i2]*cos(Rz[1]),R_vals[i2]*-sin(Rz[1]),Rz[1]);
    					camera_R[2] = rotateMat(R_vals[i3]*cos(Rz[2]),R_vals[i3]*-sin(Rz[2]),Rz[2]);
    					camera_R[3] = rotateMat(R_vals[i4]*cos(Rz[3]),R_vals[i4]*-sin(Rz[3]),Rz[3]);
    					camera_R[4] = rotateMat(R_vals[i5]*cos(Rz[4]),R_vals[i5]*-sin(Rz[4]),Rz[4]);
    				
    					//area = arrayArea(FOV_rads, camera_R, camera_t, plane_of_stitching, 25);
    					//poly = FOVproject(FOV_rads, plane_of_stitching, camera_R[0], camera_t[0]);
    					area = 5;

    					#pragma omp critical
    						if (area > max_area){

    							max_area = area;
    							cam1[0] = t[0];
    							cam1[1] = Rz[0];
    							cam1[2] = R_vals[i1];

    							cam2[0] = t[1];
    							cam2[1] = Rz[1];
    							cam2[2] = R_vals[i2];

    							cam3[0] = t[2];
    							cam3[1] = Rz[2];
    							cam3[2] = R_vals[i3];

    							cam4[0] = t[3];
    							cam4[1] = Rz[3];
    							cam4[2] = R_vals[i4];

    							cam5[0] = t[4];
    							cam5[1] = Rz[4];
    							cam5[2] = R_vals[i5];
    						}
    					 
    				}
    			}
    		}
    	}
    }
    
}while (std::prev_permutation(v.begin(), v.end()));
sw.stop();
//sort output

//display final result. 
std::cout << "Number of threads: " << N << std::endl;
std::cout << "Maximum final Area: " << max_area << std::endl;
std::cout << "Inclusive Run Time:  " << sw.count() << " ms" << std::endl;
std::cout << "Camera 1: " << std::endl;
std::cout << "T: " << cam1[0] << ", Rz: " << cam1[1] << ", R: " << cam1[2] << std::endl;
std::cout << "Camera 2: " << std::endl;
std::cout << "T: " << cam2[0] << ", Rz: " << cam2[1] << ", R: " << cam2[2] << std::endl;
std::cout << "Camera 3: " << std::endl;
std::cout << "T: " << cam3[0] << ", Rz: " << cam3[1] << ", R: " << cam3[2] << std::endl;
std::cout << "Camera 4: " << std::endl;
std::cout << "T: " << cam4[0] << ", Rz: " << cam4[1] << ", R: " << cam4[2] << std::endl;
std::cout << "Camera 5: " << std::endl;
std::cout << "T: " << cam5[0] << ", Rz: " << cam5[1] << ", R: " << cam5[2] << std::endl;

free(positionlist);
free(pose);
free(Rz);
free(t);
free(cam1);
free(cam2);
free(cam3);
free(cam4);
free(cam5);
}
