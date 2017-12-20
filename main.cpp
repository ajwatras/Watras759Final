#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <iostream>
#include <algorithm>
#include <vector>

#include "FOV_models.cpp"

int main(int argc, char* argv[]){

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


int cc = 0;
int num_pos = ((Rz_bounds[2] - Rz_bounds[0])/Rz_bounds[1] + 1)*((T_bounds[2] - T_bounds[0])/T_bounds[1] + 1);

//Initialize allowable camera positions.
double *positionlist = (double *)malloc(sizeof(double)*2*num_pos);
for (double i = Rz_bounds[0]; i <= Rz_bounds[2]; i += Rz_bounds[1]){
	for (double j = T_bounds[0]; j <= T_bounds[2]; j += T_bounds[1]){
		positionlist[cc] = i;
		positionlist[cc+1] = j;
		cc += 2;
		std::cout << cc <<": [" << i <<"," << j <<"] " << std::endl;
	}	
}

//check every possible pose
int *pose = (int *)malloc(sizeof(int)*num_cameras);
double *Rz = (double *)malloc(sizeof(double)*num_cameras);
double *t = (double *)malloc(sizeof(double)*num_cameras);
double area, max_area,max_R,max_t,max_Rz;
max_area = 0;

std::vector<bool> v(num_pos);
std::fill(v.begin(), v.begin() + num_cameras, true);
do {
	cc = 0;
    for (int i = 0; i < num_pos; ++i) {
        if (v[i]) {
            pose[cc] = i;
            cc++;
        }
    }
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
    for (double R1 = R_bounds[0]; R1 <= R_bounds[2]; R1 += R_bounds[1]){
    	for (double R2 = R_bounds[0]; R2 <= R_bounds[2]; R2 += R_bounds[1]){
    		for (double R3 = R_bounds[0]; R3 <= R_bounds[2]; R3 += R_bounds[1]){
    			for (double R4 = R_bounds[0]; R4 <= R_bounds[2]; R4 += R_bounds[1]){
    				for (double R5 = R_bounds[0]; R5 <= R_bounds[2]; R5 += R_bounds[1]){
    					//Generate rotation Matrices
    					camera_R[0] = rotateMat(R1*cos(Rz[0]),R1*-sin(Rz[0]),Rz[0]);
    					camera_R[1] = rotateMat(R2*cos(Rz[1]),R2*-sin(Rz[1]),Rz[1]);
    					camera_R[2] = rotateMat(R3*cos(Rz[2]),R3*-sin(Rz[2]),Rz[2]);
    					camera_R[3] = rotateMat(R4*cos(Rz[3]),R4*-sin(Rz[3]),Rz[3]);
    					camera_R[4] = rotateMat(R5*cos(Rz[4]),R5*-sin(Rz[4]),Rz[4]);

    					max_R = arrayArea(FOV_rads, camera_R, camera_t, plane_of_stitching, 25);
    					area = 5;
    					if (area > max_area){

    						max_area = area;
    					}

    					//Check Area
    				}
    			}
    		}
    	}
    }
    
}while (std::prev_permutation(v.begin(), v.end()));

//sort output

//display final result. 
std::cout << "Maximum final Area: " << max_area << std::endl;

free(positionlist);
free(pose);
free(Rz);
free(t);
}
