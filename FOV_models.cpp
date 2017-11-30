#include "omp.h"
#include <iostream>
#include <deque>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/io/wkt/wkt.hpp>

#include <boost/foreach.hpp>

struct Matrix{
	int x_dim;
	int y_dim;
	double *elements;
};

namespace bg = boost::geometry;
namespace bgm = bg::model;

using point      = bgm::d2::point_xy<double>;
using polygon    = bgm::polygon<point>;

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
	//incomplete, need to parallelize
	Matrix out, ray;

	out.x_dim = 3;
	out.y_dim = 4;
	double *elems = (double *) malloc(sizeof(double)*12);
	out.elements = elems;

	Matrix direction;
	direction.x_dim = 1;
	direction.y_dim = 3;
	double *dir_elems = (double *) malloc(sizeof(double)*3);

	dir_elems[0] = sin(FOV_rads[0]/2);
	dir_elems[1] = sin(FOV_rads[1]/2);
	dir_elems[2] = 1;

	direction.elements = dir_elems;

	ray = MatMult(R,direction);

	out.elements[0] = ray.elements[0];
	out.elements[1] = ray.elements[1];
	out.elements[2] = ray.elements[2];

	direction.elements[0] = -sin(FOV_rads[0]/2);
	direction.elements[1] = sin(FOV_rads[1]/2);
	direction.elements[2] = 1;

	ray = MatMult(R,direction);

	out.elements[3] = ray.elements[0];
	out.elements[4] = ray.elements[1];
	out.elements[5] = ray.elements[2];

	direction.elements[0] = -sin(FOV_rads[0]/2);
	direction.elements[1] = -sin(FOV_rads[1]/2);
	direction.elements[2] = 1;

	ray = MatMult(R,direction);

	out.elements[6] = ray.elements[0];
	out.elements[7] = ray.elements[1];
	out.elements[8] = ray.elements[2];

	direction.elements[0] = sin(FOV_rads[0]/2);
	direction.elements[1] = -sin(FOV_rads[1]/2);
	direction.elements[2] = 1;

	ray = MatMult(R,direction);

	out.elements[9] = ray.elements[0];
	out.elements[10] = ray.elements[1];
	out.elements[11] = ray.elements[2];


	return out;


}
int rayPlaneIntersect(double *plane, double *ray_normal, double *ray_translation, double *point){
	//Intersect a ray with a plane, then store the resulting 3x1 vector in point.
	double numerator, denominator, t;
	denominator = ray_normal[0]*plane[0] + ray_normal[1]*plane[1] + ray_normal[2]*plane[2];
	numerator = ray_translation[0]*plane[0] + ray_translation[1]*plane[1] + ray_translation[2]*plane[2] + plane[3];
	
	if (denominator == 0.)
		return 1;
	else{
		t = -numerator/denominator;
		for(int i = 0; i < 3; i++){
			point[i] = ray_translation[i] + ray_normal[i]*t;
		}
		return 0;
	}
}

polygon FOVproject(double *FOV_rads, double *plane_of_stitching,Matrix camera_R, Matrix camera_t){
	polygon output;
	double *coord = (double *)malloc(sizeof(double)*12);
	int flag;

	//generate FOV cone
	Matrix rays = FOVcone(FOV_rads,camera_R,camera_t,1);
	
	//Perform Ray-plane intersection
	flag = rayPlaneIntersect(plane_of_stitching,&rays.elements[0],camera_t.elements,&coord[0]);
	flag = rayPlaneIntersect(plane_of_stitching,&rays.elements[3],camera_t.elements,&coord[3]);
	flag = rayPlaneIntersect(plane_of_stitching,&rays.elements[6],camera_t.elements,&coord[6]);
	flag = rayPlaneIntersect(plane_of_stitching,&rays.elements[9],camera_t.elements,&coord[9]);

	//combine points into output
	point corners[4];

	for (int i = 0; i < 4; i++){
		corners[i] = { coord[3*i], coord[3*i+1] };
	}
	bg::assign_points(output,corners);
	return output;
}


double arrayArea(double *FOV_rads, Matrix camera_R, Matrix camera_t, double *plane_of_stitching,double thresh){
	//Finds the total area of a camera array set up.


	//incomplete
	return 0;
}

polygon combinePoly(polygon A, polygon B, char *flag){
	//Combines two polygons using intersection or union.
	polygon output;
//	boost::geometry::intersection(A, B, output);
	return output;
	//incomplete
}