#include "omp.h"
#include <iostream>
#include <deque>
#include <vector>

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
	//Basic matrix multiplication. 
	//Could easily be made parallel in CUDA or openMP
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
	Matrix out, ray1,ray2,ray3,ray4;

	out.x_dim = 3;
	out.y_dim = 4;
	double *elems = (double *) malloc(sizeof(double)*12);
	out.elements = elems;

	#pragma omp parallel sections
	{
		#pragma omp section
		{ 
			Matrix direction1;
			direction1.x_dim = 1;
			direction1.y_dim = 3;
			direction1.elements = (double *) malloc(sizeof(double)*3);

			direction1.elements[0] = sin(FOV_rads[0]/2);
			direction1.elements[1] = sin(FOV_rads[1]/2);
			direction1.elements[2] = 1;

			ray1 = MatMult(R,direction1);

			out.elements[0] = ray1.elements[0];
			out.elements[1] = ray1.elements[1];
			out.elements[2] = ray1.elements[2];
		}
		#pragma omp section
		{
			Matrix direction2;
			direction2.x_dim = 1;
			direction2.y_dim = 3;
			direction2.elements = (double *) malloc(sizeof(double)*3);

			direction2.elements[0] = -sin(FOV_rads[0]/2);
			direction2.elements[1] = sin(FOV_rads[1]/2);
			direction2.elements[2] = 1;

			ray2 = MatMult(R,direction2);

			out.elements[3] = ray2.elements[0];
			out.elements[4] = ray2.elements[1];
			out.elements[5] = ray2.elements[2];
		}
		#pragma omp section
		{
			Matrix direction3;
			direction3.x_dim = 1;
			direction3.y_dim = 3;
			direction3.elements = (double *) malloc(sizeof(double)*3);

			direction3.elements[0] = -sin(FOV_rads[0]/2);
			direction3.elements[1] = -sin(FOV_rads[1]/2);
			direction3.elements[2] = 1;

			ray3 = MatMult(R,direction3);

			out.elements[6] = ray3.elements[0];
			out.elements[7] = ray3.elements[1];
			out.elements[8] = ray3.elements[2];
		}
		#pragma omp section
		{
			Matrix direction4;
			direction4.x_dim = 1;
			direction4.y_dim = 3;
			direction4.elements = (double *) malloc(sizeof(double)*3);

			direction4.elements[0] = sin(FOV_rads[0]/2);
			direction4.elements[1] = -sin(FOV_rads[1]/2);
			direction4.elements[2] = 1;

			ray4 = MatMult(R,direction4);

			out.elements[9] = ray4.elements[0];
			out.elements[10] = ray4.elements[1];
			out.elements[11] = ray4.elements[2];
		}
	}
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
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			flag = rayPlaneIntersect(plane_of_stitching,&rays.elements[0],camera_t.elements,&coord[0]);
		}
		#pragma omp section
		{
			flag = rayPlaneIntersect(plane_of_stitching,&rays.elements[3],camera_t.elements,&coord[3]);
		}
		#pragma omp section
		{
			flag = rayPlaneIntersect(plane_of_stitching,&rays.elements[6],camera_t.elements,&coord[6]);
		}
		#pragma omp section
		{
			flag = rayPlaneIntersect(plane_of_stitching,&rays.elements[9],camera_t.elements,&coord[9]);	
		}
	}

	//combine points into output
	point corners[4];

	#pragma omp parallel for 
	{	
		for (int i = 0; i < 4; i++){
			corners[i] = { coord[3*i], coord[3*i+1] };
		}
	}

	bg::assign_points(output,corners);
	return output;
}

polygon combinePoly(polygon A, polygon B, bool flag,int type){
	//Combines two polygons using intersection or union.
	// Type = 0 => union
	// Type = 1 => Intersection
	std::vector<polygon> output;


	if (type == 0){
		bg::union_(A, B, output);
	}
	else if (type == 1){
		bg::intersection(A, B, output);
	}

	if (output.size() > 1){
		flag = false;
		return A;
	}
	else if (output.size() == 1){
		flag = true;
		return output[0];
	}

	return A;
}
bool isConnected(Matrix A){
	//Incomplete method for determining if threshold limit has been met. 
	return true;
}

double arrayArea(double *FOV_rads, Matrix *camera_R, Matrix *camera_t, double *plane_of_stitching,double thresh){
	//Finds the total area of a camera array set up.
	int n = 5;
	polygon * poly = (polygon *)malloc(sizeof(polygon)*n);
	polygon array_poly,temp_poly;
	double area = 0;
	bool flag = false;
	int *successes = (int *)malloc(sizeof(int)*n);
	Matrix overlap_area;

	

	for (int k = 0; k < n; k++){
		poly[k] = FOVproject(FOV_rads, plane_of_stitching, camera_R[k], camera_t[k]);

 	}
 	array_poly = poly[0];

 	
 	overlap_area.x_dim = n;
 	overlap_area.y_dim = n;
 	overlap_area.elements = (double *)malloc(sizeof(double)*n*n);


 	for (int i = 0; i < n; i++){
 		for (int j = 0; j < n; j++){
 			//Calculate overlap area matrix. 
 			temp_poly = combinePoly(poly[i],poly[j],flag,1);
 			if (flag == true)
 			{
	 			overlap_area.elements[i + n*j] = bg::area(temp_poly);
	 			overlap_area.elements[j + n*i] = bg::area(temp_poly);
 			}
 			//Calculate union polygon.
 			if ((area > 0) || (i != (n-1))){
 				temp_poly = combinePoly(array_poly,poly[j],flag,0);
 				if (flag == true)
 				{
 					//Calculate area
 					array_poly = temp_poly;
 					area = bg::area(array_poly); 
 					successes[j] = 1;
 				}

 			}
		}
 	}
 	//check to make sure every polygon was added
 	for (int i = 1; i < n; i++){
 		successes[0] = successes[0]*successes[i];
 	}
 	if (successes[0] == 0){
 		area = 0;
 		return area;
 	}


 	//Check for fulfillment of overlap requirement.
 	flag = isConnected(overlap_area);
 	if (flag == false){
 		area = 0;
 	}

	return area;
}

