#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"
// #include "ray_tracing_GPU.h"

#define X 0
#define Y 1
#define Z 2

#define MY_CUDA_CHECK(call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

 #define MY_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }


#define CPROD(dest,v1,v2) \
dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
dest[2]=v1[0]*v2[1]-v1[1]*v2[0]; 
    
    

    
    
#define DPROD(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])

// #define DPROD1(v1,v2) \
// return (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]);

    
    
#define SUB(dest,v1,v2) \
    dest[0]=v1[0]-v2[0]; \
    dest[1]=v1[1]-v2[1]; \
    dest[2]=v1[2]-v2[2]; 
    
    
    
#define FINDMINMAX(x0,x1,x2,min,max) \
min = max = x0;   \
if(x1<min) min=x1;\
if(x1>max) max=x1;\
if(x2<min) min=x2;\
if(x2>max) max=x2;
    
    /******** X_AXIS test *********/
#define AXISTEST_X01(a, b, fa, fb)			   \
p0 = a*v0[Y] - b*v0[Z];			       	   \
p2 = a*v2[Y] - b*v2[Z];			       	   \
if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;} \
rad = fa * boxhalfsize[Y] + fb * boxhalfsize[Z];   \
if(min>rad || max<-rad) return 0;
    
    
#define AXISTEST_X2(a, b, fa, fb)			   \
p0 = a*v0[Y] - b*v0[Z];			           \
p1 = a*v1[Y] - b*v1[Z];			       	   \
if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
rad = fa * boxhalfsize[Y] + fb * boxhalfsize[Z];   \
if(min>rad || max<-rad) return 0;
    
    
/*********** Y_AXIS test ************/
#define AXISTEST_Y02(a, b, fa, fb)			   \
p0 = -a*v0[X] + b*v0[Z];		      	   \
p2 = -a*v2[X] + b*v2[Z];	       	       	   \
if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;} \
rad = fa * boxhalfsize[X] + fb * boxhalfsize[Z];   \
if(min>rad || max<-rad) return 0;
    
    
#define AXISTEST_Y1(a, b, fa, fb)			   \
p0 = -a*v0[X] + b*v0[Z];		      	   \
p1 = -a*v1[X] + b*v1[Z];	     	       	   \
if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
rad = fa * boxhalfsize[X] + fb * boxhalfsize[Z];   \
if(min>rad || max<-rad) return 0;
    
    
/***************** Z_AXIS test ************************/
    
    
#define AXISTEST_Z12(a, b, fa, fb)			   \
p1 = a*v1[X] - b*v1[Y];			           \
p2 = a*v2[X] - b*v2[Y];			       	   \
if(p2<p1) {min=p2; max=p1;} else {min=p1; max=p2;} \
rad = fa * boxhalfsize[X] + fb * boxhalfsize[Y];   \
if(min>rad || max<-rad) return 0;
    
    
    
#define AXISTEST_Z0(a, b, fa, fb)			   \
p0 = a*v0[X] - b*v0[Y];				   \
p1 = a*v1[X] - b*v1[Y];			           \
if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
rad = fa * boxhalfsize[X] + fb * boxhalfsize[Y];   \
if(min>rad || max<-rad) return 0;
    
__device__ __host__ struct Point3D {
    double x;
    double y;
    double z;
};

__device__ __host__ struct Triangle {
    double v0[3];
    double v1[3];
    double v2[3];
};

__device__ __host__ struct Index
{
    int size;
    int count;
    int *ptr;

    __device__ __host__ Index()  
    { 
        ptr = (int *)malloc(sizeof(int) *100);
        size = 100;
        count = 0; 
    }

    __device__ __host__ void Insert(int value){
        if(count == size){
            // printf("Enters %p",ptr); 
            int *newptr = (int *)malloc(sizeof(int)*(size + 1));
            for (int index = 0; index<size; index++){
                newptr[index] = ptr[index];
            }
           // delete [] ptr;
            free(ptr);
            ptr = newptr;
            size+=1;
        }
        ptr[count] = value;
        count+=1;
    } 

};

__device__ __host__ struct Cube { 

    double xmin, xmax;
    double ymin, ymax;
    double zmin, zmax;
    double boxhalfsize[3];
    double center[3];
    int reference;
    int counter=0;
    int *triangleIndexes;
    // int triangleIndexes[12];
    int index =0;

};

__device__ __host__ struct Ray {
    //Rreze
    double orig[3];
    double dir[3];
    int intersectedCubes[4]={-1};
    int index;
    int intersects=0;
};
    
    

using namespace std;

/******************Declaring global variables ***************/
Ray *rays;
int numRays;
Cube boundingBox;
bool *isInside;
bool *isInPlane;
Triangle *triangles;
int numTriangles;

/****************** Declaring functions*********************/
__device__ __host__ double dot_product(double *v1, double *v2);
void  GETCENTER(Cube cube, double *center);
void GETHALFSIZE(Cube cube, double *boxhalfsize);
double into_double(char *str);
void getTriangles(Triangle *triangles);
void generateRandoms(double *direction, double *origin); 
void getRays(Ray *rays);
int getNumberOfCubes(Point3D lower, Point3D upper, double delta);
void createGrid3D(Cube *cubes, Point3D lower, Point3D upper, double delta, Index &frontGrid);
__device__ __host__  bool rayTriangleIntersect(Ray ray, Triangle tri);
__device__ __host__ int planeBoxOverlap(double *normal, double *vert, double *maxbox);
__device__ __host__ int triBoxOverlap(double *boxcenter, double *boxhalfsize, Triangle triverts);
__host__ __device__ double TriArea(double *P1, double *P2, double *P3);
__device__ __host__ bool IsInPlaneTest(Ray ray, Triangle tri, double epsilon);
__device__ __host__ bool  PointIsInSurface(Ray ray, Triangle tri, double epsilon);
__device__ __host__ int rayBoxIntersection_yaxis(double *rayorigin, Cube cube);
__global__ void TriangleCount_GPU(Triangle *triangles, Cube *cubes, int *numCubes, int *numTriangles, bool *CubeTriMapp);
__global__ void calcAlloc(Cube *cubes, int *frontGrid, int *cubesPerBlock, int *indexes, int *counter);
__global__ void calcAlloc(Cube *cubes, int *frontGrid, int *cubesPerBlock, int *indexes, int *counter);
__global__ void findUnique(Cube *cubes, int *end, int *cubesPerBlock, bool *CubeTriMapp, int *numTriangles, int *numCubes);
__global__ void appendTriangles(Cube *cubes, int *indexes, int *frontGrid, int *numTriangles, int *numCubes, bool *CubeTriMapp );
__global__ void getIntersectedCubes_GPU( Ray *rays, Cube *cubes, int *numrays, int *indexes,  Cube *boundingBox, int *frontGrid);
void IsInPlane(Ray *rays, Triangle *tri, bool *isInside, int nr_rays, int nr_triangles, double epsilon);
__global__ void IsInPlaneGPU(Ray *rays, Triangle *tri, bool *isInside, int *n_r, int *n_tr, Cube *cubes);
__global__  void ray_tracingGPU(Ray *ray, Triangle *tri, bool *results, int *n_r, int *n_tr, Cube *cubes);
extern "C" void SetRays(float *xyz_flat, int nr_rays);
extern "C" void SetTriangles(double *tri_coordinates, int nr_triangles);
extern "C" void  SetBoundingBox(double *x, double *y, double *z);
extern "C" bool *RayTracingGPU(int *numRays_, int *numTriangles);


/**************Defining the functions *****************/
__device__ __host__ double dot_product(double *v1, double *v2){

    double dot_prod=0;
    for (int i = 0; i< 3; i++){
        dot_prod += v1[i]*v2[i];
    }
    return dot_prod;
}


void  GETCENTER(Cube cube, double *center)
{
center[0] = (cube.xmax + cube.xmin)/2;
center[1] = (cube.ymax + cube.ymin)/2;
center[2] = (cube.zmax + cube.zmin)/2;

}


void GETHALFSIZE(Cube cube, double *boxhalfsize)
{
boxhalfsize[0] = (cube.xmax - cube.xmin)/2;
boxhalfsize[1] = (cube.ymax - cube.ymin)/2;
boxhalfsize[2] = (cube.zmax - cube.zmin)/2;

}



double into_double(char *str){
    char *ptr;
    double value;
    value = strtod(str, &ptr);
    return value;
}


void getTriangles(Triangle *triangles){

    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    FILE *fp = fopen("/home/diko/MOBIUS/MOEBIUS/BACKEND/SCRIPTS/TEMP/triangles.txt","r");
    if(fp == NULL){
        perror(" unable to open file ");
        exit(1);
    }

    int index = 0;
    int nr_vec =0;
    
    while ((read = getline(&line, &len, fp)) != -1) {
        int j =0;
        char* token = strtok(line," ");
        while(token != NULL){
            token = strtok(NULL," ");
            if(nr_vec == 0){
                if(token != NULL){
                double token_ = into_double(token);
                    double r = 0;

                    if(token_ < 0){
                        token_ = token_ - r;
                    }
                    else token_ = token_ +r;
                triangles[index].v0[j] = token_;
               
                j = j+1;
                }
            }
            else if(nr_vec == 1){
                if(token != NULL){
                double token_ = into_double(token);
                    double r = 0;

                    if(token_ < 0){
                        token_ = token_ - r;
                    }
                    else token_ = token_ +r;
                triangles[index].v1[j] = token_;
               
                j = j+1;
                }
            }
            else{
                if(token != NULL){
                double token_ = into_double(token);
                    double r = 0;
                    if(token_ < 0){
                        token_ = token_ - r;
                    }
                    else token_ = token_ +r;
                triangles[index].v2[j] = token_;
                j = j+1;
                }
            }

        }
        if(nr_vec == 2){
            index+=1;
            nr_vec=0;
        }
        else nr_vec+=1;
        }
   

    fclose(fp);
    if (line)
        free(line);
}


 void generateRandoms(double *direction, double *origin) 
{ 
    
	direction[0] = 0.0002;
	direction[1] = 1.0;
	direction[2] = 0.0002;

} 

 void getRays(Ray *rays){
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    FILE *fp = fopen("/home/diko/MOBIUS/MOEBIUS/BACKEND/SCRIPTS/TEMP/points.txt","r");
    if(fp == NULL){
        perror(" unable to open file ");
        exit(1);
    }
    int index = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        int j =0;
        char* token = strtok(line," ");            
        
        while(token != NULL && j < 3){
            rays[index].orig[j] =into_double(token);
            //printf("\n token %d %d is : %.5lf",index,j,rays[index].orig[j]);
            token = strtok(NULL," ");
            j = j+1;
        }
        generateRandoms(rays[index].dir, rays[index].orig);
        index+=1;
    }    
    fclose(fp);
    if (line)
        free(line);
    
}


int getNumberOfCubes(Point3D lower, Point3D upper, double delta){


    int xn = ceil((upper.x - lower.x)/delta);
    int yn = ceil((upper.y - lower.y)/delta);
    int zn = ceil((upper.z - lower.z)/delta);

    int num_BB = xn * yn * zn;

    return num_BB;
}


void createGrid3D(Cube *cubes, Point3D lower, Point3D upper, double delta, Index &frontGrid){

    /***********************************************
    Input: Empty allocated memory for cubes, lower point and upper point
    Outpu: It is a void function, will modify the array of cubes
    ***********************************************/
    // double Y_MAX = upper.y;

    const int xn = ceil((upper.x - lower.x)/delta);
    const int yn = ceil((upper.y - lower.y)/delta);
    const int zn = ceil((upper.z - lower.z)/delta);


    double x = 0;
    double y = 0;
    double z = 0;
    int block = 0;
    int index = 0;
    for(int i = 0; i<xn; i++){
        x = lower.x;
        x = x+delta*i;

        for(int j = 0; j<zn; j++){
            z = lower.z;
            z = z+delta*j;

            for(int k =0; k<yn; k++){
                y = lower.y;
                y = y + delta*k;

                cubes[index].xmax = x+delta;
                cubes[index].ymax = y+delta;
                cubes[index].zmax = z+delta;
                cubes[index].xmin = x;
                cubes[index].ymin = y;
                cubes[index].zmin = z;

                GETCENTER(cubes[index], cubes[index].center);
                GETHALFSIZE(cubes[index], cubes[index].boxhalfsize);

                if(k== yn-1) 
                {
                    frontGrid.Insert(index);
                }
                cubes[index].reference = (block+1)*yn -1;
                index +=1;
                
                
            }
            block+=1;
    
        }
    }

}



__device__ __host__  bool rayTriangleIntersect(Ray ray, Triangle tri){
   
    double ZERO = 0.0;
    double ONE = 1.0;

    const double epsilon = 1.e-10;
    double edge1[3] ={}; double edge2[3] ={} ; double pvec[3]={};double tvec[3] = {}; double qvec[3] = {};
    
    //Calculate edges
    SUB(edge1,tri.v1, tri.v0); 
    SUB(edge2,tri.v2, tri.v0);
    CPROD(pvec,ray.dir, edge2); 
    
    
    //Calculate the determinant 
    double det = dot_product(edge1, pvec); 
    
    if( det>-epsilon && det < epsilon) return false; //It is parallel with the triangle

    //Calculate the inverse determinant
    double invDet = ONE / det; // f
    //calculate the 'u' of baycentric 
    SUB(tvec,ray.orig, tri.v0); //s
    double prod = dot_product(tvec, pvec); // s dot prod h
    double u = prod *  invDet; //f * s dot prod h

    //Check if u is inside the allowed bounds
    if(u < ZERO || u > ONE) return false;

    //calculate the 'v' of baycentric coordinates and check if it is inside the desired interval to continue 
    CPROD(qvec,tvec, edge1);
    double prod1 = dot_product(ray.dir, qvec);
    double v = prod1 * invDet;

    if(v < ZERO || u + v > ONE ) return false;

    double t_prod = dot_product(edge2, qvec); 
    double t =t_prod * invDet;

    return (t>epsilon);

}



__device__ __host__ int planeBoxOverlap(double *normal, double *vert, double *maxbox)	
{

    int q;
    double vmin[3],vmax[3],v;
    for(q=X;q<=Z;q++)
    {
        v=vert[q];
        
        int isBigger = (int) (normal[q]>0.0f);
        
        vmin[q]=isBigger*(-maxbox[q] - v) + !isBigger*(maxbox[q] - v);	
        vmax[q]= isBigger*(maxbox[q] - v) + !isBigger *(-maxbox[q] - v) ;	
       
    }

    return !(DPROD(normal,vmin)>0.0f) * (DPROD(normal,vmax)>=0.0f);

}


__device__ __host__ int triBoxOverlap(double *boxcenter, double *boxhalfsize, Triangle triverts)
{

double v0[3]={0};
double v1[3]={0};
double v2[3]={0};

double min=1000000,max=-10000000,p0=0,p1=0,p2=0,rad=0,fex=0,fey=0,fez=0;	

double normal[3]={0};
double e0[3]={0};
double e1[3]={0};
double e2[3]={0};

SUB(v0,triverts.v0,boxcenter);
SUB(v1,triverts.v1,boxcenter);
SUB(v2,triverts.v2,boxcenter);

SUB(e0,v1,v0);     
SUB(e1,v2,v1);      
SUB(e2,v0,v2);     

CPROD(normal,e0,e1);
return planeBoxOverlap(normal, v0, boxhalfsize);

}

__host__ __device__ double TriArea(double *P1, double *P2, double *P3)
{
    double P1P2[3];
    double P1P3[3];
    double CP[3];

    SUB(P1P2, P2, P1);
    SUB(P1P3, P3, P1);
    CPROD(CP,P1P2, P1P3)

    double triArea = sqrt(CP[0]*CP[0] + CP[1]*CP[1] +CP[2]*CP[2])/2;
    return triArea;

}

__device__ __host__ bool IsInPlaneTest(Ray ray, Triangle tri, double epsilon)
{
    //Find the plane equation starting with coordinates a, b, c and then distance d

    double a1, b1, c1;
    double a2, b2, c2;
    double a, b, c; //rate of normal vector of the plane
    double d; //distance

    a1 = tri.v1[0] -tri.v0[0];
    b1 = tri.v1[1] -tri.v0[1];
    c1 = tri.v1[2] -tri.v0[2];

    a2 = tri.v2[0] -tri.v0[0];
    b2 = tri.v2[1] -tri.v0[1];
    c2 = tri.v2[2] -tri.v0[2];

    a = b1 * c2 - b2 * c1;
    b = a2 * c1 - a1 * c2;
    c = a1 * b2 - b1 * a2; 

    d = (- a * tri.v0[0] - b * tri.v0[1] - c * tri.v0[2]);

    //Check if point is in plane
    return (a*ray.orig[0]+b*ray.orig[1]+c*ray.orig[2]+d<=0+epsilon && a*ray.orig[0]+b*ray.orig[1]+c*ray.orig[2]+d>=0-epsilon );
    // return (a*ray.orig[0]+b*ray.orig[1]+c*ray.orig[2]+d==0 );

}

__device__ __host__ bool  PointIsInSurface(Ray ray, Triangle tri, double epsilon)
{
    // const double epsilon = 1.e-7;
    double tri0, tri1, tri2, tri3;
    tri0 = TriArea(tri.v0, tri.v1, tri.v2);
    tri1 = TriArea(ray.orig, tri.v0, tri.v1);
    tri2 = TriArea(ray.orig, tri.v0, tri.v2);
    tri3 = TriArea(ray.orig, tri.v1, tri.v2);

    double sum = tri1 + tri2 +tri3;
    double res = sqrt(sum)/sqrt(tri0)-1;
    return(res <= epsilon && res >= -epsilon);
    // return sum == tri0;
  

}


__device__ __host__ int rayBoxIntersection_yaxis(double *rayorigin, Cube cube)
{

    // const double epsilon = 1.e-5;
    double xmin = cube.xmin;
    double xmax = cube.xmax;
    double zmin = cube.zmin;
    double zmax = cube.zmax;
    double ymax = cube.ymax;
    double ymin = cube.ymin;
    return !(rayorigin[0]<xmin || rayorigin[0] >xmax || rayorigin[2]<zmin || rayorigin[2]>zmax || rayorigin[1]>ymax);
    // return !(rayorigin[0]<xmin || rayorigin[0] >xmax || rayorigin[2]<zmin || rayorigin[2]>zmax || rayorigin[1]>ymax || rayorigin[1]<ymin);

    // return !(rayorigin[0]<xmin-epsilon || rayorigin[0] >xmax+epsilon || rayorigin[2]<zmin-epsilon || rayorigin[2]>zmax+epsilon);

}



__global__ void TriangleCount_GPU(Triangle *triangles, Cube *cubes, int *numCubes, int *numTriangles, bool *CubeTriMapp)
    {
        int idx = threadIdx.x + blockIdx.x*blockDim.x;
        const unsigned int num_threads = gridDim.x * blockDim.x;

        for (int i=idx; i<(*numCubes)*(*numTriangles) ; i+=num_threads)
        {               
            int triIndex = (int)i/(*numCubes);
            int cubeIndex = i%(*numCubes);
            int intersects = (int) (triBoxOverlap(cubes[cubeIndex].center, cubes[cubeIndex].boxhalfsize, triangles[triIndex])==1);
            CubeTriMapp[i]=(intersects>0);
            atomicAdd(&cubes[cubeIndex].counter, intersects);
        }

    }


__global__ void calcAlloc(Cube *cubes, int *frontGrid, int *cubesPerBlock, int *indexes, int *counter)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    const unsigned int num_threads = gridDim.x * blockDim.x;
    for(int i=idx; i<*frontGrid; i+=num_threads)
    {
        int end = indexes[i];
        int start = end - (*cubesPerBlock-1);

        for(int j=start; j<end; j++)
        {
            cubes[end].counter+=cubes[j].counter;
        }


    }
    
}

__global__ void findUnique(Cube *cubes, int *end, int *cubesPerBlock, bool *CubeTriMapp, int *numTriangles, int *numCubes)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    const unsigned int num_threads = gridDim.x * blockDim.x;
    int start = *end -(*cubesPerBlock-1);
    
    for(int triIndex=idx; triIndex<*numTriangles; triIndex+=num_threads)
    {
        for(int cubeIndex=start; cubeIndex<*end; cubeIndex++)
        {
            CubeTriMapp[triIndex*(*numCubes)+(*end)]|=CubeTriMapp[triIndex*(*numCubes)+cubeIndex];
        }
        atomicAdd(&cubes[*end].index,(int)(CubeTriMapp[triIndex*(*numCubes)+(*end)]==true));
    }

}



__global__ void appendTriangles(Cube *cubes, int *indexes, int *frontGrid, int *numTriangles, int *numCubes, bool *CubeTriMapp )
    {
        int idx = threadIdx.x + blockIdx.x*blockDim.x;
        const unsigned int num_threads = gridDim.x * blockDim.x;

        for(int i=idx; i<*frontGrid; i+=num_threads)
        {
            int cubeIndex=indexes[i];
            // int counter = 0;
            cubes[cubeIndex].index=0;
            for(int triIndex=0; triIndex<*numTriangles; triIndex+=1)
            {   

                int index = triIndex*(*numCubes) + cubeIndex;
                cubes[cubeIndex].triangleIndexes[cubes[cubeIndex].index] = triIndex;
                cubes[cubeIndex].index+=1*(int)(CubeTriMapp[index]==true);
                
            }
        }

        
    }


__global__ void getIntersectedCubes_GPU( Ray *rays, Cube *cubes, int *numrays, int *indexes,  Cube *boundingBox, int *frontGrid)
    {   
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        const unsigned int num_threads = gridDim.x*blockDim.x;


        for(int index = idx; index<*numrays; index+=num_threads)
        {
            rays[index].index = 0;            
            for(int frontGridIndex=0; frontGridIndex<*frontGrid; frontGridIndex++)
            {
                    int intersects = (int)rayBoxIntersection_yaxis(rays[index].orig, cubes[indexes[frontGridIndex]]);
                    rays[index].intersectedCubes[rays[index].index] = indexes[frontGridIndex];
                    rays[index].index+=1*intersects;
            }


 
        }

    }



    
 __global__  void ray_tracingGPU(Ray *ray, Triangle *tri, bool *results, int *n_r, int *n_tr, Cube *cubes){

        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        const unsigned int num_threads = gridDim.x*blockDim.x;
        
        for(int rayIndex = idx; rayIndex <*n_r; rayIndex+=num_threads)
        {       
            int count = 0;
            if(ray[rayIndex].index > 0)
            {
                for (int j = 0; j<cubes[ray[rayIndex].intersectedCubes[0]].index; j=j+1){
                        count += (int)rayTriangleIntersect(ray[rayIndex],tri[cubes[ray[rayIndex].intersectedCubes[0]].triangleIndexes[j]]);
                }
 
                results[rayIndex] = (count % 2 !=0);       
               
            }
        
    }

    }

    __global__ void IsInPlaneGPU(Ray *rays, Triangle *tri, bool *isInside, int *n_r, int *n_tr, Cube *cubes)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        const unsigned int num_threads = gridDim.x*blockDim.x;
        const double epsilon = 1.e-10;
        for(int rayIndex=idx; rayIndex<*n_r; rayIndex+=num_threads)
        {
            if(rays[rayIndex].index > 0)
            {
                int counter = 0;
                    for(int j=0; j<cubes[rays[rayIndex].intersectedCubes[0]].index;)
                    {
    
                            counter+=(int)(PointIsInSurface(rays[rayIndex], tri[cubes[rays[rayIndex].intersectedCubes[0]].triangleIndexes[j]], epsilon)==true);
                            j+=1+counter*cubes[rays[rayIndex].intersectedCubes[0]].index;
    
                    }
                    isInside[rayIndex]=(counter>0);
            }
        }
    }
    
    

__global__ void appendALL(Cube *cubes,int *numCubes,int *numTriangles, bool *CubeTriMapp, Triangle *triangles)
    {
        int idx = threadIdx.x + blockIdx.x*blockDim.x;
        const unsigned int num_threads = gridDim.x * blockDim.x;
    
        for(int cubeIndex=idx; cubeIndex<*numCubes; cubeIndex+=num_threads)
        {
            // cubes[cubeIndex].index = 0;
            for(int triIndex=0; triIndex<*numTriangles; triIndex+=1)
            {   
                int index = triIndex*(*numCubes) + cubeIndex;
                // if(cubeIndex == 0) printf("Index: %d", index);
                cubes[cubeIndex].triangleIndexes[cubes[cubeIndex].index] = triIndex;
                int intersects = CubeTriMapp[index];
                cubes[cubeIndex].index+=1*intersects;
                // atomicAdd(&cubes[cubeIndex].index, intersects);
                
            }
        }
        
    }
    
    
__global__ void firstCube(Cube *cubes, Ray *rays, Triangle *triangles,int *numRays, int *numCubes)
    {
        int idx = threadIdx.x + blockIdx.x*blockDim.x;
        const unsigned int num_threads = gridDim.x * blockDim.x;
        for(int rayIndex=idx; rayIndex<*numRays; rayIndex+=num_threads)
        {
            rays[rayIndex].index=0;
    
            for(int cubeIndex=0; cubeIndex<*numCubes;)
            {
                if(rayBoxIntersection_yaxis(rays[rayIndex].orig, cubes[cubeIndex]))
                {
                    rays[rayIndex].intersectedCubes[rays[rayIndex].index] = cubeIndex;
                    rays[rayIndex].index +=1;
                    cubeIndex+=*numCubes;
                }
                cubeIndex+=1;
            }
            // printf("RayIndex: %d \n",rays[rayIndex].index);
        }  
    }
    


/************************ END of GPU FUNCTIONS ************************************/


extern "C" void SetRays(float *xyz_flat, int nr_rays)
    {
        numRays = nr_rays;
        rays = (Ray *)malloc(sizeof(Ray)*numRays);

        for(int i=0; i<nr_rays; i+=1)
        {
            rays[i].orig[0] = xyz_flat[0+i*3];
            rays[i].orig[1] = xyz_flat[1+i*3];
            rays[i].orig[2] = xyz_flat[2+i*3];
            generateRandoms(rays[i].dir, rays[i].orig);
           
        }
    }

    

extern "C" void SetTriangles(double *tri_coordinates, int nr_triangles)
    {
        numTriangles = nr_triangles;
        triangles = (Triangle *)malloc(sizeof(Triangle)*nr_triangles);

        for(int i=0; i<nr_triangles; i+=1)
        {
            // int tri_index = int(i/9);
            triangles[i].v0[0] = tri_coordinates[i*9];
            triangles[i].v0[1] = tri_coordinates[i*9+1];
            triangles[i].v0[2] = tri_coordinates[i*9+2];
            triangles[i].v1[0] = tri_coordinates[i*9+3];
            triangles[i].v1[1] = tri_coordinates[i*9+4];
            triangles[i].v1[2] = tri_coordinates[i*9+5];
            triangles[i].v2[0] = tri_coordinates[i*9+6];
            triangles[i].v2[1] = tri_coordinates[i*9+7];
            triangles[i].v2[2] = tri_coordinates[i*9+8];

        }
    }

// extern "C" void printTriangles()
void printTriangles()
{
    for(int i=0; i<numTriangles; i++)
    {
        printf("Triangle:%d %lf %lf %lf \n",i, triangles[i].v0[0],triangles[i].v0[1],triangles[i].v0[2]);
        printf("Triangle:%d %lf %lf %lf \n",i, triangles[i].v1[0],triangles[i].v1[1],triangles[i].v1[2]);
        printf("Triangle:%d %lf %lf %lf \n",i, triangles[i].v2[0],triangles[i].v2[1],triangles[i].v2[2]);

    }
}



extern "C" void  SetBoundingBox(double *x, double *y, double *z)
// extern "C" void  SetBoundingBox(int *mesh_bounds)
{
    boundingBox.xmin = x[0];
    boundingBox.xmax = x[1];
    boundingBox.ymin = y[0];
    boundingBox.ymax = y[1];
    boundingBox.zmin = z[0];
    boundingBox.zmax = z[1];
    // printf("XMIN: %lf  XMAX: %lf",boundingBox.xmin,boundingBox.xmax);
    // printf("yMIN: %lf  yMAX: %lf",boundingBox.ymin,boundingBox.ymax);
    // printf("zMIN: %lf  zMAX: %lf",boundingBox.zmin,boundingBox.zmax);


}


/*************** CPU VERSION OF IS IN SURFACE *****************/


void TriangleCount(Triangle *triangles, Cube *cubes, int numCubes, int numTriangles, bool *CubeTriMapp)
    {
        for(int i=0; i<numCubes; i++)
        {
            cubes[i].counter=0;
            for(int j=0; j<numTriangles; j++)
            {
                int index = j*numCubes + i;
                int intersects = triBoxOverlap(cubes[i].center, cubes[i].boxhalfsize, triangles[j]);
                CubeTriMapp[index] =(intersects>0);
                cubes[i].counter+=intersects;
                
            }
        }

    }


 void appendALL_CPU(Cube *cubes,int numCubes,int numTriangles, bool *CubeTriMapp, Triangle *triangles)
    {
        for(int cubeIndex=0; cubeIndex<numCubes; cubeIndex+=1)
        {
            cubes[cubeIndex].index = 0;
            for(int triIndex=0; triIndex<numTriangles; triIndex+=1)
            {   
                int index = triIndex*numCubes + cubeIndex;
                cubes[cubeIndex].triangleIndexes[cubes[cubeIndex].index] = triIndex;
                int intersects = CubeTriMapp[index];
                cubes[cubeIndex].index+=1*intersects;
                
            }
        }
        
    }


void firstCube_CPU(Cube *cubes, Ray *rays, Triangle *triangles,int numRays, int numCubes)
    {

        for(int rayIndex=0; rayIndex<numRays; rayIndex+=1)
        {
            rays[rayIndex].index=0;
    
            for(int cubeIndex=0; cubeIndex<numCubes;)
            {
                if(rayBoxIntersection_yaxis(rays[rayIndex].orig, cubes[cubeIndex]))
                {
                    rays[rayIndex].intersectedCubes[rays[rayIndex].index] = cubeIndex;
                    rays[rayIndex].index +=1;
                    cubeIndex+=numCubes;
                }
                cubeIndex+=1;
            }
        }  
    }


void IsInPlane(Ray *rays, Triangle *tri, bool *isInside, int nr_rays,  Cube *cubes)
{

    const double epsilon = 1.e-10;
    for(int rayIndex=0; rayIndex<nr_rays; rayIndex+=1)
    {
        if(rays[rayIndex].index > 0)
        {
            int counter = 0;
            for(int j=0; j<cubes[rays[rayIndex].intersectedCubes[0]].index;)
            {   
                counter+=(int)(PointIsInSurface(rays[rayIndex], tri[cubes[rays[rayIndex].intersectedCubes[0]].triangleIndexes[j]], epsilon)==true);
                j+=1+counter*cubes[rays[rayIndex].intersectedCubes[0]].index;
    
            }
            isInside[rayIndex]=(counter>0);
        }
    }

}

void findUnique_CPU(Cube *cubes, int end, int cubesPerBlock, bool *CubeTriMapp, int numTriangles, int numCubes)
{
    int start = end -(cubesPerBlock-1);   
    for(int triIndex=0; triIndex<numTriangles; triIndex+=1)
    {
        for(int cubeIndex=start; cubeIndex<end; cubeIndex++)
        {
            CubeTriMapp[triIndex*(numCubes)+(end)]|=CubeTriMapp[triIndex*(numCubes)+cubeIndex];
        }
        cubes[end].index+=(int)CubeTriMapp[triIndex*(numCubes)+(end)];
    }

}

void appendTriangles_CPU(Cube *cubes, int *indexes, int frontGrid, int numTriangles, int numCubes, bool *CubeTriMapp )
    {
        for(int i=0; i<frontGrid; i+=1)
        {
            int cubeIndex=indexes[i];
            cubes[cubeIndex].index=0;
            for(int triIndex=0; triIndex<numTriangles; triIndex+=1)
            {   

                int index = triIndex*numCubes + cubeIndex;
                cubes[cubeIndex].triangleIndexes[cubes[cubeIndex].index] = triIndex;
                cubes[cubeIndex].index+=1*(int)CubeTriMapp[index];
                
            }
        }

        
    }
// void IsInPlane(Ray *rays, Triangle *tri, bool *isInside, int nr_rays, int nr_triangles, double epsilon)
// {
//     for(int rayIndex=0; rayIndex<nr_rays; rayIndex+=1)
//     {
//         if(rays[rayIndex].index > 0)
//         {
//             int counter = 0;
//             for(int j=0; j<nr_triangles; j=j+1)
//             {

//                     counter+=(PointIsInSurface(rays[rayIndex], tri[j], epsilon)==true);
//                     // counter+=(IsInPlaneTest(rays[rayIndex], tri[j], epsilon)==true);

//                     // counter+=(SameSide(rays[rayIndex], tri[j])==true);


//             }
//             isInside[rayIndex]=(counter>0);
//         }
//     }

// }

void getIntersectedCubes(Ray *rays, Cube *cubes, int numrays, int *indexes,  Cube boundingBox, int frontGrid)
    {   
        for(int index = 0; index<numrays; index+=1)
        {
            rays[index].index = 0;            
            for(int frontGridIndex=0; frontGridIndex<frontGrid; frontGridIndex++)
            {
                    int intersects = (int)rayBoxIntersection_yaxis(rays[index].orig, cubes[indexes[frontGridIndex]]);
                    rays[index].intersectedCubes[rays[index].index] = indexes[frontGridIndex];
                    rays[index].index+=1*intersects;
            }


 
        }

    }

void ray_tracing(Ray *ray, Triangle *tri, bool *results, int n_r, int n_tr, Cube *cubes)
{
        for(int rayIndex = 0; rayIndex <n_r; rayIndex+=1)
        {       
            int count = 0;
            if(ray[rayIndex].index > 0)
            {
                for (int j = 0; j<cubes[ray[rayIndex].intersectedCubes[0]].index; j=j+1)
                {
                        count += (int)rayTriangleIntersect(ray[rayIndex],tri[cubes[ray[rayIndex].intersectedCubes[0]].triangleIndexes[j]]);
                }
 
                results[rayIndex] = (count % 2 !=0);       
               
            }
        
    }

    }
/*************** END OF CPU VERSION OF IS IN SURFACE *****************/


// extern "C" void RayTracingGPU(int *numRays, int *numTriangles, bool *isInside, bool *isInPlane){
extern "C" bool *RayTracingGPU(int *numRays_, int *numTriangles){



    // bool GPU = true;
    bool GPU = false;
    int nr_rays = *numRays_;
    int nr_triangles = *numTriangles;

    Point3D upper = {boundingBox.xmax, boundingBox.ymax, boundingBox.zmax};
    Point3D lower = {boundingBox.xmin, boundingBox.ymin, boundingBox.zmin};

    double delta = ((upper.x-lower.x) * (upper.z-lower.z)) / 500;
    delta = sqrt(delta);

    //number of cubes per each block
    int cubesPerBlock = ceil((upper.y - lower.y)/delta);
    int num = getNumberOfCubes(lower, upper,delta);

    Cube *cubes_d;
    if (GPU)
        cudaMallocManaged((void**)&cubes_d, sizeof(Cube)*num);
    else
        cubes_d = (Cube *)malloc(sizeof(Cube)*num);

    Index frontGrid;
    // createGrid3D(cubes, lower, upper, delta,frontGrid);
    createGrid3D(cubes_d, lower, upper, delta,frontGrid);

    bool *CubeTriMapp = (bool *)malloc(sizeof(bool)*nr_triangles*num);

    int frontGridSize = frontGrid.count;
    int *indexes = new int[frontGrid.count];

    //Copy indexes in array
    for(int i = 0; i<frontGrid.count; i++)
    {
        indexes[i] = frontGrid.ptr[i];
    }

    printf("Num cubes: %d \n",num);
    printf("Num Triangles: %d \n",nr_triangles);
    printf("Num Rays: %d \n",nr_rays);

    const double epsilon = 1.e-15 * (boundingBox.xmax - boundingBox.xmin);
  
    /******************* Device memory allocation ********************/
    Triangle *tri_d;
    MY_CUDA_CHECK(cudaMalloc((void**)&tri_d, sizeof(Triangle)*nr_triangles));
    MY_CUDA_CHECK(cudaMemcpy(tri_d, triangles,sizeof(Triangle)*nr_triangles,cudaMemcpyHostToDevice));

    Ray *ray_d;
    MY_CUDA_CHECK(cudaMalloc((void**)&ray_d, sizeof(Ray)*nr_rays));
    MY_CUDA_CHECK(cudaMemcpy(ray_d, rays, sizeof(Ray)*nr_rays, cudaMemcpyHostToDevice));

    bool *CubeTriMapp_d;
    MY_CUDA_CHECK(cudaMalloc((void**)&CubeTriMapp_d, sizeof(bool)*num*nr_triangles));
    MY_CUDA_CHECK(cudaMemcpy(CubeTriMapp_d, CubeTriMapp,sizeof(bool)*num*nr_triangles , cudaMemcpyHostToDevice));

    int *cubesPerBlock_d;
    MY_CUDA_CHECK(cudaMalloc((void**)&cubesPerBlock_d, sizeof(int)));
    MY_CUDA_CHECK(cudaMemcpy(cubesPerBlock_d, &cubesPerBlock, sizeof(int),cudaMemcpyHostToDevice));

    int *frontGridSize_d;
    MY_CUDA_CHECK(cudaMalloc((void**)&frontGridSize_d, sizeof(int)));
    MY_CUDA_CHECK(cudaMemcpy(frontGridSize_d, &frontGridSize, sizeof(int),cudaMemcpyHostToDevice));

    int *indexes_d;
    MY_CUDA_CHECK(cudaMalloc((void**)&indexes_d, sizeof(int)*frontGridSize));
    MY_CUDA_CHECK(cudaMemcpy(indexes_d, indexes, sizeof(int)*frontGridSize,cudaMemcpyHostToDevice));

    int *numRays_d;
    MY_CUDA_CHECK(cudaMalloc((void**)&numRays_d, sizeof(int)));
    MY_CUDA_CHECK(cudaMemcpy(numRays_d, &nr_rays, sizeof(int),cudaMemcpyHostToDevice));

    int *nr_triangles_d;
    MY_CUDA_CHECK(cudaMalloc((void**)&nr_triangles_d, sizeof(int)));
    MY_CUDA_CHECK(cudaMemcpy(nr_triangles_d, &nr_triangles, sizeof(int),cudaMemcpyHostToDevice));

    int *num_cubes_d;
    MY_CUDA_CHECK(cudaMalloc((void**)&num_cubes_d, sizeof(int)));
    MY_CUDA_CHECK(cudaMemcpy(num_cubes_d, &num, sizeof(int),cudaMemcpyHostToDevice));

    Cube *cube3d_d;
    MY_CUDA_CHECK(cudaMalloc((void**)&cube3d_d, sizeof(Cube)));
    MY_CUDA_CHECK(cudaMemcpy(cube3d_d, &boundingBox, sizeof(Cube),cudaMemcpyHostToDevice));

    int *pointsInside;
    MY_CUDA_CHECK(cudaMalloc((void**)&pointsInside, sizeof(int)));

    int *countTri;
    MY_CUDA_CHECK(cudaMalloc((void**)&countTri, sizeof(int)));

    bool *results;
    MY_CUDA_CHECK(cudaMalloc((void**)&results, sizeof(bool)*nr_rays*2));

    bool *isInside_d;
    MY_CUDA_CHECK(cudaMalloc((void**)&isInside_d, sizeof(bool)*nr_rays));

    int *triCubMapp_d;
    MY_CUDA_CHECK(cudaMalloc((void**)&triCubMapp_d, sizeof(int)*num));

    /**************** End of device memory allocation *****************/ 

    isInside = (bool *)malloc(2*nr_rays*sizeof(bool));
    isInPlane = (bool *)malloc(nr_rays*sizeof(bool));

    
    /**************** Begining of computations *****************/ 
    

    if(GPU){

        clock_t begin = clock();

        TriangleCount_GPU<<<256,512>>>(tri_d, cubes_d, num_cubes_d, nr_triangles_d, CubeTriMapp_d); //Count number of triangles per cube
        cudaDeviceSynchronize();

        for(int i=0; i<num; i++)
        {
            MY_CUDA_CHECK(cudaMallocManaged((void**)&cubes_d[i].triangleIndexes, sizeof(int)*cubes_d[i].counter+1));

        }

        appendALL<<<256,256>>>(cubes_d, num_cubes_d, nr_triangles_d, CubeTriMapp_d,tri_d);
        cudaDeviceSynchronize();


        firstCube<<<256,256>>>(cubes_d, ray_d, tri_d, numRays_d, num_cubes_d);
        cudaDeviceSynchronize();
        // exit(0);


        IsInPlaneGPU<<<256,256>>>(ray_d, tri_d, results, numRays_d, nr_triangles_d, cubes_d);    //Find surface points
        cudaDeviceSynchronize(); 

        cudaStream_t streams_u[frontGridSize]; //using cuda streams to parallelise kernel execution
        for(int i=0;i<frontGridSize; i++)
        {
            cudaStreamCreate(&streams_u[i]);
            int reference = indexes[i];
            int *cubeIndex;
            MY_CUDA_CHECK(cudaMalloc((void**)&cubeIndex, sizeof(int)));
            MY_CUDA_CHECK(cudaMemcpy(cubeIndex, &reference, sizeof(int),cudaMemcpyHostToDevice));

            findUnique<<<8,256,0,streams_u[i]>>>(cubes_d,cubeIndex,cubesPerBlock_d, CubeTriMapp_d, nr_triangles_d, num_cubes_d);//find unique triangles per cube

        }
        cudaDeviceSynchronize();

        for(int i=0;i<frontGridSize; i++)
        {
            cudaStreamDestroy(streams_u[i]); //Cuda streams must be destroyed at the end of execution
        }


        for(int i=0; i<frontGridSize; i++)
        {
            MY_CUDA_CHECK(cudaMalloc((void**)&cubes_d[indexes[i]].triangleIndexes, sizeof(int)*cubes_d[indexes[i]].index));
        }
        
        int numThreads = (frontGridSize + 32 - (frontGridSize%32))/2;
        appendTriangles<<<2,196>>>(cubes_d,indexes_d, frontGridSize_d,nr_triangles_d, num_cubes_d, CubeTriMapp_d); //append triangles to cubes
        cudaDeviceSynchronize();
    
        getIntersectedCubes_GPU<<<32,256>>>(ray_d, cubes_d, numRays_d , indexes_d, cube3d_d, frontGridSize_d); //ray cube intersection 
        cudaDeviceSynchronize();
        
        ray_tracingGPU<<<256,256>>>(ray_d, tri_d, isInside_d, numRays_d, nr_triangles_d, cubes_d);    // ray tracing - ray triangle intersection Moller algorithm
        cudaDeviceSynchronize();   



        clock_t end = clock();
        double time_spent = (double)(end-begin)/ CLOCKS_PER_SEC;

        MY_CUDA_CHECK(cudaMemcpy(isInside, results, sizeof(bool)*nr_rays,cudaMemcpyDeviceToHost));
        MY_CUDA_CHECK(cudaMemcpy(isInPlane, isInside_d, sizeof(bool)*nr_rays,cudaMemcpyDeviceToHost));

        // printf("\n time spent: %.10f\n",time_spent);
    }

    else{

        clock_t begin1 = clock();
        
        TriangleCount(triangles, cubes_d, num, nr_triangles, CubeTriMapp);

        for(int i=0; i<num; i++)
        {
            cubes_d[i].triangleIndexes = (int *)malloc(sizeof(int)*cubes_d[i].counter+1);
        }

        appendALL_CPU(cubes_d,num, nr_triangles, CubeTriMapp, triangles);   
        firstCube_CPU(cubes_d, rays, triangles, nr_rays, num);

        clock_t end1 = clock();
        double time_spent1 = (double)(end1-begin1)/ CLOCKS_PER_SEC;
        printf(" \n time spent for preprocessing: %.10f\n",time_spent1);

        clock_t begin2 = clock();
        IsInPlane(rays, triangles, isInside, nr_rays,  cubes_d);

        clock_t end2 = clock();
        double time_spent2 = (double)(end2-begin2)/ CLOCKS_PER_SEC;
        printf("time spent for Is In plane: %.10f\n",time_spent2);

        for(int i=0;i<frontGridSize; i++)
        {
            int reference = indexes[i];
            findUnique_CPU(cubes_d, reference, cubesPerBlock, CubeTriMapp, nr_triangles, num);
        }

        for(int i=0; i<frontGridSize; i++)
        {
            cubes_d[indexes[i]].triangleIndexes =(int *)malloc(sizeof(int)*cubes_d[indexes[i]].index);
        }

        clock_t begin3 = clock();

        appendTriangles_CPU(cubes_d, indexes, frontGridSize, nr_triangles, num, CubeTriMapp);
        getIntersectedCubes(rays, cubes_d, nr_rays, indexes,  boundingBox, frontGridSize);
        clock_t end3 = clock();
        double time_spent3 = (double)(end3-begin3)/ CLOCKS_PER_SEC;
        printf("time spent for pre-raytracing: %.10f\n",time_spent3);

        clock_t begin4 = clock();

        ray_tracing(rays, triangles, isInPlane, nr_rays, nr_triangles, cubes_d);

        clock_t end4 = clock();
        double time_spent4 = (double)(end4-begin4)/ CLOCKS_PER_SEC;
        printf("time spent for raytracing: %.10f\n",time_spent4);

    }

    /**************** End of Computations*****************/ 


    for(int i=0; i<nr_rays;i++){
        isInside[i+nr_rays]= isInPlane[i]; //Copying the results to one unique array of size 2 x number of rays
    }

    for(int i=0; i<nr_rays;i++){
        if (isInside[i]==true) isInside[nr_rays+i]= false; //Removing redundance
    }
    // int counter=0;

    // for(int i=0; i<nr_rays;i++){
    //     if (isInside[i]==true) counter+=1;
    // }
    // printf("Counter C: %d \n",counter);

//    FILE *fptr;

//    // use appropriate location if you are using MacOS or Linux
//    if (GPU)
//    fptr = fopen("/home/diko/Desktop/programGPU.txt","w");

//    else
//    fptr = fopen("/home/diko/Desktop/programCPU.txt","w");

//    if(fptr == NULL)
//    {
//       printf("Error!");   
//       exit(1);             
//    }

//    for(int i=0; i<nr_rays; i++){

//     if (isInside[i]==true) fprintf(fptr,"%d\n",i);
//    }
//    fclose(fptr);

//    printf("Ray: %lf %lf %lf",rays[20182].orig[0],rays[20182].orig[1],rays[20182].orig[2]);

    
    free(rays);
    free(triangles);
    free(isInPlane);
    cudaFree(ray_d);
    cudaFree(tri_d);
    cudaFree(results);
    cudaFree(pointsInside);
    cudaFree(cube3d_d);
    cudaFree(cubes_d);
    cudaFree(cubesPerBlock_d);
    cudaFree(indexes_d);
    cudaFree(frontGridSize_d);
    cudaFree(numRays_d);
    cudaFree(triCubMapp_d);
    cudaFree(isInside_d);

    return isInside;

}
