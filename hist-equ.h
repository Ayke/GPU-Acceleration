#ifndef HIST_EQU_COLOR_H
#define HIST_EQU_COLOR_H

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>
#define NUM_THREAD 1024

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    

typedef struct{
    int w;
    int h;
    unsigned char * img_r;
    unsigned char * img_g;
    unsigned char * img_b;
} PPM_IMG;

typedef struct{
    int w;
    int h;
    unsigned char * img_y;
    unsigned char * img_u;
    unsigned char * img_v;
} YUV_IMG;


typedef struct
{
    int width;
    int height;
    float * h;
    float * s;
    unsigned char * l;
} HSL_IMG;

    

PPM_IMG read_ppm(const char * path);
void write_ppm(PPM_IMG img, const char * path);
void free_ppm(PPM_IMG img);

PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);

HSL_IMG rgb2hsl(PPM_IMG img_in);
PPM_IMG hsl2rgb(HSL_IMG img_in);
YUV_IMG rgb2yuv(PPM_IMG img_in);
PPM_IMG yuv2rgb(YUV_IMG img_in);    


HSL_IMG rgb2hsl_gpu(PPM_IMG img_in);
PPM_IMG hsl2rgb_gpu(HSL_IMG img_in);
YUV_IMG rgb2yuv_gpu(PPM_IMG img_in);
PPM_IMG yuv2rgb_gpu(YUV_IMG img_in);
__global__ void rgb2hsl_gpu_son( unsigned char * d_r, unsigned char * d_g, unsigned char * d_b,
    float * d_h , float * d_s , unsigned char * d_l , 
    int size, int numOfThreads);
__global__ void hsl2rgb_gpu_son(float * d_h , float * d_s ,unsigned char * d_l , 
    unsigned char * d_r, unsigned char * d_g, unsigned char * d_b, 
    int size, int numOfThreads);
__global__ void rgb2yuv_gpu_son(unsigned char * d_r, unsigned char * d_g, unsigned char * d_b, 
    unsigned char * d_y , unsigned char * d_u ,unsigned char * d_v , 
    int size, int numOfThreads);
__global__ void yuv2rgb_gpu_son(unsigned char * d_y , unsigned char * d_u ,unsigned char * d_v , 
    unsigned char * d_r, unsigned char * d_g, unsigned char * d_b, 
    int size, int numOfThreads);


void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin);
void histogram_gpu(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
__global__ void histogram_gpu_son(unsigned char * d_img, unsigned int * d_hist,  int img_size,  int nbr_bin);
void histogram_equalization_gpu(unsigned char * img_out, unsigned char * img_in, int * hist_in, int img_size, int nbr_bin);
__global__ void histogram_equalization_gpu_son (unsigned char * d_in, unsigned char * d_out, int * d_lut, int img_size,  int nbr_bin);

void gpu_histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
__global__ void gpu_histogram_equalization(unsigned char * img_out, unsigned char * img_in,
                            int img_size, int nbr_bin, int numOfThreads, int * lut);

//Contrast enhancement for gray-scale images
//Contrast enhancement for color images
PGM_IMG contrast_enhancement_g(PGM_IMG img_in);
PPM_IMG contrast_enhancement_c_rgb(PPM_IMG img_in);
PPM_IMG contrast_enhancement_c_yuv(PPM_IMG img_in);
PPM_IMG contrast_enhancement_c_hsl(PPM_IMG img_in);


PGM_IMG contrast_enhancement_g_gpu(PGM_IMG img_in);
PPM_IMG contrast_enhancement_c_rgb_gpu(PPM_IMG img_in);
PPM_IMG contrast_enhancement_c_yuv_gpu(PPM_IMG img_in);
PPM_IMG contrast_enhancement_c_hsl_gpu(PPM_IMG img_in);


#endif
