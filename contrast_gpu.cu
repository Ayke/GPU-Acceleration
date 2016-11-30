#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"


__device__ unsigned char clip_rgb_gpu(int x)
{
    if(x > 255)
        return 255;
    if(x < 0)
        return 0;

    return (unsigned char)x;
}

__device__ float Hue_2_RGB_gpu( float v1, float v2, float vH )             //Function Hue_2_RGB
{
    if ( vH < 0 ) vH += 1;
    if ( vH > 1 ) vH -= 1;
    if ( ( 6 * vH ) < 1 ) return ( v1 + ( v2 - v1 ) * 6 * vH );
    if ( ( 2 * vH ) < 1 ) return ( v2 );
    if ( ( 3 * vH ) < 2 ) return ( v1 + ( v2 - v1 ) * ( ( 2.0f/3.0f ) - vH ) * 6 );
    return ( v1 );
}

PPM_IMG hsl2rgb_gpu(HSL_IMG img_in)
{
    PPM_IMG result;
    
    result.w = img_in.width;
    result.h = img_in.height;
    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    float * d_h;
    float * d_s;
    unsigned char * d_l;

    unsigned char * d_r;
    unsigned char * d_g;
    unsigned char * d_b;

    cudaMalloc(&d_h, result.w * result.h * sizeof(float));
    cudaMalloc(&d_s, result.w * result.h * sizeof(float));
    cudaMalloc(&d_l, result.w * result.h * sizeof(unsigned char));
    cudaMalloc(&d_r, result.w * result.h * sizeof(unsigned char));
    cudaMalloc(&d_g, result.w * result.h * sizeof(unsigned char));
    cudaMalloc(&d_b, result.w * result.h * sizeof(unsigned char));

    cudaMemcpy(d_h, img_in.h, sizeof(float)         * result.w * result.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, img_in.s, sizeof(float)         * result.w * result.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, img_in.l, sizeof(unsigned char) * result.w * result.h, cudaMemcpyHostToDevice);

    int numThreads = NUM_THREAD;    
    int numBlocks = (result.w * result.h)/numThreads + 1;

    hsl2rgb_gpu_son<<<numBlocks,numThreads>>>(d_h, d_s, d_l, d_r, d_g, d_b, result.w*result.h);

    cudaMemcpy(result.img_r, d_r, sizeof(unsigned char)*result.w*result.h, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.img_g, d_g, sizeof(unsigned char)*result.w*result.h, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.img_b, d_b, sizeof(unsigned char)*result.w*result.h, cudaMemcpyDeviceToHost);
    cudaFree(d_h);
    cudaFree(d_s);
    cudaFree(d_l);
    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    return result;
}

__global__ void hsl2rgb_gpu_son(float * d_h , float * d_s ,unsigned char * d_l , 
    unsigned char * d_r, unsigned char * d_g, unsigned char * d_b, 
    int size) 
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    if (x >= size) return;
    float H = d_h[x];
    float S = d_s[x];
    float L = d_l[x]/255.0f;
    float var_1, var_2;
    unsigned char r,g,b;
        
    if ( S == 0 )
    {
        r = L * 255;
        g = L * 255;
        b = L * 255;
    }
    else
    {
        
        if ( L < 0.5 )
            var_2 = L * ( 1 + S );
        else
            var_2 = ( L + S ) - ( S * L );

        var_1 = 2 * L - var_2;
        r = 255 * Hue_2_RGB_gpu( var_1, var_2, H + (1.0f/3.0f) );
        g = 255 * Hue_2_RGB_gpu( var_1, var_2, H );
        b = 255 * Hue_2_RGB_gpu( var_1, var_2, H - (1.0f/3.0f) );
    }
    d_r[x] = r;
    d_g[x] = g;
    d_b[x] = b;
}

PPM_IMG yuv2rgb_gpu(YUV_IMG img_in)
{
    PPM_IMG result;
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    unsigned char * d_y;
    unsigned char * d_u;
    unsigned char * d_v;

    unsigned char * d_r;
    unsigned char * d_g;
    unsigned char * d_b;

    cudaMalloc(&d_y, result.w * result.h * sizeof(unsigned char));
    cudaMalloc(&d_u, result.w * result.h * sizeof(unsigned char));
    cudaMalloc(&d_v, result.w * result.h * sizeof(unsigned char));
    cudaMalloc(&d_r, result.w * result.h * sizeof(unsigned char));
    cudaMalloc(&d_g, result.w * result.h * sizeof(unsigned char));
    cudaMalloc(&d_b, result.w * result.h * sizeof(unsigned char));

    cudaMemcpy(d_y, img_in.img_y, sizeof(unsigned char) * result.w * result.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, img_in.img_u, sizeof(unsigned char) * result.w * result.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, img_in.img_v, sizeof(unsigned char) * result.w * result.h, cudaMemcpyHostToDevice);

    int numThreads = NUM_THREAD;    
    int numBlocks = (result.w * result.h)/numThreads  + 1;

    yuv2rgb_gpu_son<<<numBlocks,numThreads>>>(d_y, d_u, d_v, d_r, d_g, d_b, result.w*result.h);

    cudaMemcpy(result.img_r, d_r, sizeof(unsigned char)*result.w*result.h, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.img_g, d_g, sizeof(unsigned char)*result.w*result.h, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.img_b, d_b, sizeof(unsigned char)*result.w*result.h, cudaMemcpyDeviceToHost);
    cudaFree(d_y);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    return result;
}

__global__ void yuv2rgb_gpu_son(unsigned char * d_y , unsigned char * d_u ,unsigned char * d_v , 
    unsigned char * d_r, unsigned char * d_g, unsigned char * d_b, 
    int size) 
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    if (x >= size) return;
    int rt,gt,bt;
    int y, cb, cr;
    
    y  = ((int)d_y[x]);
    cb = ((int)d_u[x]) - 128;
    cr = ((int)d_v[x]) - 128;
    
    rt  = (int)( y + 1.402*cr);
    gt  = (int)( y - 0.344*cb - 0.714*cr);
    bt  = (int)( y + 1.772*cb);

    d_r[x] = clip_rgb_gpu(rt);
    d_g[x] = clip_rgb_gpu(gt);
    d_b[x] = clip_rgb_gpu(bt);
}

HSL_IMG rgb2hsl_gpu(PPM_IMG img_in)
{
    HSL_IMG result;
    
    result.width   = img_in.w;
    result.height  = img_in.h;
    result.h = (float *)           malloc(img_in.w * img_in.h * sizeof(float));
    result.s = (float *)           malloc(img_in.w * img_in.h * sizeof(float));
    result.l = (unsigned char *)   malloc(img_in.w * img_in.h * sizeof(unsigned char));
    
    unsigned char * d_r;
    unsigned char * d_g;
    unsigned char * d_b;

    float * d_h;
    float * d_s;
    unsigned char * d_l;
    

    cudaMalloc(&d_r, img_in.w * img_in.h * sizeof(unsigned char));
    cudaMalloc(&d_g, img_in.w * img_in.h * sizeof(unsigned char));
    cudaMalloc(&d_b, img_in.w * img_in.h * sizeof(unsigned char));
    cudaMalloc(&d_h, img_in.w * img_in.h * sizeof(float));
    cudaMalloc(&d_s, img_in.w * img_in.h * sizeof(float));
    cudaMalloc(&d_l, img_in.w * img_in.h * sizeof(unsigned char));
    
    cudaMemcpy(d_r, img_in.img_r, sizeof(unsigned char) * img_in.w * img_in.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, img_in.img_g, sizeof(unsigned char) * img_in.w * img_in.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, img_in.img_b, sizeof(unsigned char) * img_in.w * img_in.h, cudaMemcpyHostToDevice);

    int numThreads = NUM_THREAD;    
    int numBlocks = (img_in.w * img_in.h)/numThreads + 1;

    rgb2hsl_gpu_son<<<numBlocks,numThreads>>>(d_r, d_g, d_b, d_h, d_s, d_l, img_in.w*img_in.h);

    cudaMemcpy(result.h, d_h, sizeof(float) * img_in.w * img_in.h, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.s, d_s, sizeof(float) * img_in.w * img_in.h, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.l, d_l, sizeof(unsigned char) * img_in.w * img_in.h, cudaMemcpyDeviceToHost);
    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_h);
    cudaFree(d_s);
    cudaFree(d_l);
    return result;
}

__global__ void rgb2hsl_gpu_son( unsigned char * d_r, unsigned char * d_g, unsigned char * d_b,
    float * d_h , float * d_s , unsigned char * d_l , 
    int size) 
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    if (x >= size) return;

    float H,S,L;
    float var_r = ( (float)d_r[x]/255 );//Convert RGB to [0,1]
    float var_g = ( (float)d_g[x]/255 );
    float var_b = ( (float)d_b[x]/255 );
    float var_min = (var_r < var_g) ? var_r : var_g;
    var_min = (var_min < var_b) ? var_min : var_b;   //min. value of RGB
    float var_max = (var_r > var_g) ? var_r : var_g;
    var_max = (var_max > var_b) ? var_max : var_b;   //max. value of RGB
    float del_max = var_max - var_min;               //Delta RGB value
        
    L = ( var_max + var_min ) / 2;
    if ( del_max == 0 )//This is a gray, no chroma...
    {
        H = 0;         
        S = 0;    
    }
    else                                    //Chromatic data...
    {
        if ( L < 0.5 )
            S = del_max/(var_max+var_min);
        else {
            S = del_max/(2-var_max-var_min );

            float del_r = (((var_max-var_r)/6)+(del_max/2))/del_max;
            float del_g = (((var_max-var_g)/6)+(del_max/2))/del_max;
            float del_b = (((var_max-var_b)/6)+(del_max/2))/del_max;
            if( var_r == var_max ) H = del_b - del_g;
            else {       
                if( var_g == var_max ) H = (1.0/3.0) + del_r - del_b;
                else H = (2.0/3.0) + del_g - del_r;
            }
        }
        
        if ( H < 0 ) H += 1;
        if ( H > 1 ) H -= 1;
    }
    d_h[x] = H;
    d_s[x] = S;
    d_l[x] = (unsigned char)(L*255);
}

YUV_IMG rgb2yuv_gpu(PPM_IMG img_in)
{
    YUV_IMG result;
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img_y = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_u = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_v = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    unsigned char * d_r;
    unsigned char * d_g;
    unsigned char * d_b;
    
    unsigned char * d_y;
    unsigned char * d_u;
    unsigned char * d_v;

    cudaMalloc(&d_r, result.w * result.h * sizeof(unsigned char));
    cudaMalloc(&d_g, result.w * result.h * sizeof(unsigned char));
    cudaMalloc(&d_b, result.w * result.h * sizeof(unsigned char));
    cudaMalloc(&d_y, result.w * result.h * sizeof(unsigned char));
    cudaMalloc(&d_u, result.w * result.h * sizeof(unsigned char));
    cudaMalloc(&d_v, result.w * result.h * sizeof(unsigned char));

    cudaMemcpy(d_r, img_in.img_r, sizeof(unsigned char) * result.w * result.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, img_in.img_g, sizeof(unsigned char) * result.w * result.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, img_in.img_b, sizeof(unsigned char) * result.w * result.h, cudaMemcpyHostToDevice);

    int numThreads = NUM_THREAD;    
    int numBlocks = (result.w * result.h)/numThreads + 1;

    rgb2yuv_gpu_son<<<numBlocks,numThreads>>>(d_r, d_g, d_b, d_y, d_u, d_v, result.w*result.h);

    cudaMemcpy(result.img_y, d_y, sizeof(unsigned char) * result.w * result.h, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.img_u, d_u, sizeof(unsigned char) * result.w * result.h, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.img_v, d_v, sizeof(unsigned char) * result.w * result.h, cudaMemcpyDeviceToHost);
    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_y);
    cudaFree(d_u);
    cudaFree(d_v);
    return result;
}

__global__ void rgb2yuv_gpu_son(unsigned char * d_r, unsigned char * d_g, unsigned char * d_b, 
    unsigned char * d_y , unsigned char * d_u ,unsigned char * d_v , 
    int size) 
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    if (x >= size) return;
    unsigned char r, g, b;
    
    r = d_r[x];
    g = d_g[x];
    b = d_b[x];
    
    d_y[x] = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
    d_u[x] = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
    d_v[x] = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);
}


PGM_IMG contrast_enhancement_g_gpu(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[256];
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    histogram_gpu(hist, img_in.img, img_in.h * img_in.w, 256);
    histogram_equalization_gpu(result.img, img_in.img, hist, result.w*result.h, 256);
    return result;
}

PPM_IMG contrast_enhancement_c_yuv_gpu(PPM_IMG img_in)
{
    YUV_IMG yuv_med;
    PPM_IMG result;
    
    unsigned char * y_equ;
    int hist[256];

    yuv_med = rgb2yuv_gpu(img_in);
    y_equ = (unsigned char *)malloc(yuv_med.h*yuv_med.w*sizeof(unsigned char));
    
    histogram_gpu(hist, yuv_med.img_y, yuv_med.h * yuv_med.w, 256);
    histogram_equalization_gpu(y_equ,yuv_med.img_y,hist,yuv_med.h * yuv_med.w, 256);

    free(yuv_med.img_y);
    yuv_med.img_y = y_equ;
    
    result = yuv2rgb_gpu(yuv_med);
    free(yuv_med.img_y);
    free(yuv_med.img_u);
    free(yuv_med.img_v);
    
    return result;
}

PPM_IMG contrast_enhancement_c_hsl_gpu(PPM_IMG img_in)
{
    HSL_IMG hsl_med;
    PPM_IMG result;
    
    unsigned char * l_equ;
    int hist[256];

    hsl_med = rgb2hsl_gpu(img_in);
    l_equ = (unsigned char *)malloc(hsl_med.height*hsl_med.width*sizeof(unsigned char));

    histogram_gpu(hist, hsl_med.l, hsl_med.height * hsl_med.width, 256);
    histogram_equalization_gpu(l_equ, hsl_med.l,hist,hsl_med.width*hsl_med.height, 256);
    
    free(hsl_med.l);
    hsl_med.l = l_equ;

    result = hsl2rgb_gpu(hsl_med);
    free(hsl_med.h);
    free(hsl_med.s);
    free(hsl_med.l);
    return result;
}
