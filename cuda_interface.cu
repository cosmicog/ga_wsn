// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>
#include <math.h>
#define MAX_DISTANCE 164025000000.0
#define NO_HEAD -1

extern "C"
//Adds two arrays
float * runCudaPart(float *h_in, float *h_out);
double runCalcClusterHeadsAndTotalEnergy(int *h_out, float *h_in_x, float * h_in_y, int *h_in_ch, int arr_size_ch, int arr_size, int base_x, int base_y);

__global__ void square(float *d_out, float *d_in)
{
    int idx = threadIdx.x;
    float f = d_in[idx];
    d_out[idx] = f * f; //node to ch d^2
}

__global__ void calcToCh(float *d_out_float, int *d_out, float *d_in_x, float * d_in_y, int * d_in_ch, int arr_size_ch, int base_x, int base_y)
{
    int idx = threadIdx.x;
    float dist2 = MAX_DISTANCE, cur_dist2;
    for(int i = 0; i < arr_size_ch; i++)
    {
        if(d_in_x[idx] == base_x && d_in_y[idx] == base_y)
        {
            dist2 = 0.0;
            d_out[idx] = idx;
            d_out_float[idx] = 0.0;
            continue;
        }
        if (d_in_ch[i] == NO_HEAD)
        {
            continue;
        }
        cur_dist2 = ( powf(fabs(d_in_x[d_in_ch[i]]-d_in_x[idx]), 2) + powf(fabs(d_in_y[d_in_ch[i]]-d_in_y[idx]), 2) );
        if( cur_dist2 < dist2)
        {
            dist2 = cur_dist2;
            d_out[idx] = d_in_ch[i];
            d_out_float[idx] = (50e-9 * 2000) + ((100e-12) * 2000 * dist2) + (50e-9 * 2000);
        }
    }
}

double runCalcClusterHeadsAndTotalEnergy(int *h_out, float *h_in_x, float * h_in_y, int *h_in_ch, int arr_size_ch, int arr_size, int base_x, int base_y)
{
    const int ARRAY_SIZE_POINTS = arr_size;
    const int ARRAY_BYTES_POINTS = ARRAY_SIZE_POINTS * sizeof(float);
    const int ARRAY_SIZE_CH = arr_size_ch;
    const int ARRAY_BYTES_CH = ARRAY_SIZE_CH * sizeof(float);

    int * d_in_ch;//declare GPU memory pointers
    float * d_in_y;//declare GPU memory pointers
    float * d_in_x;//declare GPU memory pointers
    int * d_out_int;//declare GPU memory pointers
    float * d_out_float;//declare GPU memory pointers
    float h_out_float[ARRAY_SIZE_POINTS];

    cudaMalloc((void**) &d_in_ch, ARRAY_BYTES_CH);   // allocate GPU memory
    cudaMalloc((void**) &d_out_int, ARRAY_BYTES_POINTS);
    cudaMalloc((void**) &d_out_float, ARRAY_BYTES_POINTS);
    cudaMalloc((void**) &d_in_x, ARRAY_BYTES_POINTS);
    cudaMalloc((void**) &d_in_y, ARRAY_BYTES_POINTS);

    cudaMemcpy(d_in_ch, h_in_ch, ARRAY_BYTES_CH, cudaMemcpyHostToDevice);// Transfer the array to GPU
    cudaMemcpy(d_in_y, h_in_y, ARRAY_BYTES_POINTS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_x, h_in_x, ARRAY_BYTES_POINTS, cudaMemcpyHostToDevice);

    calcToCh <<< 1, ARRAY_SIZE_POINTS >>> (d_out_float, d_out_int, d_in_x, d_in_y, d_in_ch, ARRAY_SIZE_CH, base_x, base_y);// Launch the Kernel

    cudaMemcpy(h_out, d_out_int, ARRAY_BYTES_POINTS, cudaMemcpyDeviceToHost);// copy back the result array to the CPU
    cudaMemcpy(h_out_float, d_out_float, ARRAY_BYTES_POINTS, cudaMemcpyDeviceToHost);

    double total = 0.0;
    for (int i = 1; i < ARRAY_SIZE_POINTS; i++)
    {
        total += h_out_float[i];
    }
    float bd2 = 0.0;
    for (int i = 1; i < ARRAY_SIZE_CH; i++)
    {
        if (h_in_ch[i] == NO_HEAD) continue;
        bd2 =  ( powf(fabs(h_in_x[h_in_ch[i]]-base_x), 2) + powf(fabs(h_in_y[h_in_ch[i]]-base_y), 2) );
        total += (50e-9 * 2000) + ((100e-12) * 2000 * bd2);// cluster heads to base (d^2)
    }

    cudaFree(d_in_ch);
    cudaFree(d_in_x);
    cudaFree(d_in_y);
    cudaFree(d_out_int);

    return total;
}

// Main cuda function
float * runCudaPart(float *h_in, float *h_out)
{
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	//declare GPU memory pointers
	float * d_in;
	float * d_out;

	// allocate GPU memory
	cudaMalloc((void**) &d_in, ARRAY_BYTES);
	cudaMalloc((void**) &d_out, ARRAY_BYTES);

    // Transfer the array to GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES , cudaMemcpyHostToDevice);

	// Launch the Kernel
    square <<< 1, ARRAY_SIZE >>> (d_out, d_in);
    /**  square<<<NUMBER_OF_BLOCKS, NUMBER_OF_THREADS>>>(d_out, d_in);
     *   NUMBER_OF_THREADS   max 1024, eski kartlar için 512
     *   NUMBER_OF_BLOCKS    kullanılacak blok sayısı
     *   square<<< 2, 16 >>>(d_out, d_in); TOPLAMDA 32 THREAD OLUR...
     * */

	// copy back the result array to the CPU
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	cudaFree(d_in);
    cudaFree(d_out);
    return h_out;
}

//__global__ void calcEnergyToCh_d2(float *d_out, float *d_in_x, float * d_in_y)
//{
//    int idx = threadIdx.x;
//    float f = d_in_x[idx];
//    // dizi başında ch var. diğer elemanların toplamını
//    d_out[idx] = ( powf(fabs(d_in_x[0]-d_in_x[idx]), 2) + powf(fabs(d_in_y[0]-d_in_y[idx]), 2) ); //node to ch d^2
//}

//double runCalcChTotalEnergy(float *h_out, float *h_in_x, float * h_in_y, int arr_size, int base_x, int base_y)
//{
//    const int ARRAY_SIZE = arr_size;
//    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
//    float * d_in_y;//declare GPU memory pointers
//    float * d_in_x;//declare GPU memory pointers
//    float * d_out;//declare GPU memory pointers
//    cudaMalloc((void**) &d_out, ARRAY_BYTES); // allocate GPU memory
//    cudaMalloc((void**) &d_in_x, ARRAY_BYTES); // allocate GPU memory
//    cudaMalloc((void**) &d_in_y, ARRAY_BYTES); // allocate GPU memory
//    cudaMemcpy(d_in_y, h_in_y, ARRAY_BYTES , cudaMemcpyHostToDevice);// Transfer the array to GPU
//    cudaMemcpy(d_in_x, h_in_x, ARRAY_BYTES , cudaMemcpyHostToDevice);// Transfer the array to GPU
//    calcDistToCh_d2 <<< 1, ARRAY_SIZE >>> (d_out, d_in_x, d_in_y);// Launch the Kernel
//    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);// copy back the result array to the CPU
//    cudaFree(d_in_x);
//    cudaFree(d_in_y);
//    cudaFree(d_out);
//    double total = 0.0;
//    for (int i = 1; i < ARRAY_SIZE; i++)
//    {
//        total += h_out[i];
//    }
//    total += ( pow(abs(h_in_x[0]-base_x), 2) + pow(abs(h_in_y[0]-base_y), 2) ); // ch to base (d^2)
//    printf("\n=====%f=======\n", total);
//    return total;
//}
