// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>

extern "C"
//Adds two arrays
char * runCudaPart(float *h_in, float *h_out);

__global__ void square(float *d_out , float *d_in)
{
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = f * f ;
}

// Main cuda function

char * runCudaPart(float *h_in, float *h_out)
{
    char * display = new char[4096];
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

//	//generate the input array on the host/CPU
//	float h_in[ARRAY_SIZE];
//	float h_out[ARRAY_SIZE];
//	for (int i = 0; i < ARRAY_SIZE; i++)
//	 {
//			h_in[i] = float(i);
//			h_out[i] = 99;
//	 }

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

    char * asdf = new char[256];
	// Print the result from CPU
    sprintf(display, "");
	for(int i=0 ; i< ARRAY_SIZE ; i++)
	{
        sprintf(asdf, "%f ", h_out[i]);
        std::strcat(display, asdf);
        sprintf(asdf, ((i % 4) != 3) ? "\t" : "\n");
        std::strcat(display, asdf);
	}
    std::strcat(display, "\n");
	cudaFree(d_in);
	cudaFree(d_out);

    return display;
}
