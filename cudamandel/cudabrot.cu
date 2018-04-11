#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


__managed__ int xPerLaunch = 512;

__managed__ int width = 1024;
__managed__ int height = 1024;

__managed__ int iterations = 3000;

__managed__ double xMin = 0.29375;
__managed__ double xMax = 0.29385;
__managed__ double yMin = 0.01495;
__managed__ double yMax = 0.01505;

__device__ uint32_t genMandelPixel(double x, double y) {
	double zr = 0;
	double zi = 0;
	int i;
	for (i = 0; i < iterations; i++) {
		double zrsq = zr * zr;
		double zisq = zi * zi;
		if (zrsq + zisq > 4)
			break;
		double temp = zrsq - zisq + x;
		zi = 2 * zr * zi + y;
		zr = temp;
	}
	
	if(i == iterations) return 0;
	
	double j = (double)i / iterations * 255;
	
	return ((int)j << 16) | ((int)j << 8) | (int)j;
}

__global__ void genLine(uint8_t* dat, int launchIdx) {
	int y = blockIdx.x * 512 + threadIdx.x;
	if(y >= height) return;
	for (int x = xPerLaunch * launchIdx; x < xPerLaunch * (launchIdx + 1) 
		&& x < width; x++) {
		
		uint32_t pix = genMandelPixel(xMin + (x + 0.5) / width * (xMax - xMin),
			yMax - (y + 0.5) / height * (yMax - yMin));
		dat[(x + y * width) * 3 + 0] = (uint8_t)(pix >> 16);
		dat[(x + y * width) * 3 + 1] = (uint8_t)(pix >>  8);
		dat[(x + y * width) * 3 + 2] = (uint8_t)(pix >>  0);
	}
}

int main(int argc, char* argv[]) {
	clock_t starttime = clock();
	
	if(argc > 1) {
		if(argc < 8) {
			printf("Give me all arguments or none. (or use '-' for default value)\n"
				"Syntax:\n  %s <width> <iterations> <xMin> <xMax> <yMin> <yMax> "
				"<xPerLaunch>\n",
				argv[0]);
			return 1;
		}
		
		if(strcmp(argv[1], "-")) width = atoi(argv[1]); height = width;
		if(strcmp(argv[2], "-")) iterations = atoi(argv[2]);
		if(strcmp(argv[3], "-")) xMin = atof(argv[3]);
		if(strcmp(argv[4], "-")) xMax = atof(argv[4]);
		if(strcmp(argv[5], "-")) yMin = atof(argv[5]);
		if(strcmp(argv[6], "-")) yMax = atof(argv[6]);
		if(strcmp(argv[7], "-")) xPerLaunch = atoi(argv[7]);
	}
	
	uint8_t *dat_d;
	cudaError_t err = cudaMalloc(&dat_d, width * height * 3);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(err));
		return EXIT_FAILURE;
	}
	
	for(int i = 0; i < width / xPerLaunch + (width % xPerLaunch == 0? 0 : 1); i++) {
		fprintf(stderr, "launching cuda kernels... ");
		
		genLine<<<height/512 + (height%512==0? 0 : 1), 512>>>(dat_d, i);
		
		err = cudaDeviceSynchronize();
		
		if(err != cudaSuccess) {
			fprintf(stderr, "Error\n\nUupsy Wucksie! We made a fucky wucky!...jk."
				" We dont have any code monkeys here.\n\nNvidia says: %s\n\n"
				"Try lowering xPerLaunch. (default is 512)\n",
				cudaGetErrorString(err));
			return 1;
		}
		
		fprintf(stderr, "done (%2d/%2d)\n", i + 1, 
			width/xPerLaunch + (width % xPerLaunch == 0? 0 : 1));
	}
	
	uint8_t *dat_h = (uint8_t*)malloc(width * height * 3);
	if(dat_h == NULL) {
		perror("Error: malloc");
		return EXIT_FAILURE;
	}
	
	err = cudaMemcpy(dat_h, dat_d, width * height * 3, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		fprintf(stderr, "\ncudaMemcpy: %s\n", cudaGetErrorString(err));
		return EXIT_FAILURE;
	}
	
	cudaDeviceReset();
	
	clock_t endtime = clock();
	fprintf(stderr, "Secs: %.3f\nWriting Png (yes is slow)... ",
		(endtime - starttime) / (double)CLOCKS_PER_SEC);
	
	stbi_write_png("mandel.png", width, height, 3, dat_h, width * 3);
	
	fputs("done\n", stderr);
	
	cudaFree(dat_d);
	free(dat_h);
	return 0;
}
