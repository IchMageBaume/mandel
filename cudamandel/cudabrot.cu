#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__managed__ int colorScheme = 0;

__managed__ int xPerLaunch = 512;

__managed__ int width = 1024;
__managed__ int height = 1024;

__managed__ int iterations = 3000;

__managed__ double xMin = 0.29375;
__managed__ double xMax = 0.29385;
__managed__ double yMin = 0.01495;
__managed__ double yMax = 0.01505;

__device__ uint32_t rgbFromIter(int iter)
{
	if(colorScheme == 0 || colorScheme == 1) {
		//Monochrome
		int j = iter * 256 / iterations;
		
		//white-to-black instead of black-to-white gradient
		if(colorScheme == 1) j = 256 - j;
		
		return ((int)j << 16) | ((int)j << 8) | (int)j;
	}
	else if(colorScheme == 2 || colorScheme == 3) {
		//Rainbow
		int h = int((iter * (colorScheme == 2? 1 : 8)) % 256 * 6);
		int x = h % 0x100;
		
		int r = 0, g = 0, b = 0;
		switch (h / 256)
		{
		case 0: 
			r = 255; g = x;
			break;
		case 1:
			g = 255; r = 255 - x;
			break;
		case 2:
			g = 255; b = x;
			break;
		case 3:
			b = 255; g = 255 - x;
			break;
		case 4:
			b = 255; r = x;
			break;
		case 5: 
			r = 255; b = 255 - x;
			break;
		}
		
		return r + (g << 8) + (b << 16);
	}
	
	return 0;
}

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
	
	return rgbFromIter(i);
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
		if(argc < 9) {
			if(strcmp(argv[1], "-h") && strcmp(argv[1], "--help"))
				printf("Give me all arguments or none. (or use '-' for default value)\n"
					"Syntax:\n  %s <width> <colorScheme> <iterations> <xMin> <xMax>"
					" <yMin> <yMax> <xPerLaunch>\nOr -h/--help for more info\n",
					argv[0]);
			else
				printf("Syntax:\n  %s <width> <colorScheme> <iterations> <xMin>"
					" <xMax> <yMin> <yMax> <xPerLaunch>\n"
					"width: width and height of output png, in pixels\n"
					"colorScheme:\n  0: monochrome\n  1: monochrome, inverted\n"
					"  2: rainbow\n  3: rainbow, faster color change\n"
					"iterations: max iterations. More is slower, but more accurate\n"
					"xMin-yMax: boundary in complex plane (y is the imaginary part)\n"
					"xPerLaunch: pixels per row calculated per launch of cuda"
					" kernels. If that takes > 2s on standard windows,"
					" this program crashes. So lower values are more stable"
					" but slower.\n"
					"default values: %d %d %d %f %f %f %f %d\n", argv[0],
					width, colorScheme, iterations, xMin, xMax, yMin, yMax,
					xPerLaunch);
			return 1;
		}
		
		if(strcmp(argv[1], "-")) width = atoi(argv[1]); height = width;
		if(strcmp(argv[2], "-")) colorScheme = atoi(argv[2]);
		if(strcmp(argv[3], "-")) iterations = atoi(argv[3]);
		if(strcmp(argv[4], "-")) xMin = atof(argv[4]);
		if(strcmp(argv[5], "-")) xMax = atof(argv[5]);
		if(strcmp(argv[6], "-")) yMin = atof(argv[6]);
		if(strcmp(argv[7], "-")) yMax = atof(argv[7]);
		if(strcmp(argv[8], "-")) xPerLaunch = atoi(argv[8]);
	}
	
	uint8_t *dat_d;
	cudaError_t err = cudaMalloc(&dat_d, width * height * 3);
	if (err != cudaSuccess) {
		fprintf(stderr, "Couldn't allocate memory on your card. Either you have"
			" chosen a stupidly"
			" high resolution, you don't have cuda drivers"
			" or you card is not supported.\n");
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
		
		fprintf(stderr, "done (%2d/%2d)\r", i + 1, 
			width/xPerLaunch + (width % xPerLaunch == 0? 0 : 1));
	}
	
	fprintf(stderr, "\n");
	
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
