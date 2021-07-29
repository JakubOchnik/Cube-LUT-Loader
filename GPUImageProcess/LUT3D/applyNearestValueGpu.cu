#include "applyNearestValueGpu.cuh"

namespace NearestValGpu {
	void run(dim3 grid, unsigned char* image, char channels, float* LUT, char LUTsize, float opacity)
	{
		applyNearestKernel << <grid, 1 >> > (image, channels, LUT, LUTsize, opacity);
		cudaDeviceSynchronize();
	}
}


__global__ void applyNearestKernel(unsigned char* image, char channels, float* LUT, char LUTsize, float opacity)
{
	// get the number of pixel in flattened image array
	const int x = blockIdx.x;
	const int y = blockIdx.y;
	const int pixel_index = (x + y * gridDim.x) * channels;

	// get original colors
	const int b = image[pixel_index];
	const int g = image[pixel_index + 1];
	const int r = image[pixel_index + 2];

	// get estimated indices of LUT table corresponding to original colors
	const unsigned int b_ind = roundf(b * (LUTsize - 1) / 255.0f);
	const unsigned int g_ind = roundf(g * (LUTsize - 1) / 255.0f);
	const unsigned int r_ind = roundf(r * (LUTsize - 1) / 255.0f);

	const int base_index = LUTsize * LUTsize * channels * r_ind + LUTsize * channels * g_ind + channels * b_ind;

	const unsigned char newB = LUT[base_index + 2] * 255;
	const unsigned char newG = LUT[base_index + 1] * 255;
	const unsigned char newR = LUT[base_index + 0] * 255;

	unsigned char finalB = b + (newB - b) * opacity;
	unsigned char finalG = g + (newG - g) * opacity;
	unsigned char finalR = r + (newR - r) * opacity;

	image[pixel_index] = finalB;
	image[pixel_index + 1] = finalG;
	image[pixel_index + 2] = finalR;
}
