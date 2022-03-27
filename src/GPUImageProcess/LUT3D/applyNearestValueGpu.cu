#include <GPUImageProcess/LUT3D/applyNearestValueGpu.cuh>

namespace GpuNearestValDevice {
	void run(dim3 threads, dim3 blocks, unsigned char* image, const char channels, float* LUT, const char LUTsize, const float opacity, const std::tuple<int, int>& imgSize)
	{
		applyNearestKernel <<<blocks, threads>>> (image, channels, LUT, LUTsize, opacity, std::get<0>(imgSize), std::get<1>(imgSize));
		cudaDeviceSynchronize();
	}
}


__global__ void applyNearestKernel(unsigned char* image, char channels, float* LUT, char LUTsize, float opacity, const int xMax, const int yMax)
{
	using uchar = unsigned char;
	// Get the index of pixel in flattened image array
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= xMax || y >= yMax)
	{
		// Check if the kernel coordinates are out of image bounds
		return;
	}

	const int pixel_index = (x + y * (gridDim.x * blockDim.x)) * channels;
	// Get original colors of the processed pixel
	const int b = image[pixel_index];
	const int g = image[pixel_index + 1];
	const int r = image[pixel_index + 2];

	// get estimated indices of LUT table corresponding to original colors
	const unsigned int b_ind = roundf(b * (LUTsize - 1) / 255.0f);
	const unsigned int g_ind = roundf(g * (LUTsize - 1) / 255.0f);
	const unsigned int r_ind = roundf(r * (LUTsize - 1) / 255.0f);

	// Full formula for getting the index in a flattened 4D (2x2x2x3) RowMajor tensor:
	// idx = LUTsize * LUTsize * LUTsize * ch + LUTsize * LUTsize * b + LUTsize * g + r;
	const int base_index = LUTsize * LUTsize * b_ind + LUTsize * g_ind + r_ind;
	int chIncr{static_cast<int>(pow(LUTsize, 3))}; 

	const unsigned char newB = LUT[base_index + 2 * chIncr] * 255;
	const unsigned char newG = LUT[base_index + chIncr] * 255;
	const unsigned char newR = LUT[base_index + 0] * 255;

	unsigned char finalB = b + (newB - b) * opacity;
	unsigned char finalG = g + (newG - g) * opacity;
	unsigned char finalR = r + (newR - r) * opacity;

	image[pixel_index] = finalB;
	image[pixel_index + 1] = finalG;
	image[pixel_index + 2] = finalR;
}
