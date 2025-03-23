#include <ImageProcessing/GPU/kernels/nearestValue.cuh>

namespace GpuNearestValDevice
{
	void run(dim3 threads, dim3 blocks, unsigned char *image, const char channels, float *LUT, const char LUTsize,
			 const float opacity, const std::tuple<int, int> &imgSize)
	{
		applyNearestKernel<<<blocks, threads>>>(image, channels, LUT, LUTsize, opacity, std::get<0>(imgSize),
												std::get<1>(imgSize));
		cudaErrorChk(cudaPeekAtLastError());
		cudaErrorChk(cudaDeviceSynchronize());
	}
}

__global__ void applyNearestKernel(unsigned char *image, char channels, const float *LUT, char LUTsize, float opacity,
								   const int width, const int height)
{
	using uchar = unsigned char;
	using uint = unsigned int;
	// Get the index of pixel in flattened image array
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		// Check if the kernel coordinates are out of image bounds
		return;
	}

	const int pixelIndex = (x + y * width) * channels;
	// Get original colors of the processed pixel
	const int b = image[pixelIndex];
	const int g = image[pixelIndex + 1];
	const int r = image[pixelIndex + 2];

	const int lutScale = (LUTsize - 1) / 255.0f;
	// get estimated indices of LUT table corresponding to original colors
	const uint bIdx = static_cast<uint>(roundf(b * lutScale));
	const uint gIdx = static_cast<uint>(roundf(g * lutScale));
	const uint rIdx = static_cast<uint>(roundf(r * lutScale));

	// Full formula for getting the index in a flattened 4D (2x2x2x3) RowMajor tensor:
	// idx = LUTsize * LUTsize * LUTsize * ch + LUTsize * LUTsize * b + LUTsize * g + r;
	const int baseIndex = LUTsize * LUTsize * bIdx + LUTsize * gIdx + rIdx;
	const int chIncr{static_cast<int>(pow(LUTsize, 3))};

	const auto newB = static_cast<int>(LUT[baseIndex + 2 * chIncr] * 255);
	const auto newG = static_cast<int>(LUT[baseIndex + chIncr] * 255);
	const auto newR = static_cast<int>(LUT[baseIndex + 0] * 255);

	image[pixelIndex] = static_cast<uchar>(b + (newB - b) * opacity);
	image[pixelIndex + 1] = static_cast<uchar>(g + (newG - g) * opacity);
	image[pixelIndex + 2] = static_cast<uchar>(r + (newR - r) * opacity);
}
