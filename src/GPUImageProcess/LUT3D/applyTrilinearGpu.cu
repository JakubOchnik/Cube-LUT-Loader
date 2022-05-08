#include <GPUImageProcess/LUT3D/applyTrilinearGpu.cuh>
#include <Eigen/Dense>
#include <GPUImageProcess/Utils/CudaUtils.hpp>

namespace GpuTrilinearDevice
{
	void run(dim3 threads, dim3 blocks, unsigned char* image, const char channels, float* LUT, const int LUTsize,
	         const float opacity, const std::tuple<int, int>& imgSize)
	{
		applyTrilinear<<<blocks, threads>>>(image, channels, LUT, LUTsize, opacity, std::get<0>(imgSize),
		                                    std::get<1>(imgSize));
		cudaErrorChk(cudaPeekAtLastError());
		cudaErrorChk(cudaDeviceSynchronize());
	}
}

__global__ void applyTrilinear(unsigned char* image, const char channels, const float* LUT, const int LUTsize,
                               const float opacity, const int width, const int height)
{
	using uchar = unsigned char;
	// Get the index of pixel in flattened image array
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		// Check if the kernel coordinates are out of image bounds
		return;
	}

	const int pixelIdx = (x + y * width) * channels;

	// Get original colors of the processed pixel
	const int b = image[pixelIdx];
	const int g = image[pixelIdx + 1];
	const int r = image[pixelIdx + 2];

	// Implementation of a formula in the "Method" section: https://en.wikipedia.org/wiki/Trilinear_interpolation
	// Get the min/max indices of the interpolated 3D "box"
	const int r1 = static_cast<int>(ceilf(r / 255.0f * static_cast<float>(LUTsize - 1)));
	const int r0 = static_cast<int>(floorf(r / 255.0f * static_cast<float>(LUTsize - 1)));
	const int g1 = static_cast<int>(ceilf(g / 255.0f * static_cast<float>(LUTsize - 1)));
	const int g0 = static_cast<int>(floorf(g / 255.0f * static_cast<float>(LUTsize - 1)));
	const int b1 = static_cast<int>(ceilf(b / 255.0f * static_cast<float>(LUTsize - 1)));
	const int b0 = static_cast<int>(floorf(b / 255.0f * static_cast<float>(LUTsize - 1)));

	// Get the real 3D index to be interpolated
	const float r_o = r * (LUTsize - 1) / 255.0f;
	const float g_o = g * (LUTsize - 1) / 255.0f;
	const float b_o = b * (LUTsize - 1) / 255.0f;

	// TODO comparing floats with 0 is theoretically unsafe
	const float delta_r = (r_o - r0 == 0 || r1 - r0 == 0 ? 0 : (r_o - r0) / static_cast<float>(r1 - r0));
	const float delta_g = (g_o - g0 == 0 || g1 - g0 == 0 ? 0 : (g_o - g0) / static_cast<float>(g1 - g0));
	const float delta_b = (b_o - b0 == 0 || b1 - b0 == 0 ? 0 : (b_o - b0) / static_cast<float>(b1 - b0));

	auto idx = [LUTsize](int r, int g, int b)
	{
		// Full formula for getting the index in a flattened 4D (2x2x2x3) RowMajor tensor:
		// idx = LUTsize * LUTsize * LUTsize * ch + LUTsize * LUTsize * b + LUTsize * g + r;

		// We explicitly skip the first part to easily get next channels by adding this part later.
		// Example: You'll get the idx of R as a return value. To get G, you just need to add
		// LUTsize^3 * 1. Similarly, to get B, you need to add LUTsize^3 * 2.
		return LUTsize * LUTsize * b + LUTsize * g + r;
	};

	// Channel increment (a value that should be added to the index to get the next channel)
	const int chIncr{static_cast<int>(pow(LUTsize, 3))};
	// 1st pass
	int ind1 = idx(r0, g0, b0);
	int ind2 = idx(r1, g0, b0);

	using namespace Eigen;
	Vector3f v1 = Vector3f(LUT[ind1], LUT[ind1 + chIncr], LUT[ind1 + 2 * chIncr]) * (1 - delta_r) +
		Vector3f(LUT[ind2], LUT[ind2 + chIncr], LUT[ind2 + 2 * chIncr]) * delta_r;
	ind1 = idx(r0, g0, b1);
	ind2 = idx(r1, g0, b1);
	Vector3f v2 = Vector3f(LUT[ind1], LUT[ind1 + chIncr], LUT[ind1 + 2 * chIncr]) * (1 - delta_r) +
		Vector3f(LUT[ind2], LUT[ind2 + chIncr], LUT[ind2 + 2 * chIncr]) * delta_r;
	ind1 = idx(r0, g1, b0);
	ind2 = idx(r1, g1, b0);
	Vector3f v3 = Vector3f(LUT[ind1], LUT[ind1 + chIncr], LUT[ind1 + 2 * chIncr]) * (1 - delta_r) +
		Vector3f(LUT[ind2], LUT[ind2 + chIncr], LUT[ind2 + 2 * chIncr]) * delta_r;
	ind1 = idx(r0, g1, b1);
	ind2 = idx(r1, g1, b1);
	Vector3f v4 = Vector3f(LUT[ind1], LUT[ind1 + chIncr], LUT[ind1 + 2 * chIncr]) * (1 - delta_r) +
		Vector3f(LUT[ind2], LUT[ind2 + chIncr], LUT[ind2 + 2 * chIncr]) * delta_r;

	// 2nd step
	v1 = v1 * (1 - delta_g) + v3 * delta_g;
	v2 = v2 * (1 - delta_g) + v4 * delta_g;

	// 3rd step
	v1 = v1 * (1 - delta_b) + v2 * delta_b;

	const auto newB = static_cast<uchar>(roundf(v1[2] * 255.0f));
	const auto newG = static_cast<uchar>(roundf(v1[1] * 255.0f));
	const auto newR = static_cast<uchar>(roundf(v1[0] * 255.0f));

	image[pixelIdx] = static_cast<uchar>(b + (newB - b) * opacity);
	image[pixelIdx + 1] = static_cast<uchar>(g + (newG - g) * opacity);
	image[pixelIdx + 2] = static_cast<uchar>(r + (newR - r) * opacity);
}
