#include <ImageProcessing/GPUImageProcess/LUT3D/applyTrilinearGpu.cuh>
#include <Eigen/Dense>
#include <ImageProcessing/GPUImageProcess/Utils/CudaUtils.hpp>

namespace GpuTrilinearDevice
{
	void run(dim3 threads, dim3 blocks, unsigned char *image, const char channels, float *LUT, const int LUTsize,
			 const float opacity, const std::tuple<int, int> &imgSize)
	{
		applyTrilinear<<<blocks, threads>>>(image, channels, LUT, LUTsize, opacity, std::get<0>(imgSize),
											std::get<1>(imgSize));
		cudaErrorChk(cudaPeekAtLastError());
		cudaErrorChk(cudaDeviceSynchronize());
	}
}

namespace {
__device__ float getSafeDelta(int boundingBoxA, int boundingBoxB, float floatCoordinate) {
	const int boundingBoxWidth = boundingBoxB - boundingBoxA;
	if (floatCoordinate - boundingBoxA == 0 || boundingBoxWidth == 0) {
		return .0f;
	}
	// x_d = (x - x_0) / (x_1 - x_0)
	return (floatCoordinate - boundingBoxA) / static_cast<float>(boundingBoxWidth);
}
}

__global__ void applyTrilinear(unsigned char *image, const char channels, const float *LUT, const int LUTsize,
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

	// Implementation of a formula in the "Method" section
	// https://en.wikipedia.org/wiki/Trilinear_interpolation

	const int maxLUTIndex = LUTsize - 1;
	// Map real RGB coordinates to an integral 'bounding cube' on a lower-accuracy LUT plane
	// (map RGB point from a 256^3 color cube to e.g. a 33^3 cube)
	const int r1 = static_cast<int>(ceilf(r / 255.0f * static_cast<float>(maxLUTIndex)));
	const int r0 = static_cast<int>(floorf(r / 255.0f * static_cast<float>(maxLUTIndex)));
	const int g1 = static_cast<int>(ceilf(g / 255.0f * static_cast<float>(maxLUTIndex)));
	const int g0 = static_cast<int>(floorf(g / 255.0f * static_cast<float>(maxLUTIndex)));
	const int b1 = static_cast<int>(ceilf(b / 255.0f * static_cast<float>(maxLUTIndex)));
	const int b0 = static_cast<int>(floorf(b / 255.0f * static_cast<float>(maxLUTIndex)));

	// Get the real 3D index to be interpolated
	const float real_r = r * (maxLUTIndex) / 255.0f;
	const float real_g = g * (maxLUTIndex) / 255.0f;
	const float real_b = b * (maxLUTIndex) / 255.0f;

	const float delta_r = getSafeDelta(r0, r1, real_r);
	const float delta_g = getSafeDelta(g0, g1, real_g);
	const float delta_b = getSafeDelta(b0, b1, real_b);

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
	int ind1 = idx(r0, g0, b0);
	int ind2 = idx(r1, g0, b0);

	// 1st pass - interpolate along r axis
	// vx variables are actually RGB triplets of the LUT values (in float) - we can treat the RGB vector like a single value
	// vertice_lut_value_vector3 = lut[r_0][g_0][b_0] * (1 - delta_r) + lut[r_1][g_0][b_0] * delta_r
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

	// 2nd step - interpolate along g axis
	v1 = v1 * (1 - delta_g) + v3 * delta_g;
	v2 = v2 * (1 - delta_g) + v4 * delta_g;

	// 3rd step - interpolate along b axis
	v1 = v1 * (1 - delta_b) + v2 * delta_b;

	// Change the final interpolated LUT float value to 8-bit RGB
	const auto newB = static_cast<uchar>(roundf(v1[2] * 255.0f));
	const auto newG = static_cast<uchar>(roundf(v1[1] * 255.0f));
	const auto newR = static_cast<uchar>(roundf(v1[0] * 255.0f));

	image[pixelIdx] = static_cast<uchar>(b + (newB - b) * opacity);
	image[pixelIdx + 1] = static_cast<uchar>(g + (newG - g) * opacity);
	image[pixelIdx + 2] = static_cast<uchar>(r + (newR - r) * opacity);
}
