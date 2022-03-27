#include <GPUImageProcess/LUT3D/applyTrilinearGpu.cuh>

namespace GpuTrilinearDevice {
	void run(dim3 threads, dim3 blocks, unsigned char* image, const char channels, float* LUT, const char LUTsize, const float opacity, const std::tuple<int, int>& imgSize)
	{
		applyTrilinear <<<blocks, threads>>> (image, channels, LUT, LUTsize, opacity, std::get<0>(imgSize), std::get<1>(imgSize));
		cudaDeviceSynchronize();
	}
}

__global__ void applyTrilinear(unsigned char* image, const char channels, float* LUT, const char LUTsize, const float opacity, const int xMax, const int yMax)
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

	// Implementation of a formula in the "Method" section: https://en.wikipedia.org/wiki/Trilinear_interpolation
	// Get the min/max indices of the interpolated 3D "box"
	const int R1 = static_cast<int>(ceilf(r / 255.0f * (float)(LUTsize - 1)));
	const int R0 = static_cast<int>(floorf(r / 255.0f * (float)(LUTsize - 1)));
	const int G1 = static_cast<int>(ceilf(g / 255.0f * (float)(LUTsize - 1)));
	const int G0 = static_cast<int>(floorf(g / 255.0f * (float)(LUTsize - 1)));
	const int B1 = static_cast<int>(ceilf(b / 255.0f * (float)(LUTsize - 1)));
	const int B0 = static_cast<int>(floorf(b / 255.0f * (float)(LUTsize - 1)));

	// Get the real 3D index to be interpolated
	const float r_o = r * (LUTsize - 1) / 255.0f;
	const float g_o = g * (LUTsize - 1) / 255.0f;
	const float b_o = b * (LUTsize - 1) / 255.0f;

	const float delta_r = (r_o - R0 == 0 || R1 - R0 == 0 ? 0 : (r_o - R0) / (float)(R1 - R0));
	const float delta_g = (g_o - G0 == 0 || G1 - G0 == 0 ? 0 : (g_o - G0) / (float)(G1 - G0));
	const float delta_b = (b_o - B0 == 0 || B1 - B0 == 0 ? 0 : (b_o - B0) / (float)(B1 - B0));

	auto idx = [LUTsize](int r, int g, int b) {
		// Full formula for getting the index in a flattened 4D (2x2x2x3) RowMajor tensor:
		// idx = LUTsize * LUTsize * LUTsize * ch + LUTsize * LUTsize * b + LUTsize * g + r;

		// We explicitly skip the first part to easily get next channels by adding this part later.
		// Example: You'll get the idx of R as a return value. To get G, you just need to add
		// LUTsize^3 * 1. Similarly, to get B, you need to add LUTsize^3 * 2.
		return LUTsize * LUTsize * b + LUTsize * g + r;
	};

	// channel increment (value that should be added to the index to get the next channel)
	int chIncr{static_cast<int>(pow(LUTsize, 3))}; 
	// 1st pass
	int ind1 = idx(R0, G0, B0);
	int ind2 = idx(R1, G0, B0);

	using namespace Eigen;
	Vector3f v1 = Vector3f(LUT[ind1], LUT[ind1 + chIncr], LUT[ind1 + 2 * chIncr]) * (1 - delta_r) + \
						Vector3f(LUT[ind2], LUT[ind2 + chIncr], LUT[ind2 + 2 * chIncr]) * delta_r;
	ind1 = idx(R0, G0, B1);
	ind2 = idx(R1, G0, B1);
	Vector3f v2 = Vector3f(LUT[ind1], LUT[ind1 + chIncr], LUT[ind1 + 2 * chIncr]) * (1 - delta_r) + \
						Vector3f(LUT[ind2], LUT[ind2 + chIncr], LUT[ind2 + 2 * chIncr]) * delta_r;
	ind1 = idx(R0, G1, B0);
	ind2 = idx(R1, G1, B0);
	Vector3f v3 = Vector3f(LUT[ind1], LUT[ind1 + chIncr], LUT[ind1 + 2 * chIncr]) * (1 - delta_r) + \
						Vector3f(LUT[ind2], LUT[ind2 + chIncr], LUT[ind2 + 2 * chIncr]) * delta_r;
	ind1 = idx(R0, G1, B1);
	ind2 = idx(R1, G1, B1);
	Vector3f v4 = Vector3f(LUT[ind1], LUT[ind1 + chIncr], LUT[ind1 + 2 * chIncr]) * (1 - delta_r) + \
						Vector3f(LUT[ind2], LUT[ind2 + chIncr], LUT[ind2 + 2 * chIncr]) * delta_r;

	// 2nd pass
	v1 = v1 * (1 - delta_g) + v3 * delta_g;
	v2 = v2 * (1 - delta_g) + v4 * delta_g;

	// 3rd pass
	v1 = v1 * (1 - delta_b) + v2 * delta_b;

	const uchar newB = roundf(v1[2] * 255.0f);
	const uchar newG = roundf(v1[1] * 255.0f);
	const uchar newR = roundf(v1[0] * 255.0f);

	uchar finalB = static_cast<uchar>(b + (newB - b) * opacity);
	uchar finalG = static_cast<uchar>(g + (newG - g) * opacity);
	uchar finalR = static_cast<uchar>(r + (newR - r) * opacity);

	image[pixel_index] = finalB;
	image[pixel_index + 1] = finalG;
	image[pixel_index + 2] = finalR;

}
