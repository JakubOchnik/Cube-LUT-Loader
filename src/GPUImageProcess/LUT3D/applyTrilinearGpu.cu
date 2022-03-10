#include <GPUImageProcess/LUT3D/applyTrilinearGpu.cuh>

namespace TrilinearGpu {
	void run(dim3 grid, unsigned char* image, char channels, float* LUT, char LUTsize, float opacity)
	{
		applyTrilinear << <grid, 1 >> > (image, channels, LUT, LUTsize, opacity);
		cudaDeviceSynchronize();
	}
}


__device__ float* mul(float* a, int offset, const float val)
{
	float* triplet = (float*)malloc(3 * sizeof(float));
	for (int i = 0; i < 3; ++i)
	{
		triplet[i] = a[offset + i] * val;
	}
	return triplet;
}

__device__ float* sum(float* a, float* b)
{
	float* triplet = (float*)malloc(3 * sizeof(float));
	for (int i = 0; i < 3; ++i)
	{
		triplet[i] = a[i] + b[i];
	}
	free(a);
	free(b);
	return triplet;
}

__device__ int l_index(int r, int g, int b, char LUTsize, char channels)
{
	return LUTsize * LUTsize * channels * r + LUTsize * channels * g + channels * b;
}

__global__ void applyTrilinear(unsigned char* image, char channels, float* LUT, char LUTsize, float opacity)
{
	// get the number of pixel in flattened image array
	const int x = blockIdx.x;
	const int y = blockIdx.y;
	const int pixel_index = (x + y * gridDim.x) * channels;

	// get original colors
	const int b = image[pixel_index];
	const int g = image[pixel_index + 1];
	const int r = image[pixel_index + 2];


	const int R1 = ceilf(r / 255.0f * (float)(LUTsize - 1));
	const int R0 = floorf(r / 255.0f * (float)(LUTsize - 1));
	const int G1 = ceilf(g / 255.0f * (float)(LUTsize - 1));
	const int G0 = floorf(g / 255.0f * (float)(LUTsize - 1));
	const int B1 = ceilf(b / 255.0f * (float)(LUTsize - 1));
	const int B0 = floorf(b / 255.0f * (float)(LUTsize - 1));

	const float r_o = r * (LUTsize - 1) / 255.0f;
	const float g_o = g * (LUTsize - 1) / 255.0f;
	const float b_o = b * (LUTsize - 1) / 255.0f;

	const float delta_r = (r_o - R0 == 0 || R1 - R0 == 0 ? 0 : (r_o - R0) / (float)(R1 - R0));
	const float delta_g = (g_o - G0 == 0 || G1 - G0 == 0 ? 0 : (g_o - G0) / (float)(G1 - G0));
	const float delta_b = (b_o - B0 == 0 || B1 - B0 == 0 ? 0 : (b_o - B0) / (float)(B1 - B0));
	// 1st pass
	int ind = l_index(R0, G0, B0, LUTsize, channels);

	float* vr_gz_bz = sum(mul(LUT, ind, 1 - delta_r), mul(LUT, ind, delta_r));
	ind = l_index(R0, G0, B1, LUTsize, channels);
	float* vr_gz_bo = sum(mul(LUT, ind, 1 - delta_r), mul(LUT, ind, delta_r));

	ind = l_index(R0, G1, B0, LUTsize, channels);
	float* vr_go_bz = sum(mul(LUT, ind, 1 - delta_r), mul(LUT, ind, delta_r));
	ind = l_index(R0, G0, B0, LUTsize, channels);
	float* vr_go_bo = sum(mul(LUT, ind, 1 - delta_r), mul(LUT, ind, delta_r));

	// 2nd pass
	float* vrg_b0 = sum(mul(vr_gz_bz, 0, 1 - delta_g), mul(vr_go_bz, 0, delta_g));


	float* vrg_b1 = sum(mul(vr_gz_bo, 0, 1 - delta_g), mul(vr_go_bo, 0, delta_g));
	free(vr_gz_bz);
	free(vr_gz_bo);
	free(vr_go_bz);
	free(vr_go_bo);

	// 3rd pass
	float* vrgb = sum(mul(vrg_b0, 0, 1 - delta_b), mul(vrg_b1, 0, delta_b));
	free(vrg_b0);
	free(vrg_b1);
	const unsigned char newB = roundf(vrgb[2] * 255.0f);
	const unsigned char newG = roundf(vrgb[1] * 255.0f);
	const unsigned char newR = roundf(vrgb[0] * 255.0f);

	free(vrgb);

	unsigned char finalB = b + (newB - b) * opacity;
	unsigned char finalG = g + (newG - g) * opacity;
	unsigned char finalR = r + (newR - r) * opacity;

	image[pixel_index] = finalB;
	image[pixel_index + 1] = finalG;
	image[pixel_index + 2] = finalR;

}
