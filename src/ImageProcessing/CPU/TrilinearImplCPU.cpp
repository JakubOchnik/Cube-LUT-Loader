#include <ImageProcessing/CPU/TrilinearImplCPU.hpp>
#include <ImageProcessing/CPU/WorkerData.hpp>
#include <functional>
#include <thread>
#include <vector>

void TrilinearImplCPU::calculateArea(const int x, const Table3D& lut, const WorkerData& data,
									 const int segWidth) {
	for (int localX{x}; localX < x + segWidth; ++localX) {
		for (int y{0}; y < data.height; ++y) {
			calculatePixel(localX, y, lut, data);
		}
	}
}

namespace {
float getSafeDelta(int boundingBoxA, int boundingBoxB, float floatCoordinate) {
	const int boundingBoxWidth = boundingBoxB - boundingBoxA;
	if (floatCoordinate - boundingBoxA == 0 || boundingBoxWidth == 0) {
		return .0f;
	}
	// x_d = (x - x_0) / (x_1 - x_0)
	return (floatCoordinate - boundingBoxA) / static_cast<float>(boundingBoxWidth);
}
} // namespace

void TrilinearImplCPU::calculatePixel(const int x, const int y, const Table3D& lut, const WorkerData& data) {
	const int b = data.image[(x + y * data.width) * data.channels + 0];
	const int g = data.image[(x + y * data.width) * data.channels + 1];
	const int r = data.image[(x + y * data.width) * data.channels + 2];

	// Implementation of a formula from the "Method" section:
	// https://en.wikipedia.org/wiki/Trilinear_interpolation

	const int maxLUTIndex = data.lutSize - 1;
	// Map real RGB coordinates to an integral 'bounding cube' on a lower-accuracy LUT plane
	// (map RGB point from a 256^3 color cube to e.g. a 33^3 cube)
	const int r1 = static_cast<int>(ceil(r / 255.0f * static_cast<float>(maxLUTIndex)));
	const int r0 = static_cast<int>(floor(r / 255.0f * static_cast<float>(maxLUTIndex)));
	const int g1 = static_cast<int>(ceil(g / 255.0f * static_cast<float>(maxLUTIndex)));
	const int g0 = static_cast<int>(floor(g / 255.0f * static_cast<float>(maxLUTIndex)));
	const int b1 = static_cast<int>(ceil(b / 255.0f * static_cast<float>(maxLUTIndex)));
	const int b0 = static_cast<int>(floor(b / 255.0f * static_cast<float>(maxLUTIndex)));

	// Get the real float 3D index to be interpolated (located inside the 'bounding cube')
	const float real_r = r * (maxLUTIndex) / 255.0f;
	const float real_g = g * (maxLUTIndex) / 255.0f;
	const float real_b = b * (maxLUTIndex) / 255.0f;

	// get distance from the real point to the 'left' coordinate of the bounding cube
	const float delta_r = getSafeDelta(r0, r1, real_r);
	const float delta_g = getSafeDelta(g0, g1, real_g);
	const float delta_b = getSafeDelta(b0, b1, real_b);

	using namespace Eigen;
	using Arr4 = Eigen::array<Index, 4>;
	using Vec3fWrap = Tensor<float, 1>;

	// 1st step - interpolate along r axis
	// vx variables are actually RGB triplets of the LUT values (in float) - we can treat the RGB vector like a single
	// value vertice_lut_value_vector3 = lut[r_0][g_0][b_0] * (1 - delta_r) + lut[r_1][g_0][b_0] * delta_r
	Eigen::array<Eigen::Index, 4> tensorExtents = {1, 1, 1, 3};
	Vec3fWrap v1 = (lut.slice(Arr4{r0, g0, b0, 0}, tensorExtents) * (1 - delta_r) +
					lut.slice(Arr4{r1, g0, b0, 0}, tensorExtents) * delta_r)
					   .reshape(Eigen::array<Index, 1>{3});
	Vec3fWrap v2 = (lut.slice(Arr4{r0, g0, b1, 0}, tensorExtents) * (1 - delta_r) +
					lut.slice(Arr4{r1, g0, b1, 0}, tensorExtents) * delta_r)
					   .reshape(Eigen::array<Index, 1>{3});
	const Vec3fWrap v3 = (lut.slice(Arr4{r0, g1, b0, 0}, tensorExtents) * (1 - delta_r) +
						  lut.slice(Arr4{r1, g1, b0, 0}, tensorExtents) * delta_r)
							 .reshape(Eigen::array<Index, 1>{3});
	const Vec3fWrap v4 = (lut.slice(Arr4{r0, g1, b1, 0}, tensorExtents) * (1 - delta_r) +
						  lut.slice(Arr4{r1, g1, b1, 0}, tensorExtents) * delta_r)
							 .reshape(Eigen::array<Index, 1>{3});

	// 2nd step - interpolate along g axis
	v1 = v1 * (1 - delta_g) + v3 * delta_g;
	v2 = v2 * (1 - delta_g) + v4 * delta_g;

	// 3rd step - interpolate along b axis
	v1 = v1 * (1 - delta_b) + v2 * delta_b;

	// Change the final interpolated LUT float value to 8-bit RGB
	const auto newB = static_cast<uchar>(round(v1(2) * 255));
	const auto newG = static_cast<uchar>(round(v1(1) * 255));
	const auto newR = static_cast<uchar>(round(v1(0) * 255));

	// Assign final pixel values to the output image
	data.newImage[(x + y * data.width) * data.channels + 0] = static_cast<uchar>(b + (newB - b) * data.opacity);
	data.newImage[(x + y * data.width) * data.channels + 1] = static_cast<uchar>(g + (newG - g) * data.opacity);
	data.newImage[(x + y * data.width) * data.channels + 2] = static_cast<uchar>(r + (newR - r) * data.opacity);
}