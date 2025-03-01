#include <Eigen/Core>
#include <ImageProcessing/CPU/TetrahedralImplCPU.hpp>
#include <ImageProcessing/CPU/WorkerData.hpp>
#include <algorithm>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

namespace {
float getSafeDelta(int boundingBoxA, int boundingBoxB, float floatCoordinate) {
	const int boundingBoxWidth = boundingBoxB - boundingBoxA;
	if (floatCoordinate - boundingBoxA == 0 || boundingBoxWidth == 0) {
		return .0f;
	}
	// x_d = (x - x_0) / (x_1 - x_0)
	return (floatCoordinate - boundingBoxA) / static_cast<float>(boundingBoxWidth);
}

using namespace Eigen;

Matrix<float, 1, 3> getTriple(const Table3D& lut, int r, int g, int b) {
	using Arr4 = Eigen::array<Index, 4>;
	return Matrix<float, 1, 3>(lut(Arr4{r, g, b, 0}), lut(Arr4{r, g, b, 1}), lut(Arr4{r, g, b, 2}));
}

std::vector<Matrix<float, 4, 8>> getConstantMatrices() {
    std::vector<Matrix<float, 4, 8>> matrices;
    matrices.emplace_back(Matrix<float, 4, 8>{
        {1, 0, 0, 0, 0, 0, 0, 0},
        {-1, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, -1, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, -1, 1}
    });
    matrices.emplace_back(Matrix<float, 4, 8>{
        {1, 0, 0, 0, 0, 0, 0, 0},
        {-1, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, -1, 0, 1},
        {0, 0, 0, 0, -1, 1, 0, 0}
    });
    matrices.emplace_back(Matrix<float, 4, 8>{
        {1, 0, 0, 0, 0, 0, 0, 0},
        {0, -1, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, -1, 0, 1},
        {-1, 1, 0, 0, 0, 0, 0, 0}
    });
    matrices.emplace_back(Matrix<float, 4, 8>{
        {1, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, -1, 0, 0, 0, 1, 0},
        {-1, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, -1, 1}
    });
    matrices.emplace_back(Matrix<float, 4, 8>{
        {1, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, -1, 0, 0, 0, 1},
        {-1, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, -1, 1, 0, 0, 0, 0}
    });
    matrices.emplace_back(Matrix<float, 4, 8>{
        {1, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, -1, 0, 0, 0, 1},
        {0, -1, 0, 1, 0, 0, 0, 0},
        {-1, 1, 0, 0, 0, 0, 0, 0}
    });
    return matrices;
}

size_t getConstantMatrixIdx(float d_r, float d_g, float d_b) {
	size_t idx = 0;
	if (d_r > d_g) {
		if (d_g > d_b) {
			idx = 5; // r>g>b
		} else if (d_b > d_r) {
			idx = 1; // b>r>g
		} else {
			idx = 4; // r>b>g
		}
	} else {
		if (d_b > d_g) {
			idx = 2; // b>g>r
		} else if (d_b > d_r) {
			idx = 3; // g>b>r
		} else {
			idx = 6; // g>r>b
		}
	}

	return idx - 1;
}

uchar domainScale(float value) {
	const auto roundedVal = round(value * 255.0f);
	const auto clamped = std::clamp(roundedVal, 0.0f, 255.0f);
	return static_cast<uchar>(clamped);
}

} // namespace

TetrahedralImplCPU::TetrahedralImplCPU(Table3D* lut) : LUT3DPipelineCPU(lut) {
    matrices = getConstantMatrices();
}

void TetrahedralImplCPU::calculatePixel(const int x, const int y, const Table3D& lut, const WorkerData& data) {
	const size_t pixelIndex = (x + y * data.width) * data.channels;

	const int b = data.image[pixelIndex + 0];
	const int g = data.image[pixelIndex + 1];
	const int r = data.image[pixelIndex + 2];

	// Implementation of a formula from the following presentation:
	// https://community.acescentral.com/t/3d-lut-interpolation-pseudo-code/2160

	// Get the real float 3D index to be interpolated (located inside the 'bounding cube')
	const float scaled_r = r * data.lutScale;
	const float scaled_g = g * data.lutScale;
	const float scaled_b = b * data.lutScale;

	// Map real RGB coordinates to an integral 'bounding cube' on a lower-accuracy LUT plane
	// (map RGB point from a 256^3 color cube to e.g. a 33^3 cube)
	const int r1 = static_cast<int>(ceil(scaled_r));
	const int r0 = static_cast<int>(floor(scaled_r));
	const int g1 = static_cast<int>(ceil(scaled_g));
	const int g0 = static_cast<int>(floor(scaled_g));
	const int b1 = static_cast<int>(ceil(scaled_b));
	const int b0 = static_cast<int>(floor(scaled_b));

	// get distance from the real point to the 'left' coordinate of the bounding cube
	const float d_r = getSafeDelta(r0, r1, scaled_r);
	const float d_g = getSafeDelta(g0, g1, scaled_g);
	const float d_b = getSafeDelta(b0, b1, scaled_b);

    Matrix<float, 8, 3> bBoxVal;
    bBoxVal << 
        getTriple(lut, r0, g0, b0),
        getTriple(lut, r0, g1, b0),
        getTriple(lut, r1, g0, b0),
        getTriple(lut, r1, g1, b0),
        getTriple(lut, r0, g0, b1),
        getTriple(lut, r0, g1, b1),
        getTriple(lut, r1, g0, b1),
        getTriple(lut, r1, g1, b1);

	const Matrix<float, 1, 4> deltas{{1.0f, d_b, d_r, d_g}};
	const auto& t = matrices[getConstantMatrixIdx(d_r, d_g, d_b)];
	const auto v = deltas * t * bBoxVal;

	// Change the final interpolated LUT float value to 8-bit RGB
	const auto newB = domainScale(v(2));
	const auto newG = domainScale(v(1));
	const auto newR = domainScale(v(0));

	// Assign final pixel values to the output image
	data.newImage[pixelIndex + 0] = static_cast<uchar>(b + (newB - b) * data.opacity);
	data.newImage[pixelIndex + 1] = static_cast<uchar>(g + (newG - g) * data.opacity);
	data.newImage[pixelIndex + 2] = static_cast<uchar>(r + (newR - r) * data.opacity);
}