#include <ImageProcess/LUT3D/applyTrilinear.hpp>
#include <thread>
#include <vector>
#include <functional>

void Trilinear::calculatePixel(const int x, const int y, const CubeLUT& lut, const float opacity,
                               const WorkerData& data)
{
	const int b = data.image[(x + y * data.width) * data.channels + 0]; //b
	const int g = data.image[(x + y * data.width) * data.channels + 1]; //g
	const int r = data.image[(x + y * data.width) * data.channels + 2]; //r

	// Implementation of a formula from the "Method" section: https://en.wikipedia.org/wiki/Trilinear_interpolation

	const int r1 = static_cast<int>(ceil(r / 255.0f * static_cast<float>(data.lutSize - 1)));
	const int r0 = static_cast<int>(floor(r / 255.0f * static_cast<float>(data.lutSize - 1)));
	const int g1 = static_cast<int>(ceil(g / 255.0f * static_cast<float>(data.lutSize - 1)));
	const int g0 = static_cast<int>(floor(g / 255.0f * static_cast<float>(data.lutSize - 1)));
	const int b1 = static_cast<int>(ceil(b / 255.0f * static_cast<float>(data.lutSize - 1)));
	const int b0 = static_cast<int>(floor(b / 255.0f * static_cast<float>(data.lutSize - 1)));

	// Get the real 3D index to be interpolated
	const float r_o = r * (data.lutSize - 1) / 255.0f;
	const float g_o = g * (data.lutSize - 1) / 255.0f;
	const float b_o = b * (data.lutSize - 1) / 255.0f;

	// TODO comparing floats with 0 is theoretically unsafe
	const float delta_r{r_o - r0 == 0 || r1 - r0 == 0 ? 0 : (r_o - r0) / static_cast<float>(r1 - r0)};
	const float delta_g{g_o - g0 == 0 || g1 - g0 == 0 ? 0 : (g_o - g0) / static_cast<float>(g1 - g0)};
	const float delta_b{b_o - b0 == 0 || b1 - b0 == 0 ? 0 : (b_o - b0) / static_cast<float>(b1 - b0)};

	using namespace Eigen;
	using Arr4 = Eigen::array<Index, 4>;
	using Vec3fWrap = Tensor<float, 1>;
	// 1st step
	Vec3fWrap v1 = (lut.LUT3D.slice(Arr4{r0, g0, b0, 0}, data.extents) * (1 - delta_r) +
		lut.LUT3D.slice(Arr4{r1, g0, b0, 0}, data.extents) * delta_r).reshape(Eigen::array<Index, 1>{3});
	Vec3fWrap v2 = (lut.LUT3D.slice(Arr4{r0, g0, b1, 0}, data.extents) * (1 - delta_r) +
		lut.LUT3D.slice(Arr4{r1, g0, b1, 0}, data.extents) * delta_r).reshape(Eigen::array<Index, 1>{3});
	const Vec3fWrap v3 = (lut.LUT3D.slice(Arr4{r0, g1, b0, 0}, data.extents) * (1 - delta_r) +
		lut.LUT3D.slice(Arr4{r1, g1, b0, 0}, data.extents) * delta_r).reshape(Eigen::array<Index, 1>{3});
	const Vec3fWrap v4 = (lut.LUT3D.slice(Arr4{r0, g1, b1, 0}, data.extents) * (1 - delta_r) +
		lut.LUT3D.slice(Arr4{r1, g1, b1, 0}, data.extents) * delta_r).reshape(Eigen::array<Index, 1>{3});

	// 2nd step
	v1 = v1 * (1 - delta_g) + v3 * delta_g;
	v2 = v2 * (1 - delta_g) + v4 * delta_g;

	// 3rd step
	v1 = v1 * (1 - delta_b) + v2 * delta_b;

	const auto newB = static_cast<uchar>(round(v1(2) * 255));
	const auto newG = static_cast<uchar>(round(v1(1) * 255));
	const auto newR = static_cast<uchar>(round(v1(0) * 255));

	// Assign final pixel values to the output image
	data.newImage[(x + y * data.width) * data.channels + 0] = static_cast<uchar>(b + (newB - b) * opacity);
	data.newImage[(x + y * data.width) * data.channels + 1] = static_cast<uchar>(g + (newG - g) * opacity);
	data.newImage[(x + y * data.width) * data.channels + 2] = static_cast<uchar>(r + (newR - r) * opacity);
}

void Trilinear::calculateArea(const int x, const CubeLUT& lut, const float opacity, const WorkerData& data,
                              const int segWidth)
{
	for (int localX{x}; localX < x + segWidth; ++localX)
	{
		for (int y{0}; y < data.height; ++y)
		{
			calculatePixel(localX, y, lut, opacity, data);
		}
	}
}

cv::Mat Trilinear::applyTrilinear(cv::Mat img, const CubeLUT& lut, const float opacity, const uint threadPool)
{
	// Initialize data
	cv::Mat tmp = img.clone();
	unsigned char* image = img.data;
	unsigned char* newImage = tmp.data;
	WorkerData commonData{
		image, newImage, tmp.cols, tmp.rows, img.channels(),
		static_cast<int>(lut.LUT3D.dimension(0)), {1, 1, 1, 3}
	};

	// Processing
	// Divide the picture into threadPool vertical windows and process them simultaneously.
	// threadPool - 1 threads will process (WIDTH / threadPool) slices 
	// and the last one will process (WIDTH/threadPool + (WIDTH%threadPool))
	const int threadWidth = static_cast<int>(tmp.cols / threadPool);
	const int remainder = static_cast<int>(tmp.cols % threadPool);

	// Create a vector of threads to be executed
	std::vector<std::thread> threads;
	threads.reserve(threadPool);

	// Launch threads
	int x{0};
	for (size_t tNum{0}; tNum < threadPool - 1; x += threadWidth, ++tNum)
	{
		threads.emplace_back(calculateArea, x, std::cref(lut), opacity, std::ref(commonData), threadWidth);
	}
	// Launch the last thread with a slightly larger width
	threads.emplace_back(calculateArea, x, std::cref(lut), opacity, std::ref(commonData), threadWidth + remainder);
	for (auto& thread : threads)
	{
		thread.join();
	}
	// Return the modified result
	return tmp;
}
