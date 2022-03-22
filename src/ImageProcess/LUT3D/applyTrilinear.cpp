#include <ImageProcess/LUT3D/applyTrilinear.hpp>

void Trilinear::calculatePixel(const int x, const int y, const CubeLUT& lut, const float opacity, WorkerData& data)
{
	const int b = data.image[(x + y * data.width) * data.channels + 0]; //b
	const int g = data.image[(x + y * data.width) * data.channels + 1]; //g
	const int r = data.image[(x + y * data.width) * data.channels + 2]; //r

	// Implementation of a formula in the "Method" section: https://en.wikipedia.org/wiki/Trilinear_interpolation

	const int R1 = static_cast<int>(ceil(r / 255.0f * (float)(data.lutSize - 1)));
	const int R0 = static_cast<int>(floor(r / 255.0f * (float)(data.lutSize - 1)));
	const int G1 = static_cast<int>(ceil(g / 255.0f * (float)(data.lutSize - 1)));
	const int G0 = static_cast<int>(floor(g / 255.0f * (float)(data.lutSize - 1)));
	const int B1 = static_cast<int>(ceil(b / 255.0f * (float)(data.lutSize - 1)));
	const int B0 = static_cast<int>(floor(b / 255.0f * (float)(data.lutSize - 1)));

	const float r_o = r * (data.lutSize - 1) / 255.0f;
	const float g_o = g * (data.lutSize - 1) / 255.0f;
	const float b_o = b * (data.lutSize - 1) / 255.0f;

	const float delta_r{ r_o - R0 == 0 || R1 - R0 == 0 ? 0 : (r_o - R0) / (float)(R1 - R0) };
	const float delta_g{ g_o - G0 == 0 || G1 - G0 == 0 ? 0 : (g_o - G0) / (float)(G1 - G0) };
	const float delta_b{ b_o - B0 == 0 || B1 - B0 == 0 ? 0 : (b_o - B0) / (float)(B1 - B0) };

	using namespace Eigen;
	using Arr4 = Eigen::array<Eigen::Index, 4>;
	using Vec3fWrap = Tensor<float, 1>;

	Vec3fWrap v1 = (lut.LUT3D.slice(Arr4{R0, G0, B0, 0}, data.extents) * (1 - delta_r) + \
					lut.LUT3D.slice(Arr4{R1, G0, B0, 0}, data.extents) * delta_r).reshape(Eigen::array<Eigen::Index,1>{3});
	Vec3fWrap v2 = (lut.LUT3D.slice(Arr4{R0, G0, B1, 0}, data.extents) * (1 - delta_r) + \
					lut.LUT3D.slice(Arr4{R1, G0, B1, 0}, data.extents) * delta_r).reshape(Eigen::array<Eigen::Index,1>{3});
	Vec3fWrap v3 = (lut.LUT3D.slice(Arr4{R0, G1, B0, 0}, data.extents) * (1 - delta_r) + \
					lut.LUT3D.slice(Arr4{R1, G1, B0, 0}, data.extents) * delta_r).reshape(Eigen::array<Eigen::Index,1>{3});
	Vec3fWrap v4 = (lut.LUT3D.slice(Arr4{R0, G1, B1, 0}, data.extents) * (1 - delta_r) + \
					lut.LUT3D.slice(Arr4{R1, G1, B1, 0}, data.extents) * delta_r).reshape(Eigen::array<Eigen::Index,1>{3});

	v1 = v1 * (1 - delta_g) + v3 * delta_g;
	v2 = v2 * (1 - delta_g) + v4 * delta_g;

	v1 = v1 * (1 - delta_b) + v2 * delta_b;

	unsigned char newB = static_cast<uchar>(round(v1(2) * 255));
	unsigned char newG = static_cast<uchar>(round(v1(1) * 255));
	unsigned char newR = static_cast<uchar>(round(v1(0) * 255));

	unsigned char finalB = static_cast<uchar>(b + (newB - b) * opacity);
	unsigned char finalG = static_cast<uchar>(g + (newG - g) * opacity);
	unsigned char finalR = static_cast<uchar>(r + (newR - r) * opacity);

	// Assign final pixel values to the output image
	data.new_image[(x + y * data.width) * data.channels + 0] = finalB;
	data.new_image[(x + y * data.width) * data.channels + 1] = finalG;
	data.new_image[(x + y * data.width) * data.channels + 2] = finalR;
}

void Trilinear::calculateArea(const int x, const CubeLUT& lut, const float opacity, WorkerData& data, const int segWidth)
{
	for(int localX{x}; localX < x + segWidth; ++localX)
	{
		for(int y{0}; y < data.height; ++y)
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
	unsigned char* new_image = tmp.data;
	WorkerData commonData{image, new_image, tmp.cols, tmp.rows, img.channels(), \
							static_cast<int>(lut.LUT3D.dimension(0)), {1,1,1,3}};

	// Processing
	// Divide the picture into threadPool vertical windows and process them simultaneously.
	// threadPool - 1 threads will process (WIDTH / threadPool) slices 
	// and the last one will process (WIDTH/threadPool + (WIDTH%threadPool))

	int threadWidth = tmp.cols / threadPool;
	int remainder = tmp.cols % threadPool;
	std::vector<std::thread> threads;
	threads.reserve(threadPool);

	// Launch threads
	int x{0}, tNum{0};
	for (; tNum < threadPool - 1; x += threadWidth, ++tNum)
	{
		threads.emplace_back(calculateArea, x, std::cref(lut), opacity, std::ref(commonData), threadWidth);
	}
	// Launch the last thread with a slightly larger width
	threads.emplace_back(calculateArea, x, std::cref(lut), opacity, std::ref(commonData), threadWidth + remainder);
	for(auto& thread: threads)
	{
		thread.join();
	}
	// Return the modified result
	return tmp;
}