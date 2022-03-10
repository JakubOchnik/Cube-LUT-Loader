#include <ImageProcess/LUT3D/applyNearestValue.hpp>

cv::Mat applyNearest(cv::Mat img, CubeLUT lut, const float opacity)
{
	// INIT
	cv::Mat tmp = img.clone();
	unsigned char *image = img.data;
	unsigned char *new_image = tmp.data;
	auto ch{img.channels()};
	// PROCESS
	// pixel = (x + y * COLS) * CHANNELS + channel_num
	for (int x{0}; x < tmp.cols; ++x)
	{
		for (int y{0}; y < tmp.rows; ++y)
		{
			int b = image[(x + y * tmp.cols) * ch + 0]; // b
			unsigned int b_ind = round(b * (lut.LUT3D.size() - 1) / 255.0f);
			int g = image[(x + y * tmp.cols) * ch + 1]; // g
			unsigned int g_ind = round(g * (lut.LUT3D.size() - 1) / 255.0f);
			int r = image[(x + y * tmp.cols) * ch + 2]; // r
			unsigned int r_ind = round(r * (lut.LUT3D.size() - 1) / 255.0f);

			int newB = (int)(lut.LUT3D[r_ind][g_ind][b_ind][2] * 255);
			int newG = (int)(lut.LUT3D[r_ind][g_ind][b_ind][1] * 255);
			int newR = (int)(lut.LUT3D[r_ind][g_ind][b_ind][0] * 255);

			unsigned char finalB = b + (newB - b) * opacity;
			unsigned char finalG = g + (newG - g) * opacity;
			unsigned char finalR = r + (newR - r) * opacity;

			new_image[(x + y * tmp.cols) * ch + 0] = finalB;
			new_image[(x + y * tmp.cols) * ch + 1] = finalG;
			new_image[(x + y * tmp.cols) * ch + 2] = finalR;
		}
	}
	return tmp;
}