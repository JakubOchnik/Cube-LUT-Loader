#include "applyNearestValue.hpp"

cv::Mat_<cv::Vec3b> applyNearest(cv::Mat img, CubeLUT lut, const float opacity)
{
	cv::Mat_<cv::Vec3b> tmp = img.clone();
	for (auto& pixel : tmp) {
		int b = pixel[0]; //b
		unsigned int b_ind = round(b * (lut.LUT3D.size() - 1) / 255.0f);
		int g = pixel[1]; //g
		unsigned int g_ind = round(g * (lut.LUT3D.size() - 1) / 255.0f);
		int r = pixel[2]; //r
		unsigned int r_ind = round(r * (lut.LUT3D.size() - 1) / 255.0f);

		int newB = (int)(lut.LUT3D[r_ind][g_ind][b_ind][2] * 255);
		int newG = (int)(lut.LUT3D[r_ind][g_ind][b_ind][1] * 255);
		int newR = (int)(lut.LUT3D[r_ind][g_ind][b_ind][0] * 255);

		unsigned char finalB = b + (newB - b) * opacity;
		unsigned char finalG = g + (newG - g) * opacity;
		unsigned char finalR = r + (newR - r) * opacity;

		if (finalR == 11)
		{
			printf("");
		}

		pixel[0] = finalB;
		pixel[1] = finalG;
		pixel[2] = finalR;
	}

	return tmp;
}