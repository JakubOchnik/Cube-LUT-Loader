#include <ImageProcess/LUT3D/applyTrilinear.hpp>

vector<float> mul(const vector<float>& vec, const float val)
{
	vector<float> newVec(3, 0.0f);
	for (int i{ 0 }; i < 3; ++i)
		newVec[i] = vec[i] * val;
	return newVec;
}

vector<float> sum(const vector<float>& a, const vector<float>& b)
{
	vector<float> newVec(3, 0.0f);
	for (int i{ 0 }; i < 3; ++i)
		newVec[i] = a[i] + b[i];
	return newVec;
}

cv::Mat applyTrilinear(cv::Mat img, CubeLUT lut, const float opacity)
{
	// INIT
	cv::Mat tmp = img.clone();
	unsigned char* image = img.data;
	unsigned char* new_image = tmp.data;

	auto ch{img.channels()};
	// PROCESS
	// pixel = (x + y * COLS) * CHANNELS + channel_num
	for (int x{ 0 }; x < tmp.cols; ++x)
	{
		for (int y{ 0 }; y < tmp.rows; ++y)
		{
			int b = image[(x + y * tmp.cols) * ch + 0]; //b
			int g = image[(x + y * tmp.cols) * ch + 1]; //g
			int r = image[(x + y * tmp.cols) * ch + 2]; //r


			int R1 = ceil(r / 255.0f * (float)(lut.LUT3D.size() - 1));
			int R0 = floor(r / 255.0f * (float)(lut.LUT3D.size() - 1));
			int G1 = ceil(g / 255.0f * (float)(lut.LUT3D.size() - 1));
			int G0 = floor(g / 255.0f * (float)(lut.LUT3D.size() - 1));
			int B1 = ceil(b / 255.0f * (float)(lut.LUT3D.size() - 1));
			int B0 = floor(b / 255.0f * (float)(lut.LUT3D.size() - 1));

			float r_o = r * (lut.LUT3D.size() - 1) / 255.0f;
			float g_o = g * (lut.LUT3D.size() - 1) / 255.0f;
			float b_o = b * (lut.LUT3D.size() - 1) / 255.0f;

			float delta_r{ r_o - R0 == 0 || R1 - R0 == 0 ? 0 : (r_o - R0) / (float)(R1 - R0) };
			float delta_g{ g_o - G0 == 0 || G1 - G0 == 0 ? 0 : (g_o - G0) / (float)(G1 - G0) };
			float delta_b{ b_o - B0 == 0 || B1 - B0 == 0 ? 0 : (b_o - B0) / (float)(B1 - B0) };

			vector<float> vr_gz_bz = sum(mul(lut.LUT3D[R0][G0][B0], 1 - delta_r), mul(lut.LUT3D[R0][G0][B0], delta_r));
			vector<float> vr_gz_bo = sum(mul(lut.LUT3D[R0][G0][B1], 1 - delta_r), mul(lut.LUT3D[R0][G0][B1], delta_r));
			vector<float> vr_go_bz = sum(mul(lut.LUT3D[R0][G1][B0], 1 - delta_r), mul(lut.LUT3D[R0][G1][B0], delta_r));
			vector<float> vr_go_bo = sum(mul(lut.LUT3D[R0][G1][B1], 1 - delta_r), mul(lut.LUT3D[R0][G1][B1], delta_r));

			vector<float> vrg_b0 = sum(mul(vr_gz_bz, 1 - delta_g), mul(vr_go_bz, delta_g));
			vector<float> vrg_b1 = sum(mul(vr_gz_bo, 1 - delta_g), mul(vr_go_bo, delta_g));

			vector<float> vrgb = sum(mul(vrg_b0, 1 - delta_b), mul(vrg_b1, delta_b));


			unsigned char newB = round(vrgb[2] * 255);
			unsigned char newG = round(vrgb[1] * 255);
			unsigned char newR = round(vrgb[0] * 255);

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