#include "CubeLUT.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <valarray>

using namespace cv;

int main(int argc, char* const argv[])
{

	CubeLUT theCube;
	enum { OK = 0, ErrorOpenInFile = 100, ErrorOpenOutFile };

	if (argc < 2 || 3 < argc)
	{
		cout << "Usage: " << argv[0] << " cubeFileIn" << endl;
		return OK;
	}

	ifstream infile(argv[1]);
	if (!infile.good())
	{
		cout << "Could not open input file " << argv[1] << endl;
		return ErrorOpenInFile;
	}
	int ret = theCube.LoadCubeFile(infile);
	infile.close();
	/*if (ret != OK)
	{
		cout << "Could not parse the cube info in the input file. Return code = " << ret << endl;
		return theCube.status;
	}*/

	// theCube.LUT3D
	//Mat_<Vec3b> img = imread("Test_Image.png");
	Mat_<Vec3b> img = imread("Test_Image.png");
	cv::imshow("oryg", img);
	//cv:waitKey(0);
	//cv::destroyWindow("test");
	Mat_<Vec3b> img1 = img.clone();
	float opacity = 0.9f;
	for (auto& pixel : img1) {
		int b = pixel[0]; //b
		unsigned int b_ind = b * (theCube.LUT3D.size() - 1) / 255;
		int g = pixel[1]; //g
		unsigned int g_ind = g * (theCube.LUT3D.size() - 1) / 255;
		int r = pixel[2]; //r
		unsigned int r_ind = r * (theCube.LUT3D.size() - 1) / 255;

		if (r == 255 && g == 255 && b == 255)
		{
			printf("");
		}

		int newB = (int)(theCube.LUT3D[r_ind][g_ind][b_ind][2] * 255);
		int newG = (int)(theCube.LUT3D[r_ind][g_ind][b_ind][1] * 255);
		int newR = (int)(theCube.LUT3D[r_ind][g_ind][b_ind][0] * 255);

		unsigned char finalB = b + (newB - b) * opacity;
		unsigned char finalG = g + (newG - g) * opacity;
		unsigned char finalR = r + (newR - r) * opacity;

		pixel[0] = finalB;
		pixel[1] = finalG;
		pixel[2] = finalR;
	}


	imshow("test", img1);

	Mat_<Vec3b> img2 = img.clone();

	opacity = 0.1f;
	for (auto& pixel : img2) {
		int b = pixel[0]; //b
		unsigned int b_ind = b * (theCube.LUT3D.size() - 1) / 255;
		int g = pixel[1]; //g
		unsigned int g_ind = g * (theCube.LUT3D.size() - 1) / 255;
		int r = pixel[2]; //r
		unsigned int r_ind = r * (theCube.LUT3D.size() - 1) / 255;

		int newB = (int)(theCube.LUT3D[r_ind][g_ind][b_ind][2] * 255);
		int newG = (int)(theCube.LUT3D[r_ind][g_ind][b_ind][1] * 255);
		int newR = (int)(theCube.LUT3D[r_ind][g_ind][b_ind][0] * 255);

		int finalB = b + (newB - b) * opacity;
		int finalG = g + (newG - g) * opacity;
		int finalR = r + (newR - r) * opacity;

		pixel[0] = finalB;
		pixel[1] = finalG;
		pixel[2] = finalR;
	}


	imshow("test1", img2);

	Mat_<Vec3b> img3 = img.clone();

	opacity = 1.0f;
	for (auto& pixel : img3) {
		/*int b = pixel[0]; //b
		unsigned int b_ind = b * (theCube.LUT3D.size() - 1) / 255;
		int g = pixel[1]; //g
		unsigned int g_ind = g * (theCube.LUT3D.size() - 1) / 255;
		int r = pixel[2]; //r
		unsigned int r_ind = r * (theCube.LUT3D.size() - 1) / 255;

		int newB = (int)(theCube.LUT3D[r_ind][g_ind][b_ind][2] * 255);
		int newG = (int)(theCube.LUT3D[r_ind][g_ind][b_ind][1] * 255);
		int newR = (int)(theCube.LUT3D[r_ind][g_ind][b_ind][0] * 255);

		int finalB = b + (newB - b) * opacity;
		int finalG = g + (newG - g) * opacity;
		int finalR = r + (newR - r) * opacity;

		pixel[0] = finalB;
		pixel[1] = finalG;
		pixel[2] = finalR;*/


		int b = pixel[0];
		int g = pixel[1];
		int r = pixel[2];

		// indeksy: 0-32
		// barwy: 0-255
		// barwy w LUT: 0-1
		// index = channel/255*32
		// barwa - interpolacja
		// jezeli nie ma dokladnie indeksu odpowiadajacego barwie, to musi byc interpolacja

		int R1 = ceil(r / 255.0f * (float)(theCube.LUT3D.size() - 1));
		int R0 = floor(r / 255.0f * (float)(theCube.LUT3D.size() - 1));
		int G1 = ceil(g / 255.0f * (float)(theCube.LUT3D.size() - 1));
		int G0 = floor(g / 255.0f * (float)(theCube.LUT3D.size() - 1));
		int B1 = ceil(b / 255.0f * (float)(theCube.LUT3D.size() - 1));
		int B0 = floor(b / 255.0f * (float)(theCube.LUT3D.size() - 1));
		float r_o = r * (theCube.LUT3D.size() - 1) / 255.0f;
		float g_o = g * (theCube.LUT3D.size() - 1) / 255.0f;
		float b_o = b * (theCube.LUT3D.size() - 1) / 255.0f;

		float delta_r = (r_o - R0) / (float)(R1 - R0);
		float delta_g = (g_o - G0) / (float)(G1 - G0);
		float delta_b = (b_o - B0) / (float)(B1 - B0);

		auto mul = [](const vector<float>& vec, float val) {vector<float> newVec(3, 0.0f);  for (int i{ 0 }; i < 3; ++i) newVec[i] = vec[i] * val; return newVec; };
		auto sum = [](const vector<float>& a, const vector<float>& b) {vector<float> newVec(3, 0.0f); for (int i{ 0 }; i < 3; ++i) newVec[i] = a[i] + b[i]; return newVec; };
		vector<float> vr_gz_bz = sum(mul(theCube.LUT3D[R0][G0][B0], 1 - delta_r), mul(theCube.LUT3D[R0][G0][B0], delta_r));
		vector<float> vr_gz_bo = sum(mul(theCube.LUT3D[R0][G0][B1], 1 - delta_r), mul(theCube.LUT3D[R0][G0][B1], delta_r));
		vector<float> vr_go_bz = sum(mul(theCube.LUT3D[R0][G1][B0], 1 - delta_r), mul(theCube.LUT3D[R0][G1][B0], delta_r));
		vector<float> vr_go_bo = sum(mul(theCube.LUT3D[R0][G1][B1], 1 - delta_r), mul(theCube.LUT3D[R0][G1][B1], delta_r));

		vector<float> vrg_b0 = sum(mul(vr_gz_bz, 1 - delta_g), mul(vr_go_bz, delta_g));
		vector<float> vrg_b1 = sum(mul(vr_gz_bo, 1 - delta_g), mul(vr_go_bo, delta_g));

		vector<float> vrgb = sum(mul(vrg_b0, 1 - delta_b), mul(vrg_b1, delta_b));

		pixel[0] = round(vrgb[2] * 255); //b
		pixel[1] = round(vrgb[1] * 255); //g
		pixel[2] = round(vrgb[0] * 255); //r

	}


	imshow("test2", img3);
	waitKey(0);

	destroyWindow("oryg");
	destroyWindow("test");
	destroyWindow("test1");
	destroyWindow("test2");
	return 0;
}
