#include "Loader/CubeLUT.hpp"
#include "3Dprocess/applyNearestValue.hpp"
#include "3Dprocess/applyTrilinear.hpp"
#include "1Dprocess/apply1D.hpp"

#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <opencv2/opencv.hpp>


const std::string LUTpath = "Clean & Tidy_1.AF2I2724.cube";
const std::string imgPath = "Test_Image.png";
const float opacity = 1.0f;

int main(int argc, char* const argv[])
{
	using namespace cv;

	CubeLUT theCube;
	enum { OK = 0, ErrorOpenInFile = 100, ErrorOpenOutFile };

	ifstream infile(LUTpath);
	if (!infile.good())
	{
		cout << "Could not open input file " << LUTpath << endl;
		return ErrorOpenInFile;
	}
	int ret = theCube.LoadCubeFile(infile);
	infile.close();
	if (ret != OK)
	{
		cout << "Could not parse the cube info in the input file. Return code = " << ret << endl;
		return theCube.status;
	}

	Mat_<Vec3b> img = imread(imgPath);
	Mat_<Vec3b> newImg;
	if (theCube.LUT1D.empty())
	{
		// 3D
		newImg = applyTrilinear(img, theCube, opacity);
		// Mat_<Vec3b> newImg = applyNearestValue(img, theCube, opacity);
	}
	else
	{
		// 1D
		newImg = applyBasic1D(img, theCube, opacity);
	}


	imshow("Original image", img);
	imshow("Image with LUT applied", newImg);

	waitKey(0);
	destroyWindow("Original image");
	destroyWindow("Image with LUT applied");
	return 0;
}
