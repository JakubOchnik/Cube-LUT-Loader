#include "Loader/CubeLUT.hpp"
#include "3Dprocess/applyNearestValue.hpp"
#include "3Dprocess/applyTrilinear.hpp"

#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;

const std::string LUTpath = "M31 - Rec.709.cube";
const std::string imgPath = "Test_Image.png";

int main(int argc, char* const argv[])
{

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

	Mat_<Vec3b> imgNearest = applyNearest(img, theCube, 1.0f);

	Mat_<Vec3b> imgTrilinear = applyTrilinear(img, theCube, 1.0f);

	imshow("Original image", img);
	imshow("Nearest value (no interpolation)", imgNearest);
	imshow("Trilinear interpolation", imgTrilinear);


	waitKey(0);
	destroyWindow("Original image");
	destroyWindow("Nearest value (no interpolation)");
	destroyWindow("Trilinear interpolation");
	return 0;
}
