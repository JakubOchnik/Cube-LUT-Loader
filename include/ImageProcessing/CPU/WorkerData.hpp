#pragma once

struct WorkerData {
	unsigned char* image;
	unsigned char* newImage;
	const int width;
	const int height;
	const int channels;
	const float lutScale; // Used for 3D lut: (LUT_SIZE - 1) / 255 -> as arrays are indexed from 0, size needs to be reduced by 1
	const float opacity;
	const int nValues; // Used only for 1D lut
};
