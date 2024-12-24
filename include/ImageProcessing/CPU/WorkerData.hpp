#pragma once

struct WorkerData {
	unsigned char* image;
	unsigned char* newImage;
	const int width;
	const int height;
	const int channels;
	const int lutSize;
	const float opacity;
	const int nValues; // Used only for 1D lut
};
