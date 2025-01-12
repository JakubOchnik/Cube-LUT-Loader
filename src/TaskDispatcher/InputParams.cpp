#include "TaskDispatcher/InputParams.h"
#include <string>

InputParams::InputParams(ProcessingMode processMode, unsigned int threadsNum, InterpolationMethod interpolation,
						 const std::string& inputPath, const std::string& outputPath, bool force,
						 const std::string& lut, float intensity, int width, int height)
	: processingMode(processMode), threads(threadsNum), interpolationMethod(interpolation), inputImgPath(inputPath),
	  outputImgPath(outputPath), forceOverwrite(force), inputLutPath(lut), outputImageWidth(width),
	  outputImageHeight(height) {
	effectIntensity = intensity / 100.0f;
}

ProcessingMode InputParams::getProcessingMode() const {
	return processingMode;
}

bool InputParams::getForceOverwrite() const {
	return forceOverwrite;
}

void InputParams::setForceOverwrite(bool value) {
	forceOverwrite = value;
}

std::string InputParams::getInputImgPath() const {
	return inputImgPath;
}

void InputParams::setInputImgPath(const std::string& inputPath) {
	inputImgPath = inputPath;
}

std::string InputParams::getOutputImgPath() const {
	return outputImgPath;
}

std::string InputParams::getInputLutPath() const {
	return inputLutPath;
}

void InputParams::setInputLutPath(const std::string& lutPath) {
	inputLutPath = lutPath;
}

float InputParams::getEffectIntensity() const {
	return effectIntensity;
}

unsigned int InputParams::getThreads() const {
	return threads;
}

InterpolationMethod InputParams::getInterpolationMethod() const {
	return interpolationMethod;
}

int InputParams::getOutputImageWidth() const {
	return outputImageWidth;
}

void InputParams::setOutputImageWidth(int width) {
	outputImageWidth = width;
}

int InputParams::getOutputImageHeight() const {
	return outputImageHeight;
}

void InputParams::setOutputImageHeight(int height) {
	outputImageHeight = height;
}
