#include "TaskDispatcher/InputParams.h"

InputParams::InputParams(boost::program_options::variables_map&& vm) {
	parseInputParams(std::move(vm));
}

void InputParams::parseInputParams(boost::program_options::variables_map&& vm) {
	if (vm.count("gpu")) {
		processingMode = ProcessingMode::GPU;
	}

	if (vm.count("trilinear")) {
		interpolationMethod = InterpolationMethod::Trilinear;
	}
	else if (vm.count("nearest_value")) {
		interpolationMethod = InterpolationMethod::NearestValue;
	}

	setParam("help", showHelp, vm);
	setParam("input", inputImgPath, vm);
	setParam("output", outputImgPath, vm);
	setParam("lut", inputLutPath, vm);
	setParam("strength", effectStrength, vm);
	setParam("threads", threads, vm);
}

ProcessingMode InputParams::getProcessingMode() const {
	return processingMode;
}

bool InputParams::getShowHelp() const {
	return showHelp;
}

std::string InputParams::getInputImgPath() const {
	return inputImgPath;
}

std::string InputParams::getOutputImgPath() const {
	return outputImgPath;
}

std::string InputParams::getInputLutPath() const {
	return inputLutPath;
}

float InputParams::getEffectStrength() const {
	return effectStrength;
}

unsigned int InputParams::getThreads() const {
	return threads;
}

InterpolationMethod InputParams::getInterpolationMethod() const {
	return interpolationMethod;
}
