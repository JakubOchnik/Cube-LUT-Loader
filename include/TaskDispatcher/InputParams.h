#pragma once
#include <string>
#include <fmt/format.h>

enum class InterpolationMethod {
	Trilinear,
	Tetrahedral,
	NearestValue
};

enum class ProcessingMode {
	CPU,
	GPU
};

class InputParams {
private:
	ProcessingMode processingMode{};
	bool forceOverwrite{};
	std::string inputImgPath;
	std::string outputImgPath;
	std::string inputLutPath;
	float effectIntensity{ 1.0f };
	unsigned int threads{ 1 };
	InterpolationMethod interpolationMethod{};
	int outputImageWidth{};
	int outputImageHeight{};

public:
	InputParams() = default;
	InputParams(
		ProcessingMode processMode,
		unsigned int threadsNum,
		InterpolationMethod interpolation,
		const std::string& inputPath,
		const std::string& outputPath,
		bool force,
		const std::string& lut,
		float intensity,
		int width,
		int height);

	ProcessingMode getProcessingMode() const;
	bool getForceOverwrite() const;
	void setForceOverwrite(bool value);
	std::string getInputImgPath() const;
	void setInputImgPath(const std::string& inputPath);
	std::string getOutputImgPath() const;
	std::string getInputLutPath() const;
	void setInputLutPath(const std::string& lutPath);
	float getEffectIntensity() const;
	unsigned int getThreads() const;
	InterpolationMethod getInterpolationMethod() const;
	int getOutputImageWidth() const;
	void setOutputImageWidth(int width);
	int getOutputImageHeight() const;
	void setOutputImageHeight(int height);
};
