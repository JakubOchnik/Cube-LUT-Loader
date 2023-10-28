#pragma once
#include <optional>
#include <string>
#include <boost/program_options.hpp>

enum class InterpolationMethod {
	Trilinear,
	NearestValue
};

enum class ProcessingMode {
	CPU,
	GPU
};

class InputParams {
private:
	ProcessingMode processingMode{};
	bool showHelp{};
	std::string inputImgPath;
	std::string outputImgPath;
	std::string inputLutPath;
	float effectStrength{ 1.0f };
	unsigned int threads{ 1 };
	InterpolationMethod interpolationMethod{};

	template<typename T>
	void setParam(const std::string& key, T& field, const boost::program_options::variables_map& vm) {
		if (vm.count(key)) {
			field = vm[key].as<T>();
		}
	}

public:
	explicit InputParams(boost::program_options::variables_map&& vm);
	InputParams() = default;
	void parseInputParams(boost::program_options::variables_map&& vm);

	ProcessingMode getProcessingMode() const;
	bool getShowHelp() const;
	std::string getInputImgPath() const;
	std::string getOutputImgPath() const;
	std::string getInputLutPath() const;
	float getEffectStrength() const;
	unsigned int getThreads() const;
	InterpolationMethod getInterpolationMethod() const;
};
