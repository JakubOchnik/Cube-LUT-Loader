#pragma once
#include <optional>
#include <string>
#include <boost/program_options.hpp>
#include <boost/format.hpp>

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
	unsigned int outputImageWidth{};
	unsigned int outputImageHeight{};

	template<typename T>
	void setParam(const std::string& key, T& field, const boost::program_options::variables_map& vm) {
		if (vm.count(key)) {
			field = vm[key].as<T>();
		}
	}

	template<typename SourceType, typename DestinationType>
	void setParam(const std::string& key, DestinationType& field, const boost::program_options::variables_map& vm, const std::function<bool(SourceType)>& verificationFunction) {
		if (vm.count(key)) {
			const auto value = vm[key].as<SourceType>();
			if (!verificationFunction(value)) {
				const auto message = boost::format("Incorrect value for %1% (%2%)") % key % value;
				throw std::runtime_error(message.str());
			}
			field = static_cast<DestinationType>(value);
		}
	}

public:
	explicit InputParams(boost::program_options::variables_map&& vm);
	InputParams() = default;
	void parseInputParams(boost::program_options::variables_map&& vm);

	ProcessingMode getProcessingMode() const;
	bool getShowHelp() const;
	void setShowHelp(bool value);
	std::string getInputImgPath() const;
	void setInputImgPath(const std::string& inputPath);
	std::string getOutputImgPath() const;
	std::string getInputLutPath() const;
	float getEffectStrength() const;
	unsigned int getThreads() const;
	InterpolationMethod getInterpolationMethod() const;
	unsigned int getOutputImageWidth() const;
	void setOutputImageWidth(unsigned int width);
	unsigned int getOutputImageHeight() const;
	void setOutputImageHeight(unsigned int height);
};
