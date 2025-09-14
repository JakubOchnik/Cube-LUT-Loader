#include <ImageProcessing/Metal/utils/MetalUtils.hpp>
#include <iostream>
#include <sstream>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

namespace {
[[nodiscard]] std::unordered_map<std::string, std::string> getMetalDeviceInfo() {
	const auto device = MTL::CreateSystemDefaultDevice();
	std::unordered_map<std::string, std::string> result;
	result["name"] = device->name()->cString(NS::StringEncoding::ASCIIStringEncoding);
	result["architecture"] = device->architecture()->name()->cString(NS::StringEncoding::ASCIIStringEncoding);
	result["locationNumber"] = std::to_string(device->locationNumber());

	result["recommendedMaxWorkingSetSize"] = std::to_string(device->recommendedMaxWorkingSetSize());
	result["hasUnifiedMemory"] = std::to_string(device->hasUnifiedMemory());

	result["maxThreadgroupMemoryLength"] = std::to_string(device->maxThreadgroupMemoryLength());
	const auto gridSize = device->maxThreadsPerThreadgroup();
	result["maxThreadsPerThreadgroup"] = std::to_string(gridSize.width) + " x " + std::to_string(gridSize.height) +
										 " x " + std::to_string(gridSize.depth);

	return result;
}
} // namespace

namespace util::metal {
[[nodiscard]] bool isMetalDeviceAvailable() { 
    const bool metalDeviceAvailable = MTL::CreateSystemDefaultDevice() != nullptr;
    if (!metalDeviceAvailable) {
        std::cerr << "[ERROR] No Metal devices available\n";
    }
    return metalDeviceAvailable;
}

[[nodiscard]] std::string getReadableMetalDeviceInfo() {
	const auto info = getMetalDeviceInfo();
	std::stringstream ss;
	for (const auto& [key, val] : info) {
		ss << key << ": " << val << '\n';
	}
	return ss.str();
}
} // namespace util::metal