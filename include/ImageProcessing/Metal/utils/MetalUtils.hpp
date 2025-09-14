#pragma once
#include <string>
#include <unordered_map>

namespace util::metal {
[[nodiscard]] bool isMetalDeviceAvailable();
[[nodiscard]] std::string getReadableMetalDeviceInfo();
}