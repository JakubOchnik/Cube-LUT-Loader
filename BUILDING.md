# Build instructions
## Linux
Tested on Ubuntu 22.04.
1. Install all of the required dependencies
```bash
sudo apt install build-essential cmake g++ # C++ build utilities
sudo apt install libopencv-dev # OpenCV libraries
sudo apt install libboost-all-dev libeigen3-dev # Boost & Eigen3 libraries
# Optionally install CUDA Toolkit (required for building CUDA code)
```
2. Toggle CUDA in `CMakeLists.txt`
```
set(BUILD_CUDA OFF) / set(BUILD_CUDA ON)
```
3. Build the project using the following commands:
```
cmake . -DCMAKE_BUILD_TYPE:STRING=Debug -B build/
cmake --build build/ --config Debug --target lut_loader -j
```
*Note that you may adjust the build directory as well as the build type (Release/Debug).*
## Windows
Work in progress!
