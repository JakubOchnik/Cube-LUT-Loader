# Building
## Used libraries
- OpenCV 4.5.5: `opencv_core`, `opencv_imgcodecs`
- Boost 1.78.0: `program_options`
- Eigen 3.4.0: Vectors and Tensor module, header-only
- CUDA 11.6: used for GPU mode
- fmt 10.2.1

Newer versions of the above libraries should work too.

## Build instructions
### Conan (preferred)
0. Install conan and detect profile
```
pip install conan
conan profile detect
```
1. Install / build dependencies

Debug:
```
conan install . -s build_type=Release -s "&:build_type=Debug" --build=missing
```
Release:
```
conan install . -s build_type=Release -s "&:build_type=Release" --build=missing
```
Additional notes:
- The first `-s` applies to the dependencies and the second one to the application itself (consumer).
- External libraries will be built from sources if they don't match the compiler version specified in the remote recipe (in [Conan Center](https://conan.io/center)). You can override it by fetching dependencies for a version matching the remote recipe, using the same syntax as the build type (e.g. `-s compiler.version=13 -s &:compiler.version=14`).
- If you encounter issues on Ubuntu, you might need to add `-c tools.system.package_manager:mode=install -c tools.system.package_manager:sudo=True` to enable properly installing packages from apt.

2. Configure the project

Windows:
```
cmake --preset conan-default
cmake --build --preset conan-release # or conan-debug
```
Linux:
```
conan install . --build=missing # -s build_type=Debug
cmake --preset conan-release # or conan-debug
cmake --build --preset conan-release # or conan-debug
```
### Manual
1. Install all of the required dependencies
```
# For OpenCV, only opencv_core and opencv_imgcodecs are required.
# Optionally install CUDA Toolkit (required for building CUDA code)

# Linux (Ubuntu 22.04 used as an example):
sudo apt install libopencv-dev # OpenCV libraries
sudo apt install libboost-all-dev libeigen3-dev libfmt-dev # Boost, Eigen3, fmt libraries
sudo apt install build-essential cmake g++ # C++ build utilities

# Windows:
# Manually install the libraries. If applicable, install them at their default directories or configure proper environment variables to make sure CMake automatically finds them.
```
2. Configure `BUILD_CUDA` and `BUILD_TESTS` variables in root-level `CMakeLists.txt` or later as `-D` CMake command line arguments.
3. Build the project in your IDE or using the following commands:
```
cmake . -B build/
cmake --build build/ --target lut_loader -j
```

## Tests
To build tests, enable `BUILD_TESTS` variable and build target `lut_loader_test`. Make sure to run the test executable from the proper working directory (`REPO_ROOT/src/test`).

## Performance note
Make sure to build the final executable in `release` preset (with compiler optimizations), since it may provide a major performance boost.
