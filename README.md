# Cube LUT Loader
<center><img src="docs/example_pic.jpg" alt="drawing" width="600"/></center>

![Workflow](https://github.com/JakubOchnik/Cube-LUT-Loader/actions/workflows/build-and-test.yml/badge.svg)  
A simple command-line C++ tool that allows you to apply a Cube format LUT to an image.
It offers high-performance multi-threaded and GPU-accelerated modes, various interpolation methods, and support for 1D and 3D LUTs.  
To see the performance comparison between the offered modes, see [performance tests](PERFORMANCE.md).

## Features
- Simple CLI
- 1D LUT support
- 3D LUT support
    - Nearest value mode (faster, low quality of tonal transitions)
    - Trilinear interpolation (slower, high quality)
- Multi-threaded implementation (the number of physical CPU threads is used by default)
- CUDA GPU implementation (highly effective with larger images)

## Program options
```
  -h [ --help ]                  Help screen
  -i [ --input ] arg             Input file path
  -l [ --lut ] arg               LUT file path
  -o [ --output ] arg (=out.png) Output file path [= out.png]
  -s [ --strength ] arg (=1)     Strength of the effect [= 1.0]
  -t [ --trilinear ]             Trilinear interpolation of 3D LUT
  -n [ --nearest_value ]         No interpolation of 3D LUT
  -j [ --threads ] arg (=8)      Number of threads [= Number of physical threads]
  --gpu                          Use GPU acceleration
  --width arg                    Output image width
  --height arg                   Output image height
```

## Hardware requirements
A suitable NVIDIA GPU is required for the GPU mode to work (preferably with compute capability 6.1 or up).
However, if you don't have one, you can still build and use the program in CPU mode!

## In progress
- Tetrahedral interpolation
- GPU acceleration for 1D LUTs
- Support of color depths other than 8-bit
- Batch processing
- Increasing unit test coverage
