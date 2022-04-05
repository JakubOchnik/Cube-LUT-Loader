# Cube LUT Loader
A simple C++ tool that lets you apply a Cube format LUT to your image.

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
```

## Used libraries
- OpenCV 4.5.5
- Boost 1.78.0
- Eigen 3.4.0 (Tensor module)
- CUDA 11.6

## Features
- 1D LUT support
- 3D LUT support
    - Nearest value mode (faster, low quality of tonal transitions)
    - Trilinear interpolation (slower, higher quality)
- Simple input arguments-based interface
- Multithreaded implementation (the number of physical CPU threads is used by default)
- CUDA GPU implementation (highly effective with larger images)

## Performance
### Trilinear interpolation
![Trilinear interpolation graph](docs/performance_comparison/img/tri_interp.png "Trilinear interpolation graph").  

The trilinear interpolation method provides excellent results at the cost of high mathematical complexity.
This is a perfect scenario for a GPU, as it fully utilizes its potential to accelerate heavy compute operations combined with the parallel nature of the image matrix.
The memory copying/allocation costs are fully compensated by the performance gains up until 1920x1080 resolution. For very small images, multithreaded implementation
seems to be the best fit.  
### Nearest value interpolation
![Nearest-value interpolation graph](docs/performance_comparison/img/nv_interp.png "Nearest-value interpolation graph").  

The results are different for the nearest-value interpolation method. Its computational complexity is not high enough to compensate for the GPU memory I/O latency, so the performance of CUDA kernels is generally worse than multithreaded implementation. However, GPU may start to provide some benefit over the 8-threaded CPU for ultra-high-resolution images (over 50 MP).

## In progress
- GPU acceleration for 1D LUTs
- Comprehensive performance comparison between all implementations
- Support of color depths other than 8-bit
- Batch processing

### Legal disclaimer
The .cube LUT parser is used under Creative Commons Attribuition Non-Commercial 3.0 License.
It was created by Adobe Inc.  
Source: Cube LUT Specification 1.0  
https://wwwimages2.adobe.com/content/dam/acom/en/products/speedgrade/cc/pdfs/cube-lut-specification-1.0.pdf
