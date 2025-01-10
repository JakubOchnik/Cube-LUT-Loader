# Cube LUT Loader
<center><img src="docs/example_pic.jpg" alt="drawing" width="600"/></center>

![Workflow](https://github.com/JakubOchnik/Cube-LUT-Loader/actions/workflows/build-and-test.yml/badge.svg)  
A simple command-line C++ tool that allows you to apply a Cube format LUT to an image.
It offers high-performance multi-threaded and GPU-accelerated modes, various interpolation methods, and support for 1D and 3D LUTs.  
See [performance tests](PERFORMANCE.md) for a performance comparison between the offered modes.

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
      -h, --help                        Display this help menu
      -i [input], --input=[input]       Input file path
      -l [lut], --lut=[lut]             LUT file path
      -o [output], --output=[output]    Output file path
      -s [intensity],
      --strength=[intensity]            Intensity of the applied LUT (0-100)
      -m [method], --method=[method]    Interpolation method (allowed values:
                                        'trilinear', 'nearest-value')
      -f, --force                       Force overwrite file
      -j [threads], --threads=[threads] Number of threads
      -p [processor],
      --processor=[processor]           Processing mode (allowed values: 'cpu',
                                        'gpu')
      --width=[width]                   Output image width
      --height=[height]                 Output image height
```

## Hardware requirements
A suitable NVIDIA GPU is required for the GPU mode to work (preferably with compute capability 6.1 or up).
However, if you don't have one, you can still build and use the program in CPU mode!

## In progress
- Tetrahedral interpolation
- GPU acceleration for 1D LUTs
- Support of color depths other than 8-bit
- Batch processing
