# Cube LUT Loader
A simple C++ tool which lets you apply a Cube LUT to an image.

## Program options
```
  -h [ --help ]                  Help screen
  -i [ --input ] arg             Input file path
  -l [ --lut ] arg               LUT file path
  -o [ --output ] arg (=out.png) Output file path
  -s [ --strength ] arg (=1)     Strength of the effect
  -t [ --trilinear ]             Trilinear interpolation of 3D LUT
  -n [ --nearest_value ]         No interpolation of 3D LUT
```

## Dependencies
- OpenCV 4.5.1

## Working
- 1D LUT support
- 3D LUT support
    - Nearest value mode (faster, low quality of tonal transitions)
    - Trilinear interpolation (slower, higher quality)
- Simple command-line interface

## In progress
- Support of color depths other than 8-bit
- CUDA GPU implementation (could significantly speed up the process)

### Legal disclaimer
The .cube LUT parser is used under Creative Commons Attribuition Non-Commercial 3.0 License.
It was created by Adobe Inc.  
Source: Cube LUT Specification 1.0  
https://wwwimages2.adobe.com/content/dam/acom/en/products/speedgrade/cc/pdfs/cube-lut-specification-1.0.pdf