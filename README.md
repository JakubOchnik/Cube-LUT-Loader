# Cube LUT Loader
A simple C++ tool which lets you apply a Cube LUT to an image.

## Dependencies
- OpenCV 4.5.1

## Working
- 1D LUT support
- 3D LUT support
    - Nearest value mode (faster, low quality of tonal transitions)
    - Trilinear interpolation (slower, higher quality)

## In progress
- Simple command-line interface
- Support of color depths other than 8-bit
- CUDA GPU implementation (could significantly speed up the process)

### Legal disclaimer
The .cube LUT parser is used under Creative Commons Attribuition Non-Commercial 3.0 License.
It was created by Adobe Inc.  
Source: Cube LUT Specification 1.0  
https://wwwimages2.adobe.com/content/dam/acom/en/products/speedgrade/cc/pdfs/cube-lut-specification-1.0.pdf