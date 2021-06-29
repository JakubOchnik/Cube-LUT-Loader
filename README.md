# Cube LUT Loader
A C++ app which lets you apply a Cube LUT to an image.

## Dependencies
- OpenCV 4.5.1

## Working
- 3D LUT support
- Nearest value mode (fast, but low quality of tonal transitions)
- Trilinear interpolation (slower, but higher quality)

## In progress
- 1D LUT support
- Simple command-line interface

### Legal disclaimer
The .cube LUT parser is used under Creative Commons Attribuition Non-Commercial 3.0 License.
It was created by Adobe Inc.  
Source: Cube LUT Specification 1.0  
https://wwwimages2.adobe.com/content/dam/acom/en/products/speedgrade/cc/pdfs/cube-lut-specification-1.0.pdf