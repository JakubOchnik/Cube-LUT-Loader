
__kernel void applyLinear1DKernel(
    __global unsigned char* image,
    const int channels,
    __read_only image1d_t LUT1D,
    const float opacity,
    const int width,
    const int height)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    const int idx = (x + y * width) * channels;

    float b = image[idx + 0] / 255.0f;
    float g = image[idx + 1] / 255.0f;
    float r = image[idx + 2] / 255.0f;

    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

    float4 lutR = read_imagef(LUT1D, sampler, r); // R channel
    float4 lutG = read_imagef(LUT1D, sampler, g); // G channel
    float4 lutB = read_imagef(LUT1D, sampler, b); // B channel

    float newR = lutR.x * 255.0f;
    float newG = lutG.y * 255.0f;
    float newB = lutB.z * 255.0f;

    image[idx + 2] = (uchar)(image[idx + 2] + (newR - image[idx + 2]) * opacity); // R
    image[idx + 1] = (uchar)(image[idx + 1] + (newG - image[idx + 1]) * opacity); // G
    image[idx + 0] = (uchar)(image[idx + 0] + (newB - image[idx + 0]) * opacity); // B
}
