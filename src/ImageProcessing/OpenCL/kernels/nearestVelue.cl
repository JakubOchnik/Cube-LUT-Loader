__kernel void applyNearestKernel(
    __global unsigned char* image,
    const int channels,
    __read_only image3d_t LUT3D,  
    const float opacity,
    const int width,
    const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    int idx = (x + y * width) * channels;

    float b = image[idx + 0] / 255.0f;
    float g = image[idx + 1] / 255.0f;
    float r = image[idx + 2] / 255.0f;

    const sampler_t sampler =
        CLK_NORMALIZED_COORDS_TRUE |
        CLK_ADDRESS_CLAMP |
        CLK_FILTER_NEAREST;

    float4 lut = read_imagef(LUT3D, sampler, (float4)(r, g, b, 0));

    float newR = lut.x * 255.0f;
    float newG = lut.y * 255.0f;
    float newB = lut.z * 255.0f;

    image[idx + 2] = (uchar)(image[idx + 2] + (newR - image[idx + 2]) * opacity); // R
    image[idx + 1] = (uchar)(image[idx + 1] + (newG - image[idx + 1]) * opacity); // G
    image[idx + 0] = (uchar)(image[idx + 0] + (newB - image[idx + 0]) * opacity); // B
}

