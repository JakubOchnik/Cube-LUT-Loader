#pragma once
#include <gmock/gmock.h>
#include <ImageProcessing/CPUImageProcess/CPUProcessor.hpp>

class CPUProcessorMock: public CPUProcessor {
public:
    using CPUProcessor::CPUProcessor;
    MOCK_METHOD(cv::Mat, process, (float strength, InterpolationMethod method), (override));
};
