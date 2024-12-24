#pragma once
#include <gmock/gmock.h>
#include <ImageProcessing/CPU/CPUModeExecutor.hpp>

class CPUProcessorMock: public CPUModeExecutor {
public:
    using CPUModeExecutor::CPUModeExecutor;
    MOCK_METHOD(cv::Mat, process, (float strength, InterpolationMethod method), (override));
};
