#include <FileIO/CubeLUT.hpp>
#include <ImageProcessing/CPU/LUT3DPipelineCPU.hpp>
#include <ImageProcessing/CPU/TrilinearImplCPU.hpp>
#include <color/deltaE.hpp>
#include <fmt/format.h>
#include <fstream>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

using namespace ::testing;

// A higher-level test used to track regressions.
// It does so by detecting an increase of the color difference from the "state-of-the-art" implementation
// (reference images were processed in DaVinci Resolve 18.5)
class InterpolationRegressionTest : public ::testing::Test {
public:
	const std::string lutPath = "resources/17p_cmp.cube";
};

TEST_F(InterpolationRegressionTest, SLOW_trilinear) {
	CubeLUT lut;
	std::ifstream lutStream(lutPath);
	ASSERT_NO_THROW(lut.loadCubeFile(lutStream));

	auto source = cv::imread("resources/orig_cmp.png");
	auto reference = cv::imread("resources/cmp_tri.png");

	auto lut3d = std::get<Table3D>(lut.getTable());
	TrilinearImplCPU interpolator(&lut3d);
	auto result = interpolator.execute(source, 1.0f, 1);

	const auto sumDiff = color::getTotalDifferenceCIEDE2000(result, reference);
	const auto avgDiff = sumDiff / source.total();
	std::cout << fmt::format("CIEDE2000 Color difference sum: {} Avg color difference per pixel: {}\n", sumDiff, avgDiff);

	EXPECT_LT(sumDiff, 217500.0);
	EXPECT_LT(avgDiff, 0.21);

	const auto physicalThreads = std::thread::hardware_concurrency();
	const auto threadsToUse = physicalThreads != 0 ? physicalThreads : 8;
	result = interpolator.execute(source, 1.0f, threadsToUse);
	const auto sumDiffMultiThread = color::getTotalDifferenceCIEDE2000(result, reference);
	const auto avgDiffMultiThread = sumDiffMultiThread / source.total();
	EXPECT_EQ(sumDiff, sumDiffMultiThread) << "Multi-threading influenced the sum of the color difference";
	EXPECT_EQ(avgDiff, avgDiffMultiThread) << "Multi-threading influenced the average color difference";
}
