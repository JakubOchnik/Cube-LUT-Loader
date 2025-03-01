#include <FileIO/CubeLUT.hpp>
#include <ImageProcessing/CPU/LUT3DPipelineCPU.hpp>
#include <ImageProcessing/CPU/TetrahedralImplCPU.hpp>
#include <ImageProcessing/CPU/TrilinearImplCPU.hpp>
#include <TaskDispatcher/InputParams.h>
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
	const std::string originalImagePath = "resources/orig_cmp.png";
};

struct ReferenceTestParams {
	std::string referencePath;
	InterpolationMethod type;
	double maxSum;
	double maxAvg;
	bool compareMultiThreaded;
};

struct ReferenceDistance : public InterpolationRegressionTest,
						   public ::testing::WithParamInterface<ReferenceTestParams> {
public:
	cv::Mat getResultWithInterpolator(InterpolationMethod type, Table3D& lut3d, cv::Mat source, bool threads = 1) {
		cv::Mat result;
		if (type == InterpolationMethod::Tetrahedral) {
			TetrahedralImplCPU interpolator(&lut3d);
			result = interpolator.execute(source, 1.0f, threads);
		} else {
			TrilinearImplCPU interpolator(&lut3d);
			result = interpolator.execute(source, 1.0f, threads);
		}
		return result;
	}
};

TEST_P(ReferenceDistance, SLOW_referenceComparison) {
	CubeLUT lut;
	std::ifstream lutStream(lutPath);
	ASSERT_NO_THROW(lut.loadCubeFile(lutStream));

	const auto params = GetParam();
	const auto source = cv::imread(originalImagePath);
	auto reference = cv::imread(params.referencePath);

	auto lut3d = std::get<Table3D>(lut.getTable());
	cv::Mat result = getResultWithInterpolator(params.type, lut3d, source);

	const auto sumDiff = color::getTotalDifferenceCIEDE2000(result, reference);
	const auto avgDiff = sumDiff / source.total();
	std::cout << fmt::format("CIEDE2000 Color difference sum: {} Avg color difference per pixel: {}\n", sumDiff,
							 avgDiff);

	EXPECT_LT(sumDiff, params.maxSum);
	EXPECT_LT(avgDiff, params.maxAvg);

	if (params.compareMultiThreaded) {
		const auto physicalThreads = std::thread::hardware_concurrency();
		const auto threadsToUse = physicalThreads != 0 ? physicalThreads : 8;
		result = getResultWithInterpolator(params.type, lut3d, source, threadsToUse);
		const auto sumDiffMultiThread = color::getTotalDifferenceCIEDE2000(result, reference);
		const auto avgDiffMultiThread = sumDiffMultiThread / source.total();
		EXPECT_EQ(sumDiff, sumDiffMultiThread) << "Multi-threading influenced the sum of the color difference";
		EXPECT_EQ(avgDiff, avgDiffMultiThread) << "Multi-threading influenced the average color difference";
	}
}

INSTANTIATE_TEST_SUITE_P(
	InterpolationRegressionTest,
	ReferenceDistance,
	::testing::Values(
		ReferenceTestParams{"resources/cmp_tri.png", InterpolationMethod::Trilinear, 217'500.0, 0.21, true},
		ReferenceTestParams{"resources/cmp_tet.png", InterpolationMethod::Tetrahedral, 181'500.0, 0.18, true}
	),
	[](const testing::TestParamInfo<ReferenceTestParams>& info) {
		switch (info.param.type) {
			case InterpolationMethod::Trilinear:
				return "Trilinear";
				break;
			case InterpolationMethod::Tetrahedral:
				return "Tetrahedral";
				break;
			default:
				return "Other";
		}
	}
);
