#include "deltaE.hpp"
#define _USE_MATH_DEFINES
#include <fmt/format.h>
#include <math.h>

namespace {
double rad2deg(double rad) { return rad * (180.0 / M_PI); }

double deg2rad(double deg) { return deg * (M_PI / 180.0); }

double getPixelDiffrenceCIEDE2000(cv::Vec3f inputColor, cv::Vec3f referenceColor) {
	const double avgL = (inputColor[0] + referenceColor[0]) / 2.0;
	const double c1 = std::sqrt(std::pow(inputColor[1], 2) + std::pow(inputColor[2], 2));
	const double c2 = std::sqrt(std::pow(referenceColor[1], 2) + std::pow(referenceColor[2], 2));
	const double avgC = (c1 + c2) / 2.0;
	const double g = (1 - std::sqrt(std::pow(avgC, 7) / (std::pow(avgC, 7) + std::pow(25, 7)))) / 2;

	const double a1p = inputColor[1] * (1 + g);
	const double a2p = referenceColor[1] * (1 + g);

	const double c1p = std::sqrt(std::pow(a1p, 2) + std::pow(inputColor[2], 2));
	const double c2p = std::sqrt(std::pow(a2p, 2) + std::pow(referenceColor[2], 2));

	const double avgCp = (c1p + c2p) / 2.0;

	double h1p;
	if (inputColor[2] == 0 && a1p == 0) {
		h1p = 0.0;
	} else {
		h1p = rad2deg(std::atan2(inputColor[2], a1p));
		if (h1p < 0) {
			h1p += 360;
		}
	}

	double h2p;
	if (referenceColor[2] == 0 && a2p == 0) {
		h2p = 0.0;
	} else {
		h2p = rad2deg(std::atan2(referenceColor[2], a2p));
		if (h2p < 0) {
			h2p += 360;
		}
	}

	double deltahp, avghp;
	if (c1p == 0.0f || c2p == 0.0f) {
		deltahp = 0.0f;
		avghp = h1p + h2p;
	} else {
		deltahp = h2p - h1p;
		if (std::abs(deltahp) > 180) {
			if (h2p < h1p) {
				deltahp += 360;
			} else {
				deltahp -= 360;
			}
		}

		avghp = (h1p + h2p) / 2;
		if (std::abs(h1p - h2p) > 180) {
			if (h1p + h2p < 360) {
				avghp += 180;
			}
			if (h1p + h2p >= 360) {
				avghp -= 180;
			}
		}
	}

	const double t = 1 - 0.17 * std::cos(deg2rad(avghp - 30)) + 0.24 * std::cos(deg2rad(2 * avghp)) +
					 0.32 * std::cos(deg2rad(3 * avghp + 6)) - 0.2 * std::cos(deg2rad(4 * avghp - 63));

	const double deltalp = inputColor[0] - referenceColor[0];
	const double deltacp = c2p - c1p;

	deltahp = 2 * std::sqrt(c1p * c2p) * std::sin(deg2rad(deltahp) / 2.0);

	const double sl = 1 + ((0.015 * std::pow(avgL - 50, 2)) / std::sqrt(20 + std::pow(avgL - 50, 2)));
	const double sc = 1 + 0.045 * avgCp;
	const double sh = 1 + 0.015 * avgCp * t;

	const double deltaro = 30 * std::exp(-(std::pow((avghp - 275) / 25, 2)));
	const double rc = 2 * std::sqrt(std::pow(avgCp, 7) / (std::pow(avgCp, 7) + std::pow(25, 7)));
	const double rt = -rc * std::sin(2 * deg2rad(deltaro));

	constexpr double kl = 1;
	constexpr double kc = 1;
	constexpr double kh = 1;

	const double deltaE =
		std::sqrt(std::pow(deltalp / (kl * sl), 2) + std::pow(deltacp / (kc * sc), 2) +
				  std::pow(deltahp / (kh * sh), 2) + rt * (deltacp / (kc * sc)) * (deltahp / (kh * sh)));
	return deltaE;
}

cv::Mat bgrToLab(cv::Mat src) {
	cv::Mat floatMat;
	src.convertTo(floatMat, CV_32FC3);

	cv::Mat output;
	cv::cvtColor(floatMat / 255.0f, output, cv::COLOR_BGR2Lab);
	return output;
}

} // namespace
namespace color {
double getTotalDifferenceCIEDE2000(cv::Mat src, cv::Mat ref) {
	if (src.size != ref.size) {
		return 0;
	}

	cv::Mat srcLab = bgrToLab(src);
	cv::Mat refLab = bgrToLab(ref);

	double totalDifference = {};
	for (int row = 0; row < srcLab.rows; ++row) {
		const auto srcRow = srcLab.ptr<cv::Vec3f>(row);
		const auto refRow = refLab.ptr<cv::Vec3f>(row);
		for (int col = 0; col < srcLab.cols; ++col) {
			totalDifference += getPixelDiffrenceCIEDE2000(srcRow[col], refRow[col]);
		}
	}
	return totalDifference;
}
} // namespace color