#pragma once

#include <Eigen/Dense>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

class CubeLUT
{
public:
	using tableRow = Eigen::Vector3f;
	using table1D  = Eigen::Tensor<float, 2>;
	using table3D  = Eigen::Tensor<float, 4>;

	enum LUTState
	{
		OK			   = 0,
		NotInitialized = 1,
		ReadError	   = 10,
		WriteError,
		PrematureEndOfFile,
		LineError,
		UnknownOrRepeatedKeyword = 20,
		TitleMissingQuote,
		DomainBoundsReversed,
		LUTSizeOutOfRange,
		CouldNotParseTableData,
		OutOfDomain
	};

	LUTState		   status;
	std::string		   title;
	std::vector<float> domainMin{0.0f, 0.0f, 0.0f};
	std::vector<float> domainMax{1.0f, 1.0f, 1.0f};
	table1D			   LUT1D;
	table3D			   LUT3D;

	CubeLUT();

	LUTState loadCubeFile(std::ifstream& infile);

	bool is3D() const;

private:
	std::string readLine(std::ifstream& infile, char lineSeparator);
	void		parseTableRow(const std::string& lineOfText,
							  const int			 r,
							  const int			 g,
							  const int			 b);
	void		parseTableRow(const std::string& lineOfText, const int i);
	float clipValue(float input, int channel) const;

	bool domainViolationDetected {false};
};
