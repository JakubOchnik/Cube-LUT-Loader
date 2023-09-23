#pragma once

#include <Eigen/Dense>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <variant>

enum class LUTType {
	LUT1D,
	LUT3D,
	UNKNOWN
};

using Table1D  = Eigen::Tensor<float, 2>;
using Table3D  = Eigen::Tensor<float, 4>;

class CubeLUT
{
public:
	using TableRow = Eigen::Vector3f;

	enum LUTState
	{
		OK			   = 0,
		NotInitialized = 1,
		ReadError	   = 10,
		WriteError,
		PrematureEndOfFile,
		LineError,
		DomainBoundsReversed,
		LUTSizeOutOfRange,
		CouldNotParseTableData,
		OutOfDomain,
		CouldNotParseParams
	};

	LUTState parsingStatus;
	CubeLUT();

	LUTState loadCubeFile(std::ifstream& infile);
	LUTType getType() const;
	const std::variant<Table1D, Table3D>& getTable() const;

private:
	std::variant<Table1D, Table3D> table;


	LUTType type = LUTType::UNKNOWN;
	uint32_t size = 0;
	std::string title;
	std::vector<float> domainMin{0.0f, 0.0f, 0.0f};
	std::vector<float> domainMax{1.0f, 1.0f, 1.0f};

	bool hasType() {
		return type != LUTType::UNKNOWN;
	}

	bool hasRange() {
		return size > 0;
	}

	std::string readLine(std::ifstream& infile);
	void parseTableRow3D(const std::string& lineOfText, const int r, const int g, const int b);
	void parseTableRow1D(const std::string& lineOfText, const int i);
	void parseLUTTable(std::ifstream& infile);
	float clipValue(float input, int channel) const;
	void parseLUTParameters(std::ifstream& infile, long& linePos);

	bool domainViolationDetected {false};
};
