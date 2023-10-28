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

	void loadCubeFile(std::istream& infile);
	LUTType getType() const;
	const std::variant<Table1D, Table3D>& getTable() const;

private:
	std::variant<Table1D, Table3D> table;

	LUTType type = LUTType::UNKNOWN;
	bool hasType() {
		return type != LUTType::UNKNOWN;
	}

	std::string readLine(std::istream& infile);
	void parseTableRow3D(const std::string& lineOfText, const int r, const int g, const int b);
	void parseTableRow1D(const std::string& lineOfText, const int i);
	bool parseLUTParameters(std::istream& infile, long& linePos);
	float parseColorValue(std::istringstream& line, unsigned char channel);
	void clear();

protected:
	virtual float clipValue(float input, int channel) const;
	virtual void parseLUTTable(std::istream& infile);

	uint32_t size = 0;
	std::string title;
	std::array<float, 3> domainMin{0.0f, 0.0f, 0.0f};
	std::array<float, 3> domainMax{1.0f, 1.0f, 1.0f};
	bool domainViolationDetected{false};
};
