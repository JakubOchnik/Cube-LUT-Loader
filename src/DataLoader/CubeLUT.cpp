#include <DataLoader/CubeLUT.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <boost/format.hpp>

/**
 * @file CubeLUT.cpp
 * @author Jakub Ochnik
 * @note Parts of the .cube file parser code are courtesy of Adobe Systems Inc.
 * It is licensed under the Creative Commons Attribution Non-Commercial 3.0.
 * Source: https://web.archive.org/web/20220220033515/https://wwwimages2.adobe.com/content/dam/acom/en/products/speedgrade/cc/pdfs/cube-lut-specification-1.0.pdf
 */

constexpr char QUOTE_SYMBOL = '"';
constexpr char NEWLINE_SEPARATOR = '\n';
constexpr char COMMENT_MARKER = '#';

CubeLUT::CubeLUT() : parsingStatus{NotInitialized} {}

std::string CubeLUT::readLine(std::ifstream& infile)
{
	// Skip empty lines and comments
	std::string textLine;
	while (textLine.empty() || textLine[0] == COMMENT_MARKER)
	{
		if (infile.eof())
		{
			parsingStatus = PrematureEndOfFile;
			break;
		}
		std::getline(infile, textLine, NEWLINE_SEPARATOR);
		if (infile.fail())
		{
			parsingStatus = ReadError;
			break;
		}
		if (!textLine.empty() && textLine.back() == '\r') {
			// Strip \r from line endings (relevant for files with CRLF line endings)
			textLine.pop_back();
		}
	}
	return textLine;
}

float CubeLUT::clipValue(float input, int channel) const {
	if (input < domainMin[channel]) {
		return domainMin[channel];
	}
	if (input > domainMax[channel]) {
		return domainMax[channel];
	}
	return input;
}

void CubeLUT::parseTableRow3D(const std::string& lineOfText, const int r, const int g, const int b)
{
	// Parse values from the file and assign them to the LUT tensor (4D matrix)
	const int N = 3;
	std::istringstream line(lineOfText);
	float tmp;
	auto& lut3d = std::get<Table3D>(table);
	for (int i{0}; i < N; ++i)
	{
		line >> tmp;
		if (line.fail())
		{
			parsingStatus = CouldNotParseTableData;
			break;
		}
		else if (tmp < domainMin[i] || tmp > domainMax[i])
		{
			if (!domainViolationDetected)
			{
				domainViolationDetected = true;
				std::cerr << boost::format("[WARNING] Detected LUT values outside of domain <%1% - %2%>. Clipping the input.\n") % domainMin[i] % domainMax[i];
			}
			tmp = clipValue(tmp, i);
		}
		lut3d(r, g, b, i) = tmp;
	}
}
void CubeLUT::parseTableRow1D(const std::string& lineOfText, const int i)
{
	// Parse values from the file and assign them to the LUT tensor (2D matrix)
	const int N = 3;
	std::istringstream line(lineOfText);
	float tmp;
	auto& lut1d = std::get<Table1D>(table);
	for (int j{0}; j < N; ++j)
	{
		line >> tmp;
		if (line.fail())
		{
			parsingStatus = CouldNotParseTableData;
			break;
		}
		lut1d(i, j) = tmp;
	}
}

void CubeLUT::parseLUTParameters(std::ifstream& infile, long& linePos) {
	bool titleFound{false};
	bool domainMinFound{false}, domainMaxFound{false};
	bool lutSizeFound{false};

	while (parsingStatus == OK)
	{
		linePos = infile.tellg();
		std::string lineOfText = readLine(infile);
		if (parsingStatus != OK)
		{
			break;
		}

		std::istringstream line(lineOfText);
		std::string keyword;
		line >> keyword;

		if (keyword > "+"  && keyword < ":") // number
		{
			infile.seekg(linePos);
			// finished parsing parameters
			break;
		}
		if (keyword == "TITLE" && !titleFound)
		{
			char startOfTitle;
			line >> startOfTitle;
			if (startOfTitle != QUOTE_SYMBOL)
			{
				std::cerr << "[WARNING] Missing quote for the LUT title";
				continue;
			}
			titleFound = true;
			std::getline(line, title, QUOTE_SYMBOL);
		}
		else if (keyword == "DOMAIN_MIN" && !domainMinFound)
		{
			line >> domainMin[0] >> domainMin[1] >> domainMin[2];
		}
		else if (keyword == "DOMAIN_MAX" && !domainMaxFound)
		{
			line >> domainMax[0] >> domainMax[1] >> domainMax[2];
		}
		else if (keyword == "LUT_1D_SIZE" && !lutSizeFound)
		{
			line >> size;
			if (size < 2 || size > 65536)
			{
				parsingStatus = LUTSizeOutOfRange;
				break;
			}
			type = LUTType::LUT1D;
		}
		else if (keyword == "LUT_3D_SIZE" && !lutSizeFound)
		{
			line >> size;
			if (size < 2 || size > 256)
			{
				parsingStatus = LUTSizeOutOfRange;
				break;
			}
			type = LUTType::LUT3D;
		}
		else
		{
			std::cerr << "[WARNING] Unknown or repeated keyword: " << keyword;
			continue;
		}

		if (line.fail())
		{
			parsingStatus = ReadError;
			break;
		}
	}
}

void CubeLUT::parseLUTTable(std::ifstream& infile) {
	if (type == LUTType::LUT1D) {
		table = Table1D(size, 3);
		auto& lut1d = std::get<Table1D>(table);
		const auto N = lut1d.dimension(0);
		for (int i{0}; i < N && parsingStatus == OK; ++i)
		{
			parseTableRow1D(readLine(infile), i);
		}
	} else if (type == LUTType::LUT3D) {
		auto N = size;
		table = Table3D(N, N, N, 3);
		auto& lut3d = std::get<Table3D>(table);
		N = lut3d.dimension(0);
		for (int b{0}; b < N && parsingStatus == OK; ++b)
		{
			for (int g{0}; g < N && parsingStatus == OK; ++g)
			{
				for (int r{0}; r < N && parsingStatus == OK; ++r)
				{
					parseTableRow3D(readLine(infile), r, g, b);
				}
			}
		}
	}
}

CubeLUT::LUTState CubeLUT::loadCubeFile(std::ifstream& infile)
{
	parsingStatus = OK;
	title.clear();

	long linePos = 0;
	parseLUTParameters(infile, linePos);
	
	if (!hasType()) {
		return CouldNotParseParams;
	}

	if (!hasRange()) {
		// no range
	}

	if (parsingStatus == LUTSizeOutOfRange)
	{
		// no size or out of range
		return LUTSizeOutOfRange;
	}

	if (parsingStatus == OK && domainMin[0] >= domainMax[0] || domainMin[1] >= domainMax[1] || domainMin[2] >= domainMax[2])
	{
		// reverse domain bounds
		return DomainBoundsReversed;
	}

	// Rewind the file to the beginning of LUT table
	infile.seekg(linePos - 1);
	while (linePos > 0 && infile.get() != '\n')
	{
		infile.seekg(--linePos);
	}

	parseLUTTable(infile);
	return parsingStatus;
}

LUTType CubeLUT::getType() const
{
	return type;
}

const std::variant<Table1D, Table3D>& CubeLUT::getTable() const {
	return table;
}
