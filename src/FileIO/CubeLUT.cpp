#include <FileIO/CubeLUT.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <fmt/format.h>

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
constexpr unsigned char N_CHANNELS = 3;
constexpr uint32_t LUT_1D_MAX_SIZE = 65536;
constexpr uint32_t LUT_1D_MIN_SIZE = 2;
constexpr uint32_t LUT_3D_MAX_SIZE = 256;
constexpr uint32_t LUT_3D_MIN_SIZE = 2;

std::string CubeLUT::readLine(std::istream& infile)
{
	// Return the next line, but:
	// - skip empty lines and comments
	// - strip CR symbols
	std::string textLine;
	while (textLine.empty() || textLine[0] == COMMENT_MARKER)
	{
		if (infile.eof())
		{
			throw std::runtime_error{"Premature EOF"};
		}
		std::getline(infile, textLine, NEWLINE_SEPARATOR);
		if (infile.fail())
		{
			throw std::runtime_error{"Read error"};
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

float CubeLUT::parseColorValue(std::istringstream& line, unsigned char channel) {
	float tmp;
	line >> tmp;
	if (line.fail())
	{
		throw std::runtime_error("Could not parse table data");
	}
	else if (tmp < domainMin[channel] || tmp > domainMax[channel])
	{
		if (!domainViolationDetected)
		{
			domainViolationDetected = true;
			std::cerr << fmt::format("[WARNING] Detected LUT values outside of domain <{} - {}>. Clipping the input.\n", domainMin[channel], domainMax[channel]);
		}
		tmp = clipValue(tmp, channel);
	}
	return tmp;
}

void CubeLUT::parseTableRow3D(const std::string& lineOfText, const int r, const int g, const int b)
{
	// Parse values from the file and assign them to the LUT tensor (4D matrix)
	std::istringstream line(lineOfText);
	auto& lut3d = std::get<Table3D>(table);
	for (unsigned char ch{0}; ch < N_CHANNELS; ++ch)
	{
		lut3d(r, g, b, ch) = parseColorValue(line, ch);
	}
}
void CubeLUT::parseTableRow1D(const std::string& lineOfText, const int i)
{
	// Parse values from the file and assign them to the LUT tensor (2D matrix)
	std::istringstream line(lineOfText);
	auto& lut1d = std::get<Table1D>(table);
	for (unsigned char ch{0}; ch < N_CHANNELS; ++ch)
	{
		lut1d(i, ch) = parseColorValue(line, ch);
	}
}

bool CubeLUT::parseLUTParameters(std::istream& infile, std::streamoff& linePos) {
	bool titleFound{false};
	bool domainMinFound{false}, domainMaxFound{false};
	bool lutSizeFound{false};

	if (infile.fail() || infile.eof()) {
		return false;
	}

	while (!infile.fail() && !infile.eof())
	{
		linePos = infile.tellg();
		std::string lineOfText = readLine(infile);

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
				std::cerr << "[WARNING] Missing quote for the LUT title\n";
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
			if (size < LUT_1D_MIN_SIZE || size > LUT_1D_MAX_SIZE)
			{
				const auto errorMsg = fmt::format("1D LUT size ({}) is out of range <{}, {}>", size, LUT_1D_MIN_SIZE, LUT_1D_MAX_SIZE);
				throw std::runtime_error(errorMsg);
			}
			type = LUTType::LUT1D;
		}
		else if (keyword == "LUT_3D_SIZE" && !lutSizeFound)
		{
			line >> size;
			if (size < LUT_3D_MIN_SIZE || size > LUT_3D_MAX_SIZE)
			{
				const auto errorMsg = fmt::format("3D LUT size ({}) is out of range <{}, {}>", size, LUT_3D_MIN_SIZE, LUT_3D_MAX_SIZE);
				throw std::runtime_error(errorMsg);
			}
			type = LUTType::LUT3D;
		}
		else
		{
			std::cerr << fmt::format("[WARNING] Unknown or repeated keyword: {} \n", keyword);
		}

		if (line.fail())
		{
			return false;
		}
	}
	return true;
}

void CubeLUT::parseLUTTable(std::istream& infile) {
	if (type == LUTType::LUT1D) {
		table = Table1D(size, 3);
		auto& lut1d = std::get<Table1D>(table);
		const auto N = lut1d.dimension(0);
		for (int i{0}; i < N; ++i)
		{
			parseTableRow1D(readLine(infile), i);
		}
	} else if (type == LUTType::LUT3D) {
		int64_t N = size;
		table = Table3D(N, N, N, 3);
		auto& lut3d = std::get<Table3D>(table);
		N = lut3d.dimension(0);
		for (int b{0}; b < N; ++b)
		{
			for (int g{0}; g < N; ++g)
			{
				for (int r{0}; r < N; ++r)
				{
					parseTableRow3D(readLine(infile), r, g, b);
				}
			}
		}
	}
}

void CubeLUT::loadCubeFile(std::istream& infile)
{
	clear();

	std::streamoff linePos = 0;
	if (!parseLUTParameters(infile, linePos)) {
		throw std::runtime_error("Failed to read LUT file");
	}
	
	if (!hasType()) {
		throw std::runtime_error("Unknown LUT type: specify the LUT_1D_SIZE/LUT_3D_SIZE tag");
	}

	if (domainMin[0] >= domainMax[0] || domainMin[1] >= domainMax[1] || domainMin[2] >= domainMax[2])
	{
		throw std::runtime_error("Domain bounds are reversed (DOMAIN_MIN is larger than DOMAIN_MAX)");
	}

	// Rewind the file to the beginning of LUT table
	infile.seekg(linePos - 1);
	while (linePos > 0 && infile.get() != '\n')
	{
		infile.seekg(--linePos);
	}

	if (infile.fail()) {
		throw std::runtime_error("Failed to parse LUT values");
	}
	parseLUTTable(infile);
}

LUTType CubeLUT::getType() const
{
	return type;
}

const std::variant<Table1D, Table3D>& CubeLUT::getTable() const {
	return table;
}

void CubeLUT::clear() {
	type = LUTType::UNKNOWN;
	title.clear();
	domainMin = {0.0, 0.0, 0.0};
	domainMax = {1.0, 1.0, 1.0};
	size = 0;
	domainViolationDetected = false;
	if (table.index() == 0) {
		auto& table1D = std::get<Table1D>(table);
		table1D = Table1D();
	} else {
		auto& table3D = std::get<Table3D>(table);
		table3D = Table3D();
	}
}
