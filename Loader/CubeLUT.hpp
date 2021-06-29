#pragma once
#ifndef CubeLUT_H
#define CubeLUT_H

#include <string>
#include <vector>
#include <fstream>

using namespace std;

// Licensed under Creative Commons Attribution Non-Commercial 3.0 License
// Author: Adobe Inc.
// Source:
// "Cube LUT Specification 1.0"
// https://wwwimages2.adobe.com/content/dam/acom/en/products/speedgrade/cc/pdfs/cube-lut-specification-1.0.pdf

class CubeLUT {
public:
	typedef vector<float> tableRow;
	typedef vector<tableRow> table1D;
	typedef vector<table1D> table2D;
	typedef vector<table2D> table3D;

	enum LUTState { OK = 0, NotInitialized = 1, ReadError = 10, WriteError, PrematureEndOfFile, LineError, UnknownOrRepeatedKeyword = 20, TitleMissingQuote, DomainBoundsReversed, LUTSizeOutOfRange, CouldNotParseTableData };
	LUTState status;
	string title;
	tableRow domainMin;
	tableRow domainMax;
	table1D LUT1D;
	table3D LUT3D;

	CubeLUT()
	{
		status = NotInitialized;
	}

	LUTState LoadCubeFile(ifstream& infile);

private:
	string ReadLine(ifstream& infile, char lineSeparator);
	tableRow ParseTableRow(const string& lineOfText);
};

#endif