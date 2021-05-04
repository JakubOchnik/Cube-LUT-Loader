#include "CubeLUT.h"
#include <iostream>
#include <sstream>

// Credit of this parser code: Adobe Inc.
// Cube LUT Specification 1.0
// https://wwwimages2.adobe.com/content/dam/acom/en/products/speedgrade/cc/pdfs/cube-lut-specification-1.0.pdf

string CubeLUT::ReadLine(ifstream& infile, char lineSeparator)
{
	// Skip empty lines and comments
	const char CommentMarker = '#';
	string textLine("");
	while (textLine.size() == 0 || textLine[0] == CommentMarker)
	{
		if (infile.eof())
		{
			status = PrematureEndOfFile;
			break;
		}
		getline(infile, textLine, lineSeparator);
		if (infile.fail())
		{
			status = ReadError;
			break;
		}
	}
	return textLine;
}

vector<float> CubeLUT::ParseTableRow(const string& lineOfText)
{
	int N = 3;
	tableRow f(N);
	istringstream line(lineOfText);
	for (int i{ 0 }; i < N; ++i)
	{
		line >> f[i];
		if (line.fail())
		{
			status = CouldNotParseTableData;
			break;
		}
	}
	return f;
}

CubeLUT::LUTState CubeLUT::LoadCubeFile(ifstream& infile)
{
	// defaults
	status = OK;
	title.clear();
	domainMin = tableRow(3, 0.0);
	domainMax = tableRow(3, 1.0);

	LUT1D.clear();
	LUT3D.clear();

	const char NewlineCharacter = '\n';
	char lineSeparator = NewlineCharacter;

	const char CarriageReturnCharacter = '\r';
	for (int i{ 0 }; i < 255; ++i)
	{
		char inc = infile.get();
		if (inc == NewlineCharacter)
			break;
		if (inc == CarriageReturnCharacter)
		{
			if (infile.get() == NewlineCharacter)
				break;
			lineSeparator = CarriageReturnCharacter;
			clog << "INFO: This file uses non-compliant line separator \\r (0x0D)" << endl;
		}
		if (i > 250)
		{
			status = LineError;
			break;
		}
	}
	infile.seekg(0);
	infile.clear();

	int N, CntTitle, CntSize, CntMin, CntMax;
	N = CntTitle = CntSize = CntMin = CntMax = 0;
	long linePos;
	while (status == OK)
	{
		linePos = infile.tellg();
		string lineOfText = ReadLine(infile, lineSeparator);
		if (!status == OK)
			break;

		istringstream line(lineOfText);
		string keyword;
		line >> keyword;

		if ("+" < keyword && keyword < ":") //numbers
		{
			infile.seekg(linePos);
			break;
		}
		else if (keyword == "TITLE" && CntTitle++ == 0)
		{
			const char QUOTE = '"';
			char startOfTitle;
			line >> startOfTitle;
			if (startOfTitle != QUOTE)
			{
				status = TitleMissingQuote;
				break;
			}
			getline(line, title, QUOTE);
		}
		else if (keyword == "DOMAIN_MIN" && CntMin++ == 0)
		{
			line >> domainMin[0] >> domainMin[1] >> domainMin[2];
		}
		else if (keyword == "DOMAIN_MAX" && CntMax++ == 0)
		{
			line >> domainMax[0] >> domainMax[1] >> domainMax[2];
		}
		else if (keyword == "LUT_1D_SIZE" && CntSize++ == 0)
		{
			line >> N;
			if (N < 2 || N>65536)
			{
				status = LUTSizeOutOfRange;
				break;
			}
			LUT1D = table1D(N, tableRow(3));
		}
		else if (keyword == "LUT_3D_SIZE" && CntSize++ == 0)
		{
			line >> N;
			if (N < 2 || N>256)
			{
				status = LUTSizeOutOfRange;
				break;
			}
			LUT3D = table3D(N, table2D(N, table1D(N, tableRow(3))));
		}
		else
		{
			status = UnknownOrRepeatedKeyword;
			break;
		}

		if (line.fail())
		{
			status = ReadError;
			break;
		}
	}

	if (status == OK && CntSize == 0)
		status = LUTSizeOutOfRange;
	if (status == OK && domainMin[0] >= domainMax[0] || domainMin[1] >= domainMax[1] || domainMin[2] >= domainMax[2])
		status = DomainBoundsReversed;

	// parsowanie wczytanych danych

	if (LUT1D.size() > 0)
	{
		N = LUT1D.size();
		for (int i{ 0 }; i < N && status == OK; ++i)
		{
			LUT1D[i] = ParseTableRow(ReadLine(infile, lineSeparator));
		}
	}
	else
	{
		N = LUT3D.size();
		//infile.seekg(linePos);
		//getline(infile, NULL);
		//cout << infile.get();
		char current = infile.get();
		/*while (current == '\n' || current == '\t')
		{
			current = infile.get();
		}*/
		if (current < 48 || current > 57)
			ReadLine(infile, lineSeparator);
		else
			infile.seekg(infile.tellg());

		for (int b{ 0 }; b < N && status == OK; ++b)
		{
			for (int g{ 0 }; g < N && status == OK; ++g)
			{
				for (int r{ 0 }; r < N && status == OK; ++r)
				{
					LUT3D[r][g][b] = ParseTableRow
					(ReadLine(infile, lineSeparator));
				}
			}
		}
	}
	return status;
}