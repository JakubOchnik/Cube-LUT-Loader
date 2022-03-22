#include <Loader/CubeLUT.hpp>

// Credit of this parser code: Adobe Inc.
// Author: Adobe Inc.
// Source:
// Cube LUT Specification 1.0
// https://wwwimages2.adobe.com/content/dam/acom/en/products/speedgrade/cc/pdfs/cube-lut-specification-1.0.pdf

string CubeLUT::ReadLine(ifstream& infile, const char lineSeparator)
{
	// Skip empty lines and comments
	const char CommentMarker = '#';
	string textLine;
	while (textLine.empty() || textLine[0] == CommentMarker)
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

void CubeLUT::ParseTableRow(const string& lineOfText, const int r, const int g, const int b)
{
	// Parse values from the file and assign them to the LUT tensor (4D matrix)
	const int N = 3;
	istringstream line(lineOfText);
	float tmp;
	for (int i{ 0 }; i < N; ++i)
	{
		line >> tmp;
		if (line.fail())
		{
			status = CouldNotParseTableData;
			break;
		}
		LUT3D(r, g, b, i) = tmp;
	}
}
void CubeLUT::ParseTableRow(const string& lineOfText, const int i)
{
	// Parse values from the file and assign them to the LUT tensor (2D matrix)
	const int N = 3;
	istringstream line(lineOfText);
	float tmp;
	for (int j{ 0 }; j < N; ++j)
	{
		line >> tmp;
		if (line.fail())
		{
			status = CouldNotParseTableData;
			break;
		}
		LUT1D(i, j) = tmp;
	}
}

CubeLUT::LUTState CubeLUT::LoadCubeFile(ifstream& infile)
{
	// defaults
	status = OK;
	title.clear();
	domainMin = {0,0,0};
	domainMax = {1,1,1};

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
		if (keyword == "TITLE" && CntTitle++ == 0)
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
			if (N < 2 || N > 65536)
			{
				status = LUTSizeOutOfRange;
				break;
			}
			LUT1D = table1D(N, 3);
		}
		else if (keyword == "LUT_3D_SIZE" && CntSize++ == 0)
		{
			line >> N;
			if (N < 2 || N > 256)
			{
				status = LUTSizeOutOfRange;
				break;
			}
			LUT3D = table3D(N, N, N, 3);
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
	infile.seekg(linePos - 1);
	while (infile.get() != '\n')
	{
		infile.seekg(--linePos);
	}
	// sprawdzenie czy jest poczatek linijki

	if (LUT1D.size() > 0)
	{
		N = LUT1D.dimension(0);
		for (int i{ 0 }; i < N && status == OK; ++i)
		{
			ParseTableRow(ReadLine(infile, lineSeparator), i);
		}
	}
	else
	{
		N = LUT3D.dimension(0);
		for (int b{ 0 }; b < N && status == OK; ++b)
		{
			for (int g{ 0 }; g < N && status == OK; ++g)
			{
				for (int r{ 0 }; r < N && status == OK; ++r)
				{
					ParseTableRow(ReadLine(infile, lineSeparator), r, g, b);
				}
			}
		}
	}
	return status;
}

bool CubeLUT::is3D() const
{
	return LUT1D.size() == 0;
}
