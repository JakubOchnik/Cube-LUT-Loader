#pragma once
#include <boost/program_options.hpp>
#include <iostream>

class InitHandler
{
	int arg_count;
	char** args;

public:
	InitHandler(int aCnt, char* aVal[]);
	int start();
	boost::program_options::variables_map parseInputArgs(const int argc, char** argv) const;

};