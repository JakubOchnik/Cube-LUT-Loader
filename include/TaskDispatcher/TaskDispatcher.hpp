#pragma once
#include <DataLoader/dataLoader.hpp>
#include <boost/program_options.hpp>

class TaskDispatcher
{
	int		   argCount;
	char**	   args;
	DataLoader loader;

public:
	TaskDispatcher(int aCnt, char** aVal);

	int start();

	boost::program_options::variables_map parseInputArgs(int   argc,
														 char* aVal[]) const;
};
