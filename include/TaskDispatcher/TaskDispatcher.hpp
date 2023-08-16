#pragma once
#include <DataLoader/dataLoader.hpp>
#include <boost/program_options.hpp>
#include <variant>

class TaskDispatcher
{
	int		   argCount;
	char**	   args;

public:
	TaskDispatcher(int aCnt, char** aVal);

	int start();

	std::variant<boost::program_options::variables_map, boost::program_options::options_description> parseInputArgs(int argc,
																													char *aVal[]) const;
};
