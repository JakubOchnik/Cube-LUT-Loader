#pragma once
#include <DataLoader/DataLoader.hpp>
#include <boost/program_options.hpp>
#include <variant>

class TaskDispatcher
{
	using VariablesMap = boost::program_options::variables_map;
	using OptionsDescription = boost::program_options::options_description;

	int argCount;
	char** args;

public:
	TaskDispatcher(int aCnt, char** aVal);

	int start();
	std::variant<VariablesMap, OptionsDescription> parseInputArgs(int argc, char *aVal[]) const;
};
