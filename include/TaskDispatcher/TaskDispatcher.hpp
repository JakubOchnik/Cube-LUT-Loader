#pragma once
#include <FileIO/FileIO.hpp>
#include <boost/program_options.hpp>
#include <TaskDispatcher/InputParams.h>

class TaskDispatcher
{
	int argCount;
	char** args;

public:
	TaskDispatcher(int aCnt, char** aVal);

	int start();
	InputParams parseInputArgs(std::string& helpText) const;
};
