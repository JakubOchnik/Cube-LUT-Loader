#pragma once
#include <FileIO/FileIO.hpp>
#include <args.hxx>
#include <TaskDispatcher/InputParams.h>

class TaskDispatcher
{
	int argCount;
	char** argv;

public:
	TaskDispatcher(int aCnt, char** aVal);

	int start();
	InputParams parseInputArgs() const;
};
