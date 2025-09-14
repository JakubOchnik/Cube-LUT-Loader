#pragma once
#include <FileIO/FileIO.hpp>
#include <args.hxx>
#include <TaskDispatcher/InputParams.h>

enum ExitCode  {
	FAIL_EXIT = -1,
	SUCCESS_EXIT
};

class TaskDispatcher
{
	int argCount;
	char** argv;

private:
	ExitCode dispatchTask(const InputParams& parameters, FileIO& fileIO);

public:
	TaskDispatcher(int aCnt, char** aVal);

	int start();
	InputParams parseInputArgs() const;
};
