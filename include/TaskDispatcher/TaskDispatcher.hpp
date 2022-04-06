#pragma once
#include <boost/program_options.hpp>
#include <iostream>
#include <thread>

#include <ImageProcess/Processor.hpp>
#include <GPUImageProcess/GPUprocessor.hpp>
#include <DataLoader/dataLoader.hpp>

class TaskDispatcher
{
	int argCount;
	char** args;
	DataLoader loader;

public:
	TaskDispatcher(int aCnt, char** aVal);
	int start();
	boost::program_options::variables_map parseInputArgs(int argc, char* aVal[]) const;
};
