#pragma once
#include <boost/program_options.hpp>
#include <iostream>
#include <thread>

#include <ImageProcess/Processor.hpp>
//#include <GPUImageProcess/GPUprocessor.hpp>
#include <Loader/Loader.hpp>


class InitHandler
{
	int arg_count;
	char** args;
	Loader loader;

public:
	InitHandler(int aCnt, char** aVal);
	int start();
	boost::program_options::variables_map parseInputArgs(int argc, char* aVal[]) const;
};
