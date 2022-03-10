#include <InitHandler/initHandler.hpp>

int main(int argc, char* argv[])
{
	InitHandler initHandler(argc, argv);
	return initHandler.start();
}
