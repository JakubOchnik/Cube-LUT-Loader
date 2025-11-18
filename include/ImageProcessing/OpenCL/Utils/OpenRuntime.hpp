#include "TaskDispatcher/InputParams.h"
#include <CL/cl.h>

enum class AccessQualifier { read_write, read_only, write_only };

class OpenCLRuntime {
public:
	OpenCLRuntime() = default;
	~OpenCLRuntime() = default;

	void initialize();

	cl_mem createBuffer(const size_t size, const cl_mem_flags flags, void* hostPtr);

	std::string loadKernel(std::string filename);

	cl_kernel createKernel(const std::string& filename, const std::string& kernelNam);

	cl_program buildProgram(const char* source, size_t size);
	// cl_mam createImage();

	cl_sampler createSampler(InterpolationMethod mathod);

	void runKernel(cl_kernel kernel, size_t globalWorkSize[]);

	cl_command_queue queue() { return queue_; }

	cl_context context() { return context_; }

private:
	cl_platform_id platform_;
	cl_device_id device_;
	cl_context context_;
	cl_command_queue queue_;
};
