from conan import ConanFile
from conan.tools.cmake import cmake_layout, CMakeToolchain, CMakeDeps
from conan.tools.microsoft import is_msvc

class CubeLUTLoader(ConanFile):
    name = "Cube LUT Loader"
    requires = "taywee-args/6.4.6", "opencv/4.9.0", "eigen/3.4.0", "fmt/10.2.1"
    settings = "os", "build_type", "compiler", "arch"
    build_policy = "missing"
    options = {
        "shared": [True, False]
    }
    default_options = {
        "shared": False
    }

    def layout(self):
        cmake_layout(self)

    def configure(self):
        if "shared" in self.options:
            self.options["opencv/*"].shared = self.options.shared

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()

        tc = CMakeToolchain(self)
        if is_msvc(self) and "vs_runtime" in tc.blocks.keys():
            tc.blocks.remove("vs_runtime")
        tc.generate()
