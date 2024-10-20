from conan import ConanFile
from conan.tools.cmake import cmake_layout

class CubeLUTLoader(ConanFile):
    name = "Cube LUT Loader"
    requires = "taywee-args/6.4.6", "opencv/4.9.0", "eigen/3.4.0", "fmt/10.2.1"
    generators = "CMakeDeps", "CMakeToolchain"
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
