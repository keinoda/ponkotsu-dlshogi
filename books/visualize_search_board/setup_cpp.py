from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "mcts_cpp",
        ["mcts_cpp.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3", "-DNDEBUG", "-fno-math-errno", "-fno-trapping-math"],
        extra_link_args=["-O3"],
    )
]

setup(
    name="mcts_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
