from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
import platform
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))

# Find cshogi C++ source
cshogi_src_candidates = [
    os.path.join(script_dir, 'cshogi_src', 'src'),
    '/tmp/cshogi_src/src',
]

cshogi_src_dir = None
for d in cshogi_src_candidates:
    if os.path.isdir(d) and os.path.isfile(os.path.join(d, 'position.hpp')):
        cshogi_src_dir = d
        break

if cshogi_src_dir is None:
    clone_target = os.path.join(script_dir, 'cshogi_src')
    if not os.path.isdir(clone_target):
        print("Cloning cshogi source for native book parser...")
        subprocess.check_call([
            'git', 'clone', '--depth', '1', '--branch', 'v0.9.7',
            'https://github.com/TadaoYamaoka/cshogi.git', clone_target
        ])
    cshogi_src_dir = os.path.join(clone_target, 'src')

sources = ["mcts_cpp.cpp"]
include_dirs = []
define_macros = []
extra_compile_args = ["-O3", "-DNDEBUG", "-fno-math-errno", "-fno-trapping-math"]

if cshogi_src_dir and os.path.isdir(cshogi_src_dir):
    cshogi_cpps = [
        'bitboard.cpp', 'book.cpp', 'common.cpp', 'generateMoves.cpp',
        'hand.cpp', 'init.cpp', 'move.cpp', 'mt64bit.cpp',
        'position.cpp', 'search.cpp', 'square.cpp', 'usi.cpp',
        'mate.cpp', 'dfpn.cpp',
    ]
    sources.extend([os.path.join(cshogi_src_dir, f) for f in cshogi_cpps])
    include_dirs.append(cshogi_src_dir)
    define_macros.append(("USE_CSHOGI_NATIVE", None))
    extra_compile_args.append("-Wno-enum-constexpr-conversion")

    machine = platform.machine().lower()
    if machine in ('x86_64', 'amd64'):
        define_macros.extend([
            ("HAVE_SSE4", None),
            ("HAVE_SSE42", None),
            ("HAVE_AVX2", None),
        ])
        extra_compile_args.extend(["-msse4.2", "-mavx2"])
    print(f"Using cshogi native source from: {cshogi_src_dir}")

    # Patch SquareDelta enum for Apple Clang >= 21 constexpr enum range check
    square_hpp = os.path.join(cshogi_src_dir, 'square.hpp')
    if os.path.isfile(square_hpp):
        with open(square_hpp, 'r') as f:
            content = f.read()
        if 'enum SquareDelta {' in content and 'enum SquareDelta : int {' not in content:
            content = content.replace('enum SquareDelta {', 'enum SquareDelta : int {')
            with open(square_hpp, 'w') as f:
                f.write(content)
            print("Patched square.hpp: SquareDelta -> SquareDelta : int")
else:
    print("WARNING: cshogi source not found. parse_book_cpp will not be available.")

ext_modules = [
    Pybind11Extension(
        "mcts_cpp",
        sources,
        cxx_std=17,
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=["-O3"],
    )
]

setup(
    name="mcts_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
