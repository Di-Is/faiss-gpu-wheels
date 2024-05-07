"""Serve package information

Copyright (c) 2024 Di-Is

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import os
from typing import List

from setuptools import Extension

from .config import Config, GPUConfig
from .type import BuildOption, BuildType, InstructionSet
from .util import get_project_root


class IncludesFactory:
    """Include diractory configure factory"""

    @classmethod
    def generate(cls, build_type: BuildType) -> List[str]:
        """generate Include diractory configure

        Args:
            build_type: build type

        Returns:
            Include diractory configure
        """
        if build_type == BuildType.CPU:
            includes = cls._generate_cpu()
        elif build_type == BuildType.GPU:
            includes = cls._generare_gpu()
        return includes

    @classmethod
    def _generate_cpu(cls) -> List[str]:
        import numpy as np

        project_root = get_project_root()
        include_dirs = [
            np.get_include(),
            os.path.join(project_root, "faiss"),
            os.path.join(Config().faiss_home, "include"),
        ]
        return include_dirs

    @classmethod
    def _generare_gpu(cls) -> List[str]:
        include_dirs = cls._generate_cpu()
        include_dirs.append(os.path.join(GPUConfig().cuda_home, "include"))
        return include_dirs

    @classmethod
    def _generarte_raft(cls) -> List[str]:
        pass


class LibrariesFactory:
    """Library diractory configure factory"""

    @classmethod
    def generate(cls, build_type: BuildType) -> List[str]:
        """generate Library diractory configure

        Args:
            build_type: build type

        Returns:
            Library diractory configure
        """
        if build_type == BuildType.CPU:
            libraries = cls._generate_cpu()
        elif build_type == BuildType.GPU:
            libraries = cls._generare_gpu()
        return libraries

    @classmethod
    def _generate_cpu(cls) -> List[str]:
        return [os.path.join(Config().faiss_home, "lib")]

    @classmethod
    def _generare_gpu(cls) -> List[str]:
        libraries = cls._generate_cpu()
        libraries += [os.path.join(GPUConfig().cuda_home, "lib64")]
        return libraries


class SwigOptions:
    @classmethod
    def generate(cls, build_type: BuildType):
        swig_opts = ["-c++", "-Doverride=", "-doxygen"]
        swig_opts += [f"-I{x}" for x in IncludesFactory.generate(build_type)]
        if build_type == BuildType.GPU:
            swig_opts.append("-DGPU_WRAPPER")
        return swig_opts


class ExtentionFactory:
    """Extension factory"""

    @staticmethod
    def generate(
        platform: str, instruction_set: InstructionSet, build_type: BuildType
    ) -> Extension:
        """generate Extension

        Args:
            platform: platform name
            instruction_set: instruction set specified when building faiss
            build_type: build type

        Returns:
            Extension object
        """
        if platform == "windows":
            raise NotImplementedError
        elif platform == "darwin":
            raise NotImplementedError
        elif platform == "linux":
            options = LinuxBuildOptionFactory.generate(instruction_set, build_type)
        else:
            raise ValueError

        root = os.path.join(get_project_root(), "faiss", "faiss", "python")
        extension = Extension(
            language="c++",
            sources=[
                os.path.join(root, "swigfaiss.i"),
                os.path.join(root, "python_callbacks.cpp"),
            ],
            depends=[os.path.join(root, "python_callbacks.h")],
            define_macros=[("FINTEGER", "int")],  # type: ignore
            include_dirs=IncludesFactory.generate(build_type),
            library_dirs=LibrariesFactory.generate(build_type),
            **options,
        )
        return extension


class LinuxBuildOptionFactory:
    """Linux build option generator"""

    @classmethod
    def generate(
        cls, instruction_set: InstructionSet, build_type: BuildType
    ) -> BuildOption:
        """generate build option for linux

        Args:
            instruction_set: instruction set specified when building faiss
            build_type: build type

        Returns:
            BuildOption object
        """
        if instruction_set == InstructionSet.GENERIC:
            option = cls._gen_generic_option(build_type)
        elif instruction_set == InstructionSet.AVX2:
            option = cls._gen_avx2_option(build_type)
        elif instruction_set == InstructionSet.AVX512:
            option = cls._gen_avx512_option(build_type)
        else:
            raise ValueError("Invalid instrunction set.")
        return option

    @classmethod
    def _gen_generic_option(self, build_type: BuildType) -> BuildOption:
        option = BuildOption()

        option["name"] = "faiss._swigfaiss"

        option["extra_compile_args"] = ["-std=c++17", "-Wno-sign-compare", "-fopenmp"]
        option["extra_compile_args"] += ["-fdata-sections", "-ffunction-sections"]

        option["extra_link_args"] = ["-fopenmp", "-lrt", "-s", "-Wl,--gc-sections"]
        option["extra_link_args"] += ["-l:libfaiss.a", "-l:libopenblas.a", "-lgfortran"]

        if build_type == BuildType.GPU:
            option["extra_link_args"].append("-l:libfaiss_gpu.a")
            if GPUConfig().dynamic_link:
                option["extra_link_args"] += [
                    "-lcublas",
                    "-lcudart",
                ]
            else:
                option["extra_link_args"] += [
                    "-lcublas_static",
                    "-lcublasLt_static",
                    "-lcudart_static",
                    "-lculibos",
                ]
        swig_opts = SwigOptions.generate(build_type)
        option["swig_opts"] = swig_opts
        option["swig_opts"] += ["-DSWIGWORDSIZE64"]
        return option

    @classmethod
    def _gen_avx2_option(cls, build_type: BuildType) -> BuildOption:
        option = cls._gen_generic_option(build_type)

        option["name"] = "faiss._swigfaiss_avx2"
        option["extra_compile_args"] += ["-mavx2", "-mfma", "-mf16c", "-mpopcnt"]
        option["extra_link_args"] = [
            x.replace("libfaiss.a", "libfaiss_avx2.a")
            for x in option["extra_link_args"]
        ]
        option["swig_opts"] += ["-module", "swigfaiss_avx2"]
        return option

    @classmethod
    def _gen_avx512_option(cls, build_type: BuildType) -> BuildOption:
        option = cls._gen_generic_option(build_type)

        option["name"] = "faiss._swigfaiss_avx512"
        option["extra_compile_args"] += [
            "-mavx2",
            "-mfma",
            "-mf16c",
            "-mavx512f",
            "-mavx512cd",
            "-mavx512vl",
            "-mavx512dq",
            "-mavx512bw",
            "-mpopcnt",
        ]
        option["extra_link_args"] = [
            x.replace("libfaiss.a", "libfaiss_avx512.a")
            for x in option["extra_link_args"]
        ]
        option["swig_opts"] += ["-module", "swigfaiss_avx512"]
        return option


class ExtensionsFactory:
    """Extensions factory"""

    @staticmethod
    def generate(
        platform: str, instruction_set: InstructionSet, build_type: BuildType
    ) -> List[Extension]:
        """generate extension list

        Args:
            platform: platform name
            instruction_set: instruction set specified when building faiss
            build_type: build type

        Returns:
            Extension object list
        """
        generate = ExtentionFactory.generate

        extensions = [generate(platform, InstructionSet.GENERIC, build_type)]
        if instruction_set in [InstructionSet.AVX2, InstructionSet.AVX512]:
            extensions.append(generate(platform, InstructionSet.AVX2, build_type))
        elif instruction_set in [InstructionSet.AVX512]:
            extensions.append(generate(platform, InstructionSet.AVX512, build_type))
        return extensions
