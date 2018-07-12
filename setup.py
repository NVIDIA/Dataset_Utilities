# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

from setuptools import Command, find_packages, setup
import os
from os import path
import glob

def get_all_files(find_dir):
    all_files = []
    for check_path in os.listdir(find_dir):
        full_path = path.join(find_dir, check_path)
        if (path.isdir(full_path)):
            all_files.extend(get_all_files(full_path))
        else:
            all_files.append(full_path)
    return all_files

_ROOT = os.path.abspath(os.path.dirname(__file__))
all_config_files = get_all_files(path.join(_ROOT, path.join('nvdu', 'config')))


__version__ = "0.0.2"
setup(
    name = "nvdu",
    version = __version__,
    description = "Nvidia Dataset Utilities scripts",
    long_description = "A collection of Python scripts to help working with the DeepLearning projects at Nvidia easier",
    url = "https://gitlab-master.nvidia.com/thangt/nvdu",
    author = "Thang To",
    author_email = "thangt@nvidia.com",
    license = "MIT",
    classifiers = [
        "Intended Audience :: Developers",
        "Topic :: Utilities",
        "License :: MIT",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
    ],
    keywords = "nvdu, nvidia",
    packages=find_packages(),
    package_data = {
        'nvdu_config': all_config_files,
    },
    # data_files=["nvdu/data/ycb/*"],
    install_requires = [
        "numpy",
        "opencv-python",
        "pyrr",
        "PyWavefront==0.2.0",
        "pyglet",
        "fuzzyfinder"
    ],
    extras_require = {
        # "test": [ "pytest" ]
    },
    entry_points = {
        "console_scripts": [
            "nvdu_viz=nvdu.tools.test_nvdu_visualizer:main",
            "nvdu_ycb=nvdu.tools.nvdu_ycb:main",
        ]
    },
    # cmdclass = { 
    #     "test": 
    # }
    scripts=[],
)