# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

from setuptools import Command, find_packages, setup
import os
from os import path
import glob

__version_info__ = (1, 0, 0, 0)

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

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
print("all_config_files: {}".format(all_config_files))

__version__ = '.'.join(map(str, __version_info__))

setup(
    name = "nvdu",
    version = __version__,
    description = "Nvidia Dataset Utilities",
    long_description = read('readme.md'),
    long_description_content_type = 'text/markdown',
    url = "https://github.com/NVIDIA/Dataset_Utilities",
    author = "NVIDIA Corporation",
    author_email = "info@nvidia.com",
    maintainer = "Thang To",
    maintainer_email = "thangt@nvidia.com",
    license = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0. https://creativecommons.org/licenses/by-nc-sa/4.0/",
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers = [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Topic :: Utilities",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords = "nvdu, nvidia",
    packages=find_packages(),
    data_files=[('nvdu_config', all_config_files)],
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
    scripts=[],
)