# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

import future
import os
from os import path
import numpy as np
import pyrr
from pyrr import Quaternion, Matrix33, Matrix44, Vector4
import urllib.request
import tarfile
import zipfile
import shutil
from enum import IntEnum, unique
import argparse

import nvdu
from nvdu.core.nvdu_data import *

# =============================== Constant variables ===============================
# YCB_DIR_ORIGINAL = "ycb/original"
# YCB_DIR_ALIGNED = "ycb/aligned_m"
# YCB_DIR_ALIGNED_SCALED = "ycb/aligned_cm"
YCB_DATA_URL = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
YCB_URL_POST_FIX = "_google_16k"    # Only support the 16k meshes at the moment

@unique
class YCBModelType(IntEnum):
    Original = 0        # Original model, no modifications
    AlignedOnly = 1     # The model is aligned but doesn't get scaled
    AlignedCm = 2       # The model is aligned and get scaled to centimeter unit

YCB_DIR = [
    path.join('ycb', 'original'),
    path.join('ycb', 'aligned_m'),
    path.join('ycb', 'aligned_cm'),
]

YCB_OBJECT_SETTINGS = [
    path.join('object_settings', '_ycb_original.json'),
    path.join('object_settings', '_ycb_aligned_m.json'),
    path.join('object_settings', '_ycb_aligned_cm.json'),
]

# =============================== Helper functions ===============================
def get_data_root_path():
    nvdu_root_path = nvdu.__path__[0]
    data_root_path = path.join(nvdu_root_path, "data")
    return data_root_path

def get_config_root_path():
    nvdu_root_path = nvdu.__path__[0]
    config_root_path = path.join(nvdu_root_path, "config")
    return config_root_path

def get_ycb_object_settings_path(ycb_model_type: YCBModelType):
    config_root_path = get_config_root_path()
    ycb_object_settings_path = path.join(config_root_path, YCB_OBJECT_SETTINGS[ycb_model_type])
    return ycb_object_settings_path

def get_ycb_root_dir(ycb_model_type: YCBModelType):
    return path.join(get_data_root_path(), YCB_DIR[ycb_model_type])

def get_ycb_object_url(ycb_obj_name):
    ycb_obj_full_url = YCB_DATA_URL + "google/" + ycb_obj_name + YCB_URL_POST_FIX + ".tgz"
    return ycb_obj_full_url

def get_ycb_object_dir(ycb_obj_name, model_type):
    """
    Get the directory path of an ycb object
    ycb_obj_name: name of the YCB object
    model_type: YCBModelType - type of the YCB model
    """
    ycb_obj_dir = path.join(get_ycb_root_dir(model_type), ycb_obj_name)
    return ycb_obj_dir

def get_ycb_model_path(ycb_obj_name, model_type):
    """
    Get the path to the .obj model of an ycb object
    ycb_obj_name: name of the YCB object
    model_type: YCBModelType - type of the YCB model
    """
    ycb_obj_dir = path.join(get_ycb_root_dir(model_type), ycb_obj_name)
    ycb_model_path = path.join(ycb_obj_dir, 'google_16k', 'textured.obj')
    return ycb_model_path

def log_all_path_info():
    ycb_dir_org = get_ycb_root_dir(YCBModelType.Original)
    ycb_dir_aligned_cm = get_ycb_root_dir(YCBModelType.AlignedCm)
    print("YCB original models: {}\nYCB aligned models in centimeter: {}".format(ycb_dir_org, ycb_dir_aligned_cm))

def log_path_info(ycb_obj_name):
    ycb_obj_dir_org = get_ycb_object_dir(ycb_obj_name, YCBModelType.Original)
    ycb_obj_dir_aligned_cm = get_ycb_object_dir(ycb_obj_name, YCBModelType.AlignedCm)
    print("YCB object: '{}'\nOriginal model: {}\nAligned model:{}".format(ycb_obj_name, ycb_obj_dir_org, ycb_obj_dir_aligned_cm))
    if not (path.exists(ycb_obj_dir_org) and path.exists(ycb_obj_dir_aligned_cm)):
        print("WARNING: This YCB object model does not exist, please run 'nvdu_ycb --setup'")

def log_all_object_names():
    print("Supported YCB objects:")
    ycb_object_settings_org_path = get_ycb_object_settings_path(YCBModelType.Original)
    all_ycb_object_settings = DatasetSettings.parse_from_file(ycb_object_settings_org_path)

    for obj_name, obj_settings in all_ycb_object_settings.obj_settings.items():
        # NOTE: The name of object in the object settings have postfix '_16k', we need to remove it
        if obj_name.endswith('_16k'):
            obj_name = obj_name[:-4]

        print("'{}'".format(obj_name))

# =============================== Mesh functions ===============================
def transform_wavefront_file(src_file_path, dest_file_path, transform_matrix):
    dest_dir = path.dirname(dest_file_path)
    if (not path.exists(dest_dir)):
        os.makedirs(dest_dir)

    src_file = open(src_file_path, 'r')
    dest_file = open(dest_file_path, 'w')

    # Keep a separated non-translation matrix to use on the mesh's vertex normal
    non_translation_matrix = Matrix44.from_matrix33(transform_matrix.matrix33)
    # print("non_translation_matrix: {}".format(non_translation_matrix))

    # Parse each lines in the original mesh file
    for line in src_file:
        line_args = line.split()
        if len(line_args):
            type = line_args[0]
            # Transform each vertex
            if (type == 'v'):
                src_vertex = pyrr.Vector4([float(line_args[1]), float(line_args[2]), float(line_args[3]), 1.0])
                dest_vertex = transform_matrix * src_vertex
                dest_file.write("v {:.6f} {:.6f} {:.6f}\n".format(dest_vertex.x, dest_vertex.y, dest_vertex.z))
                continue

            # Transform each vertex normal of the mesh
            elif (type == 'vn'):
                src_vertex = pyrr.Vector4([float(line_args[1]), float(line_args[2]), float(line_args[3]), 1.0])
                dest_vertex = non_translation_matrix * src_vertex
                dest_vertex = pyrr.vector.normalize([dest_vertex.x, dest_vertex.y, dest_vertex.z])
                dest_file.write("vn {:.6f} {:.6f} {:.6f}\n".format(dest_vertex[0], dest_vertex[1], dest_vertex[2]))
                continue
        dest_file.write(line)
    
    src_file.close()
    dest_file.close()

def extract_ycb_model(ycb_obj_name):
    ycb_obj_dir = get_ycb_root_dir(YCBModelType.Original)
    
    # Extract the .tgz to .tar
    ycb_obj_local_file_name = ycb_obj_name + ".tgz"
    ycb_obj_local_path = path.join(ycb_obj_dir, ycb_obj_local_file_name)
    print("Extracting: '{}'".format(ycb_obj_local_path))
    tar = tarfile.open(ycb_obj_local_path, 'r:gz')
    tar.extractall(path=ycb_obj_dir)
    tar.close()

def download_ycb_model(ycb_obj_name, auto_extract=False):
    """ 
    Download an ycb object's 3d models
    ycb_obj_name: string - name of the YCB object to download
    auto_extract: bool - if True then automatically extract the downloaded tgz
    """
    ycb_obj_full_url = get_ycb_object_url(ycb_obj_name)
    ycb_obj_local_file_name = ycb_obj_name + ".tgz"
    ycb_obj_local_dir = get_ycb_root_dir(YCBModelType.Original)
    ycb_obj_local_path = path.join(ycb_obj_local_dir, ycb_obj_local_file_name)

    if (not path.exists(ycb_obj_local_dir)):
        os.makedirs(ycb_obj_local_dir)

    print("Downloading:\nURL: '{}'\nFile:'{}'".format(ycb_obj_full_url, ycb_obj_local_path))
    urllib.request.urlretrieve(ycb_obj_full_url, ycb_obj_local_path)

    if (auto_extract):
        extract_ycb_model(ycb_obj_name)

def align_ycb_model(ycb_obj_name, ycb_obj_settings=None):
    # Use the default object settings file if it's not specified
    if (ycb_obj_settings is None):
        ycb_object_settings_org_path = get_ycb_object_settings_path(YCBModelType.Original)
        all_ycb_object_settings = DatasetSettings.parse_from_file(ycb_object_settings_org_path)
        ycb_obj_settings = all_ycb_object_settings.get_object_settings(ycb_obj_name)
        if (ycb_obj_settings is None):
            print("Can't find settings of object: '{}'".format(ycb_obj_name))
            return

    src_file_path = get_ycb_model_path(ycb_obj_name, YCBModelType.Original)
    dest_file_path = get_ycb_model_path(ycb_obj_name, YCBModelType.AlignedCm)
    # Transform the original models
    print("Align model:\nSource:{}\nTarget:{}".format(src_file_path, dest_file_path))
    # Use the fixed transform matrix to align YCB models
    transform_wavefront_file(src_file_path, dest_file_path, ycb_obj_settings.initial_matrix)
    
    src_dir = path.dirname(src_file_path)
    dest_dir = path.dirname(dest_file_path)
    # Copy the material and texture to the new directory
    shutil.copy(path.join(src_dir, 'textured.mtl'), path.join(dest_dir, 'textured.mtl'))
    shutil.copy(path.join(src_dir, 'texture_map.png'), path.join(dest_dir, 'texture_map.png'))

def setup_all_ycb_models():
    """
    Read the original YCB object settings
    For each object in the list:
        Download the 16k 3d model
        Extract the .tgz file
        Convert the original model into the aligned one
    """
    ycb_object_settings_org_path = get_ycb_object_settings_path(YCBModelType.Original)
    all_ycb_object_settings = DatasetSettings.parse_from_file(ycb_object_settings_org_path)

    for obj_name, obj_settings in all_ycb_object_settings.obj_settings.items():
        # NOTE: The name of object in the object settings have postfix '_16k', we need to remove it
        if obj_name.endswith('_16k'):
            obj_name = obj_name[:-4]
        print("Setting up object: '{}'".format(obj_name))
        download_ycb_model(obj_name, True)
        align_ycb_model(obj_name, obj_settings)
        # break

# =============================== Main ===============================
def main():
    parser = argparse.ArgumentParser(description='NVDU YCB models Support')
    parser.add_argument('ycb_object_name', type=str, nargs='?',
        help="Name of the YCB object to check info", default=None)
    parser.add_argument('-s', '--setup', action='store_true', help="Setup the YCB models for the FAT dataset", default=False)
    parser.add_argument('-l', '--list', action='store_true', help="List all the supported YCB objects", default=False)

    args = parser.parse_args()

    if (args.list):
        log_all_object_names()

    if (args.setup):
        setup_all_ycb_models()
    else:
        if (args.ycb_object_name):
            log_path_info(args.ycb_object_name)
        else:
            # Print the info
            log_all_path_info()


if __name__ == "__main__":
    main()