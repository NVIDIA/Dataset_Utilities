# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

#!/usr/bin/env python
try:
    import pyglet
except Exception as ex:
    print("Can't import pyglet module: {}".format(ex))

from pyrr import Quaternion, Matrix44, Vector3, euler
import numpy as np

from ctypes import *
import argparse
from os import path
import sys

import nvdu
from nvdu.core import *
from nvdu.viz.nvdu_visualizer import *
from nvdu.viz.nvdu_viz_window import *

from nvdu.tools.nvdu_ycb import *

CAMERA_FOV_HORIZONTAL = 90

# By default, just visualize the current directory
DEFAULT_data_dir_path = '.'
DEFAULT_output_dir_path = ''
DEFAULT_CAMERA_SETTINGS_FILE_PATH = './config/camera_lov.json'

data_dir_path = DEFAULT_data_dir_path
# mesh_file_name = 'textured.obj'

# ============================= MAIN  =============================
def main():
    DEFAULT_WINDOW_WIDTH = 0
    DEFAULT_WINDOW_HEIGHT = 0

    # Launch Arguments
    parser = argparse.ArgumentParser(description='NVDU Data Visualiser')
    parser.add_argument('dataset_dir', type=str, nargs='?',
        help="Dataset directory. This is where all the images (required) and annotation info (optional) are. Defaulted to the current directory", default=DEFAULT_data_dir_path)
    parser.add_argument('-a', '--data_annot_dir', type=str, help="Directory path - where to find the annotation data. Defaulted to be the same directory as the dataset directory", default="")
    parser.add_argument('-s', '--size', type=int, nargs=2, help="Window's size: [width, height]. If not specified then the window fit the resolution of the camera", default=[DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT])
    parser.add_argument('-o', '--object_settings_path', type=str, help="Object settings file path")
    parser.add_argument('-c', '--camera_settings_path', type=str, help="Camera settings file path", default=None)
    parser.add_argument('-n', '--name_filters', type=str, nargs='*', help="The name filter of each frame. e.g: *.png", default=["*.png"])
    parser.add_argument('--fps', type=float, help="How fast do we want to automatically change frame", default=10)
    parser.add_argument('--auto_change', action='store_true', help="If added this flag, the visualizer will automatically change the frame", default=False)
    parser.add_argument('-e', '--export_dir', type=str, help="Directory path - where to store the visualized images. If this is set, the script will automatically export the visualized image to the export directory", default='')
    parser.add_argument('--auto_export', action='store_true', help="If added this flag, the visualizer will automatically export the visualized frame to image file in the `export_dir` directory", default=False)
    parser.add_argument('--ignore_fixed_transform', action='store_true', help="If added this flag, the visualizer will not use the fixed transform matrix for the 3d model", default=False)
    # parser.add_argument('--gui', type=str, help="Show GUI window")
    
    # subparsers = parser.add_subparsers(help='sub-command help')
    # parser_export = subparsers.add_parser('export', help="Export the visualized frames to image files")
    # parser_export.add_argument('--out_dir', type=str, help="Directory path - where to store the visualized images. If this is set, the script will automatically export the visualized image to the export directory", default='viz')
    # parser_export.add_argument('--movie_name', type=str, help="Name of the movie", default='viz.mp4')
    # parser_export.add_argument('--movie_fps', type=float, help="Framerate of the generated movie", default=30)

    args = parser.parse_args()
    print("args: {}".format(args))

    # TODO: May want to add auto_export as a launch arguments flag
    # auto_export = not (not args.export_dir)
    auto_export = args.auto_export
    
    dataset_dir_path = args.dataset_dir
    data_annot_dir_path = args.data_annot_dir if (args.data_annot_dir) else dataset_dir_path

    name_filters = args.name_filters
    print("name_filters: {}".format(name_filters))
    viz_dataset = NVDUDataset(dataset_dir_path, data_annot_dir_path, name_filters)
    # frame_count = viz_dataset.scan()
    # print("Number of frames in the dataset: {}".format(frame_count))

    # NOTE: Just use the YCB models path for now
    # model_dir_path = args.model_dir
    model_dir_path = get_ycb_root_dir(YCBModelType.Original)

    object_settings_path = args.object_settings_path
    camera_settings_path = args.camera_settings_path
    # If not specified then use the default object and camera settings files from the dataset directory
    if (object_settings_path is None):
        object_settings_path = NVDUDataset.get_default_object_setting_file_path(dataset_dir_path)
    if (camera_settings_path is None):
        camera_settings_path = NVDUDataset.get_default_camera_setting_file_path(dataset_dir_path)

    dataset_settings = DatasetSettings.parse_from_file(object_settings_path, model_dir_path)
    camera_intrinsic_settings = CameraIntrinsicSettings.from_json_file(camera_settings_path)
    # print("camera_intrinsic_settings: {} - {}".format(camera_settings_path, camera_intrinsic_settings))
    
    # By default fit the window size to the resolution of the images
    # NOTE: Right now we don't really support scaling
    width, height = args.size
    if (width <= 0) or (height <= 0):
        width = camera_intrinsic_settings.res_width
        height = camera_intrinsic_settings.res_height

    main_window = NVDUVizWindow(width, height, 'NVDU Data Visualiser')
    main_window.visualizer.dataset_settings = dataset_settings
    main_window.visualizer.visualizer_settings.ignore_initial_matrix = args.ignore_fixed_transform
    main_window.dataset = viz_dataset
    main_window.set_auto_fps(args.fps)
    main_window.should_export = auto_export
    main_window.set_auto_change_frame(args.auto_change)
    main_window.export_dir = args.export_dir
    main_window.setup()

    main_window.set_camera_intrinsic_settings(camera_intrinsic_settings)

    pyglet.app.run()

if __name__ == '__main__':
    main()
