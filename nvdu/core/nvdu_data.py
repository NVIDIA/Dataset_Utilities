# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

import future
from os import listdir, path
from ctypes import *
import json
# from typing import List, Dict, Tuple
import fnmatch

import numpy as np
import cv2
import pyrr
import copy
from fuzzyfinder import fuzzyfinder

from .utils3d import *
from .transform3d import *
from .cuboid import *
from .pivot_axis import *
from .mesh import *
from .camera import *

FrameDataExt = ".json"
FrameImageExt = ".png"
FrameImageExt_Depth = ".depth"
FrameImageExt_Pls = ".pls"
FrameImageExt_Pls_no = ".pls_no"

FrameAspectDict = {
    'main': {
        'name': 'Main',
        'ext': ''
    },
    'depth': {
        'name': 'Depth',
        'ext': FrameImageExt_Depth
    },
    'pls': {
        'name': 'Pixel Level Segmentation',
        'ext': FrameImageExt_Pls
    },
    'pls_no': {
        'name': 'Pixel Level Segmentation No Occlusion',
        'ext': FrameImageExt_Pls_no
    }
    }

# =============================== Helper functions ===============================
DEFAULT_MESH_NAME_FORMAT = "{}/google_16k/textured.obj"
def get_mesh_file_path(mesh_folder_path, mesh_name, mesh_name_format=DEFAULT_MESH_NAME_FORMAT):
    # NOTE: The name of object in the object settings have postfix '_16k', we need to remove it
    if mesh_name.endswith('_16k'):
        mesh_name = mesh_name[:-4]
    return path.join(mesh_folder_path, mesh_name_format.format(mesh_name))

DEFAULT_FRAME_NAME_FORMAT = "{0:06d}"
def get_frame_name(frame_index, frame_name_format=DEFAULT_FRAME_NAME_FORMAT):
    """Get the number part of a frame's name"""
    frame_number_name = frame_name_format.format(frame_index)
    return frame_number_name

def is_frame_exists(data_dir_path, frame_index, frame_name_format=DEFAULT_FRAME_NAME_FORMAT):
    frame_data_file_name = get_frame_name(frame_index, frame_name_format) + FrameImageExt
    frame_data_file_path = path.join(data_dir_path, frame_data_file_name)
    # print("Checking frame {} at: {}".format(frame_index, frame_data_file_path))
    return path.exists(frame_data_file_path)

def is_frame_data_exists(data_dir_path, frame_index, frame_name_format=DEFAULT_FRAME_NAME_FORMAT):
    frame_data_file_name = get_frame_name(frame_index, frame_name_format) + FrameDataExt
    frame_data_file_path = path.join(data_dir_path, frame_data_file_name)
    # print("Checking frame {} at: {}".format(frame_index, frame_data_file_path))
    return path.exists(frame_data_file_path)

def get_dataset_setting_file_path(data_dir_path):
    return path.join(data_dir_path, '_settings.json')

def get_dataset_object_setting_file_path(data_dir_path):
    return path.join(data_dir_path, '_object_settings.json')

def get_frame_data_path(data_dir_path, frame_index, frame_name_format=DEFAULT_FRAME_NAME_FORMAT):
    frame_data_file_name = get_frame_name(frame_index, frame_name_format) + FrameDataExt
    frame_data_file_path = path.join(data_dir_path, frame_data_file_name)
    return frame_data_file_path

def get_frame_image_path(data_dir_path, frame_index, frame_name_format=DEFAULT_FRAME_NAME_FORMAT, aspect_id='main'):
    frame_name = get_frame_name(frame_index, frame_name_format)
    frame_aspect_data = FrameAspectDict[aspect_id]
    aspect_ext = frame_aspect_data['ext']
    frame_aspect_image_file = path.join(data_dir_path, frame_name) + aspect_ext + FrameImageExt
    return frame_aspect_image_file

# Get the total number of frames in a data set
# data_dir_path: path to the dataset's directory
def get_number_of_frames(data_dir_path, frame_name_format=DEFAULT_FRAME_NAME_FORMAT):
    number_of_frame = 0
    min_index = 1
    # NOTE: We only count till 1000000 frames for now
    max_index = 1000000
    while (min_index <= max_index):
        check_index = int((min_index + max_index) / 2)
        frame_exists = is_frame_exists(data_dir_path, check_index, frame_name_format)
        if frame_exists:
            # NOTE: The frame start from 0 => the number of frame will be the last index + 1
            number_of_frame = check_index + 1
            min_index = check_index + 1
        else:
            max_index = check_index - 1

    return number_of_frame

class NVDUDataset(object):
    DEFAULT_IMAGE_NAME_FILTERS = ["*.png"]
    
    def __init__(self, in_dataset_dir: str, 
            in_annotation_dir: str = None,
            in_img_name_filters = None):
        self._dataset_dir: str = in_dataset_dir
        self._annotation_dr: str = in_annotation_dir if (not in_annotation_dir is None) else self._dataset_dir
        self._img_name_filters = in_img_name_filters if (not in_img_name_filters is None) else NVDUDataset.DEFAULT_IMAGE_NAME_FILTERS
        
        self._frame_names = []
        self._frame_count: int = 0

    @property
    def frame_names(self):
        return self._frame_names

    @property
    def frame_count(self):
        return self._frame_count

    @property
    def camera_setting_file_path(self):
        return NVDUDataset.get_camera_setting_file_path(self._dataset_dir)

    @property
    def object_setting_file_path(self):
        return NVDUDataset.get_object_setting_file_path(self._dataset_dir)
    
    # Scane the dataset and return how many frames are in it
    def scan(self):
        self._frame_names = []
        if not path.exists(self._dataset_dir):
            return 0
        
        print("scan - _dataset_dir: {} - _img_name_filters: {}".format(self._dataset_dir, self._img_name_filters))
        for file_name in listdir(self._dataset_dir):
            check_file_path = path.join(self._dataset_dir, file_name)
            if path.isfile(check_file_path):
                is_name_match_filters = any(fnmatch.fnmatch(file_name, check_filter) for check_filter in self._img_name_filters)
                # is_name_match_filters = False
                # for check_filter in self._img_name_filters:
                #     if fnmatch.fnmatch(file_name, check_filter):
                #         is_name_match_filters = True
                #         print("check_filter: {} - file_name: {} - is good".format(check_filter, file_name))
                #         break
                        
                if (is_name_match_filters):
                    # NOTE: Consider frame name as the file name without its extension
                    frame_name = path.splitext(file_name)[0]
                    # NOTE: Consider frame name as the first part of the string before the '.'
                    # frame_name = file_name.split(".")[0]
                    # Check if it have annotation data or not
                    check_annotation_file_path = self.get_annotation_file_path_of_frame(frame_name)
                    # print("file_name: {} - check_file_path: {} - frame_name: {} - check_annotation_file_path: {}".format(file_name, check_file_path, frame_name, check_annotation_file_path))
                    if path.exists(check_annotation_file_path):
                        self._frame_names.append(frame_name)
        
        self._frame_names = sorted(self._frame_names)
        self._frame_count = len(self._frame_names)
        # print("_frame_names: {}".format(self._frame_names))
        return self._frame_count

    def get_image_file_path_of_frame(self, in_frame_name):
        return path.join(self._dataset_dir, in_frame_name + FrameImageExt)
    
    def get_annotation_file_path_of_frame(self, in_frame_name):
        return path.join(self._annotation_dr, in_frame_name + FrameDataExt)

    def get_frame_name_from_index(self, in_frame_index):
        return self._frame_names[in_frame_index] if ((in_frame_index >= 0) and (in_frame_index < len(self._frame_names))) else ""

    # Return the frame's image file path and annotation file path when know its index
    def get_frame_file_path_from_index(self, in_frame_index):
        frame_name = self.get_frame_name_from_index(in_frame_index)
        return self.get_frame_file_path_from_name(frame_name)
    
    # Return the frame's image file path and annotation file path when know its name
    def get_frame_file_path_from_name(self, in_frame_name):
        if (not in_frame_name):
            return ("", "")

        image_file_path = self.get_image_file_path_of_frame(in_frame_name)
        annotation_file_path = self.get_annotation_file_path_of_frame(in_frame_name)
        return (image_file_path, annotation_file_path)

    @staticmethod
    def get_default_camera_setting_file_path(data_dir_path):
        return path.join(data_dir_path, '_camera_settings.json')

    @staticmethod
    def get_default_object_setting_file_path(data_dir_path):
        return path.join(data_dir_path, '_object_settings.json')


# =============================== Dataset Settings ===============================
#  Class represent the settings data for each exported object
class ExportedObjectSettings(object):
    def __init__(self, name = '', mesh_file_path = '', initial_matrix = None,
            cuboid_dimension = Vector3([0, 0, 0]), cuboid_center = Vector3([0, 0, 0]),
            coord_system = CoordinateSystem(), obj_class_id = 0):
        self.name = ''
        self.mesh_file_path = mesh_file_path
        self.initial_matrix = initial_matrix

        self.class_id = obj_class_id
        # TODO: Add option to customize the color for different object classes
        self.class_color = [0, 0, 0, 255]
        self.class_color[0] = int((self.class_id >> 5) / 7.0 * 255)
        self.class_color[1] = int(((self.class_id >> 2) & 7) / 7.0 * 255)
        self.class_color[2] = int((self.class_id & 3) / 3.0 * 255)

        self.cuboid_dimension = cuboid_dimension
        self.cuboid_center_local = cuboid_center
        self.coord_system = coord_system
        self.cuboid3d = Cuboid3d(self.cuboid_dimension, self.cuboid_center_local, self.coord_system)

        self.mesh_model = None
        self.pivot_axis = PivotAxis(np.array(self.cuboid_dimension))

    def __str__(self):
        return "({} - {} - {})".format(self.name, self.mesh_file_path, self.initial_matrix)

class ExporterSettings(object):
    def __init__(self):
        self.captured_image_size = [1280, 720]

    @classmethod
    def parse_from_json_data(cls, json_data):
        parsed_exporter_settings = ExporterSettings()
        parsed_exporter_settings.captured_image_size = json_data['capturedImageSize']
        print("parsed_exporter_settings.captured_image_size: {}".format(parsed_exporter_settings.captured_image_size))
        
        return parsed_exporter_settings

class DatasetSettings():
    def __init__(self, mesh_dir_path=''):
        self.mesh_dir_path = mesh_dir_path
        self.obj_settings = {}
        self.exporter_settings = ExporterSettings()
        self.coord_system = CoordinateSystem()

    @classmethod
    def parse_from_json_data(cls, json_data, mesh_dir_path=''):
        parsed_settings = DatasetSettings(mesh_dir_path)
        
        coord_system = None
        parsed_settings.coord_system = coord_system
        
        for check_obj in json_data['exported_objects']:
            obj_class = check_obj['class']
            obj_mesh_file_path = get_mesh_file_path(parsed_settings.mesh_dir_path, obj_class)
            obj_initial_matrix = Matrix44(check_obj['fixed_model_transform'])
            obj_cuboid_dimension = check_obj['cuboid_dimensions'] if ('cuboid_dimensions' in check_obj) else Vector3([0, 0, 0])
            obj_cuboid_center = check_obj['cuboid_center_local'] if ('cuboid_center_local' in check_obj) else Vector3([0, 0, 0])
            obj_class_id = int(check_obj['segmentation_class_id']) if ('segmentation_class_id' in check_obj) else 0

            new_obj_info = ExportedObjectSettings(obj_class, obj_mesh_file_path, 
                obj_initial_matrix, obj_cuboid_dimension, obj_cuboid_center, coord_system, obj_class_id)

            # print('scan_settings_data: {}'.format(new_obj_info))
            parsed_settings.obj_settings[obj_class] = new_obj_info
        
        return parsed_settings

    @classmethod
    def parse_from_file(cls, setting_file_path, mesh_dir_path=''):
        parsed_settings = None

        if (path.exists(setting_file_path)):
            json_data = json.load(open(setting_file_path))
            parsed_settings = cls.parse_from_json_data(json_data, mesh_dir_path)
            # print('parse_from_file: setting_file_path: {} - mesh_dir_path: {} - parsed_settings: {}'.format(
            #     setting_file_path, mesh_dir_path, parsed_settings))

        return parsed_settings

    @classmethod
    def parse_from_dataset(cls, dataset_dir_path, mesh_dir_path=''):
        setting_file_path = get_dataset_object_setting_file_path(dataset_dir_path)        
        # print('parse_from_dataset: dataset_dir_path: {} - mesh_dir_path: {}'.format(dataset_dir_path, mesh_dir_path))
        return cls.parse_from_file(setting_file_path, mesh_dir_path)
        
    # Get the settings info for a specified object class
    def get_object_settings(self, object_class):
        if (object_class in self.obj_settings):
            return self.obj_settings[object_class]
        # TODO: If there are no match object_class name then try to find the closest match using fuzzy find
        all_object_classes = list(self.obj_settings.keys())
        fuzzy_object_classes = list(fuzzyfinder(object_class, all_object_classes))
        if (len(fuzzy_object_classes) > 0):
            fuzzy_object_class = fuzzy_object_classes[0]
            # print("fuzzy_object_classes: {} - fuzzy_object_class: {}".format(fuzzy_object_classes, fuzzy_object_class))
            return self.obj_settings[fuzzy_object_class]

        return None

# =============================== AnnotatedObjectInfo ===============================
# Class contain annotation data of each object in the scene
class AnnotatedObjectInfo(SceneObject):
    def __init__(self, dataset_settings, obj_class = '', name = ''):
        super(AnnotatedObjectInfo, self).__init__()

        self.name = name
        self.obj_class = obj_class
        self.object_settings = dataset_settings.get_object_settings(obj_class) if not (dataset_settings is None) else None
        
        self.location = pyrr.Vector3()
        self.cuboid_center = None
        self.quaternion = pyrr.Quaternion([0.0, 0.0, 0.0, 1.0])
        # self.bb2d = BoundingBox()
        self.cuboid2d = None

        if not (self.object_settings is None):
            self.dimension = self.object_settings.cuboid_dimension
            self.cuboid3d = copy.deepcopy(self.object_settings.cuboid3d) if (not self.object_settings is None) else None
            self.mesh = Mesh(self.object_settings.mesh_file_path)
            self.mesh.set_initial_matrix(self.object_settings.initial_matrix)
            self.pivot_axis = self.object_settings.pivot_axis
        else:
            self.dimension = None
            self.cuboid3d = None
            self.mesh = None
            self.pivot_axis = None
        
        self.is_modified = False
        self.relative_transform = transform3d()

    # Parse and create an annotated object from a json object
    @classmethod
    def parse_from_json_object(self, dataset_settings, json_obj):
        try:
            obj_class = json_obj['class']
            # print('parse_from_json_object: dataset_settings: {} - name: {} - class: {}'.format(
            #     dataset_settings, obj_name, obj_class))
        except KeyError:
            print("*** Error ***:  'class' is not present in annotation file.  Using default '002_master_chef_can_16k'.")
            obj_class = '002_master_chef_can_16k'

        parsed_object = AnnotatedObjectInfo(dataset_settings, obj_class)
        if ('location' in json_obj):
            parsed_object.location = json_obj['location']
        if ('quaternion_xyzw' in json_obj):
            parsed_object.quaternion = Quaternion(json_obj['quaternion_xyzw'])
        if ('cuboid_centroid' in json_obj):
            parsed_object.cuboid_center = json_obj['cuboid_centroid']

        # TODO: Parse bounding box 2d
        # json_obj['bounding_rectangle_imagespace']

        # Parse the cuboid in image space
        if ('projected_cuboid' in json_obj):
            img_width, img_height = dataset_settings.exporter_settings.captured_image_size
            cuboid2d_vertices_json_data = json_obj['projected_cuboid']
            # Convert the fraction coordinate to absolute coordinate
            # cuboid2d_vertices = list([img_width * vertex['x'], img_height * vertex['y']] for vertex in cuboid2d_vertices_json_data)
            cuboid2d_vertices = cuboid2d_vertices_json_data
            # print('img_width: {} - img_height: {}'.format(img_width, img_height))
            # print('cuboid2d_vertices: {}'.format(cuboid2d_vertices))
            parsed_object.cuboid2d = Cuboid2d(cuboid2d_vertices)

        parsed_object.update_transform()

        return parsed_object

    def set_transform(self, new_location, new_quaternion):
        self.location = new_location
        self.quaternion = new_quaternion
        self.is_modified = True

    def set_location(self, new_location):
        self.location = new_location
        self.is_modified = True

    def set_quaternion(self, new_quaternion):
        self.quaternion = new_quaternion
        self.is_modified = True

    def update_transform(self):
        self.set_relative_transform(self.location, self.quaternion)

        # print('update_transform: location: {} - quaternion: {}'.format(self.location, self.quaternion))
        should_show = not (self.location is None) and not (self.quaternion is None)
        if (not self.mesh is None) and should_show:
            self.mesh.set_relative_transform(self.location, self.quaternion)

        if (not self.cuboid3d is None) and should_show:
            cuboid_location = self.cuboid_center if (not self.cuboid_center is None) else self.location
            # self.cuboid3d.set_relative_transform(self.location, self.quaternion)
            self.cuboid3d.set_relative_transform(cuboid_location, self.quaternion)

        if (not self.pivot_axis is None) and should_show:
            self.pivot_axis.set_relative_transform(self.location, self.quaternion)

        self.is_modified = False

# =============================== AnnotatedSceneInfo ===============================
class AnnotatedSceneInfo(object):
    """Annotation data of a scene"""
    def __init__(self, dataset_settings):
        self.source_file_path = ""
        self.dataset_settings = dataset_settings
        self.objects = []
        # Numpy array of pixel data
        self.image_data = None
        self.camera_intrinsics = None

    def get_object_info(self, object_class_name):
        found_objects = []
        for check_object in self.objects:
            if not (check_object is None) and (check_object.obj_class == object_class_name):
                found_objects.append(check_object)
        return found_objects

    def set_image_data(self, new_image_numpy_data):
        self.image_data = new_image_numpy_data

    def get_scene_info_str(self):
        info_str = path.splitext(path.basename(self.source_file_path))[0]
        return info_str

    # Parse and create an annotated scene from a json object
    @classmethod
    def create_from_json_data(cls, dataset_settings, frame_json_data, image_data):
        parsed_scene = AnnotatedSceneInfo(dataset_settings)

        # self.camera_intrinsics = dataset_settings.camera_intrinsics
        if ('view_data' in frame_json_data):
            view_data = frame_json_data['view_data']
            # TODO: Need to handle the randomized FOV in different frame
            # if ((not view_data is None) and ('fov' in view_data)):
            #     camera_fovx = frame_json_data['view_data']['fov']
            #     parsed_scene.camera_fovx = camera_fovx
            #     parsed_scene.camera_intrinsics = CameraIntrinsicSettings.from_perspective_fov_horizontal(hfov=camera_fovx)
        
        parsed_scene.objects = []
        try:
            objects_data = frame_json_data['objects']
            for check_obj_info in objects_data:
                new_obj = AnnotatedObjectInfo.parse_from_json_object(dataset_settings, check_obj_info)
                parsed_scene.objects.append(new_obj)
        except KeyError:
            print("*** Error ***:  'objects' is not present in annotation file.  No annotations will be displayed.")

            # !TEST
            # obj_transform = new_obj.get_world_transform_matrix()
            # projected_cuboid = new_obj.cuboid3d.get_projected_cuboid2d(obj_transform, self.camera_intrinsics.get_intrinsic_matrix())

        parsed_scene.image_data = image_data

        return parsed_scene

    # Parse and create an annotated scene from a json object
    @classmethod
    def create_from_file(cls, dataset_settings, frame_file_path, image_file_path=""):
        json_data = json.load(open(frame_file_path))
        if (path.exists(image_file_path)):
            image_data = np.array(cv2.imread(image_file_path))
            image_data = image_data[:,:,::-1] # Reorder color channels to be RGB
        else:
            image_data = None

        new_scene_info = cls.create_from_json_data(dataset_settings, json_data, image_data)
        new_scene_info.source_file_path = frame_file_path
        return new_scene_info
