# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

import future
import pyrr
import pywavefront
import numpy as np
from ctypes import *
import json
from os import listdir, path
import pickle
import cv2
import pyglet

from nvdu.core.nvdu_data import *
from .camera import *
from .cuboid import *
from .viewport import *
# from .pivot_axis import *
from .mesh import *
from .background_image import *

# =============================== Helper functions ===============================
# =============================== Data parsing ===============================
# =============================== Dataset Settings ===============================
# =============================== AnnotatedObjectViz ===============================
# Class contain annotation data of each object in the scene
class AnnotatedObjectViz(object):
    def __init__(self, annotated_object_info):
        self.object_info = annotated_object_info
        object_settings = self.object_info.object_settings
        # self.bb2d = BoundingBox()
        self.cuboid2d = Cuboid2dViz(self.object_info.cuboid2d, object_settings.class_color)
        self.cuboid3d = Cuboid3dViz(self.object_info.cuboid3d, object_settings.class_color)
        
        self.pivot_axis = PivotAxis(self.object_info.pivot_axis)
        self.mesh = MeshViz(self.object_info.mesh)
        
        self.is_modified = False

    def update_transform(self):
        # print('update_transform: location: {} - quaternion: {}'.format(self.location, self.quaternion))
        should_show = not (self.location is None) and not (self.quaternion is None)
        if (self.mesh):
            self.mesh.set_visibility(should_show)

        if (self.cuboid3d):
            self.cuboid3d.set_visibility(should_show)

        if (self.pivot_axis):
            self.pivot_axis.set_visibility(should_show)

        self.is_modified = False
        
    def draw(self, visualizer_settings=None):
        if (self.is_modified):
            self.update_transform()

        glPolygonMode(GL_FRONT_AND_BACK, visualizer_settings.render_mode)
        if ((visualizer_settings is None) or visualizer_settings.show_mesh) and self.mesh:
            self.mesh.draw()
        
        if ((visualizer_settings is None) or visualizer_settings.show_cuboid3d) and self.cuboid3d:
            self.cuboid3d.draw()
            # self.cuboid2d.draw()
        
        if ((visualizer_settings is None) or visualizer_settings.show_pivot_axis) and self.pivot_axis:
            self.pivot_axis.draw()

    def update_settings(self, visualizer_settings=None):
        if (visualizer_settings is None):
            return

        have_valid_transform = not (self.object_info.location is None) and not (self.object_info.quaternion is None)

        if (self.is_modified):
            self.update_transform()

        if self.mesh:
            self.mesh.set_visibility(visualizer_settings.show_mesh and have_valid_transform)
            self.mesh.render_mode = visualizer_settings.render_mode
        
        if self.cuboid3d:
            self.cuboid3d.set_visibility(visualizer_settings.show_cuboid3d and have_valid_transform)

        if self.cuboid2d:
            # print('update_settings: self.cuboid2d = {} - visualizer_settings.show_cuboid2d = {}'.format(self.cuboid2d, visualizer_settings.show_cuboid2d))
            self.cuboid2d.set_visibility(visualizer_settings.show_cuboid2d)
        
        if self.pivot_axis:
            self.pivot_axis.set_visibility(visualizer_settings.show_pivot_axis and have_valid_transform)

# =============================== AnnotatedSceneViz ===============================
class AnnotatedSceneViz(object):
    """Class contain annotation data of a scene"""
    def __init__(self, annotated_scene_info):
        self._scene_info = annotated_scene_info
        self.camera_intrinsics = self._scene_info.camera_intrinsics
        self._object_vizs = []
        for check_object in self._scene_info.objects:
            if not (check_object is None):
                new_object_viz = AnnotatedObjectViz(check_object)
                self._object_vizs.append(new_object_viz)

        dataset_settings = self._scene_info.dataset_settings
        if not (dataset_settings is None or dataset_settings.exporter_settings is None):
            img_width, img_height = dataset_settings.exporter_settings.captured_image_size
        # NOTE: Fallback to a default resolution.
        else:
            img_width, img_height = 640, 480
            print("There are no exporter_setting, set image size to 640x480")
        # print("AnnotatedSceneViz - dataset_settings: {} - dataset_settings.exporter_settings: {}".format(
        #     dataset_settings, dataset_settings.exporter_settings))
        # print("AnnotatedSceneViz - img_width: {} - img_height: {}".format(img_width, img_height))
        # print("_scene_info.image_data = {}".format(self._scene_info.image_data))
        # print("object_vizs = {}".format(self._object_vizs))
        if not (self._scene_info.image_data is None):
            self.background_image = BackgroundImage.create_from_numpy_image_data(self._scene_info.image_data, img_width, img_height)
        else:
            self.background_image = None
        
        info_str = self._scene_info.get_scene_info_str()
        # print("Scene info: {}".format(info_str))
        self.info_text = pyglet.text.Label(info_str, 
                font_size=16,
                x=0, y=0,
                # color=(255, 0, 0, 255),
                anchor_x='left', anchor_y='baseline')

    def set_image_data(self, new_image_numpy_data):
        img_width, img_height = self.dataset_settings.exporter_settings.captured_image_size
        print("set_image_data - img_width: {} - img_height: {}".format(img_width, img_height))
        if (self.background_image is None):
            self.background_image = BackgroundImage.create_from_numpy_image_data(new_image_numpy_data, img_width, img_height)
        else:
            self.background_image.load_image_data_from_numpy(new_image_numpy_data)
    
    def draw(self, visualizer_settings=None):
        for obj_viz in self._object_vizs:
            # print('draw object: {}'.format(obj.source_file_path))
            obj_viz.draw(visualizer_settings)

    def set_text_color(self, new_text_color):
        self.info_text.color = new_text_color

    def update_settings(self, visualizer_settings=None):
        if (visualizer_settings):
            for obj_viz in self._object_vizs:
                obj_viz.update_settings(visualizer_settings)

        # TODO: Create a new viz object to handle the text overlay
        # if (self.info_text):
        #     self.info_text.set_visibility()

# =============================== Visualizer ===============================
class VisualizerSettings(object):
    def __init__(self):
        self.render_mode = RenderMode.normal
        self.show_mesh = True
        self.show_pivot_axis = True
        self.show_cuboid3d = True
        self.show_cuboid2d = True
        self.show_bb2d = True
        self.show_info_text = True
        self.ignore_initial_matrix = False

    # TODO: Find a way to use template for all these flags
    def toggle_mesh(self):
        self.show_mesh = not self.show_mesh

    def toggle_pivot_axis(self):
        self.show_pivot_axis = not self.show_pivot_axis

    def toggle_cuboid3d(self):
        self.show_cuboid3d = not self.show_cuboid3d

    def toggle_cuboid2d(self):
        self.show_cuboid2d = not self.show_cuboid2d
        
    def toggle_bb2d(self):
        self.show_bb2d = not self.show_bb2d

    def toggle_info_overlay(self):
        self.show_info_text = not self.show_info_text

class NVDUVisualizer():
    def __init__(self):
        self.render_mode = RenderMode.normal
        self.camera = Camera()

        self.dataset_settings = None
        self.visualizer_settings = VisualizerSettings()
        self.annotated_scene = None
        self.scene_viz = None

        self.viewport = Viewport(None)
        self.viewport.size = [512, 512]

    def draw(self):
        if (self.annotated_scene is None) or (self.scene_viz is None):
            return

        self.viewport.clear()

        # TODO: Should let the AnnotatedSceneViz handle all these draw logic
        self.viewport.scene_bg.add_object(self.scene_viz.background_image)

        self.viewport.scene3d.camera = self.camera
        if (not self.scene_viz.camera_intrinsics is None):
            self.viewport.scene3d.camera.set_instrinsic_settings(self.scene_viz.camera_intrinsics)
        # self.viewport.scene3d.camera.set_fovx(self.scene_viz.camera_fovx)

        #TODO: Move this code to a separated function
        # mesh_paths = []
        # for obj_viz in self.scene_viz._object_vizs:
        #     mesh_paths.append(obj_viz.mesh.mesh_obj.source_file_path)
        # GlobalModelManager.load_model_list(mesh_paths)
        # print("NVDUVisualizer - draw - mesh_paths: {}".format(mesh_paths))
        
        for obj in self.scene_viz._object_vizs:
            if (obj.mesh):
                obj.mesh.ignore_initial_matrix = self.visualizer_settings.ignore_initial_matrix
                self.viewport.scene3d.add_object(obj.mesh)
            self.viewport.scene3d.add_object(obj.cuboid3d)
            self.viewport.scene3d.add_object(obj.pivot_axis)
            self.viewport.scene_overlay.add_object(obj.cuboid2d)
        
        if (self.visualizer_settings.show_info_text):
            self.scene_viz.info_text.draw()

        self.scene_viz.update_settings(self.visualizer_settings)

        self.viewport.draw()

    # ========================== CONTROL ==========================
    def toggle_cuboid2d_overlay(self):
        self.visualizer_settings.toggle_cuboid2d()

    def toggle_cuboid3d_overlay(self):
        self.visualizer_settings.toggle_cuboid3d()
    
    def toggle_object_overlay(self):
        self.visualizer_settings.toggle_mesh()

    def toggle_pivot_axis(self):
        self.visualizer_settings.toggle_pivot_axis()

    def toggle_info_overlay(self):
        self.visualizer_settings.toggle_info_overlay()

    def set_render_mode(self, new_render_mode):
        self.visualizer_settings.render_mode = new_render_mode

    def set_text_color(self, new_text_color):
        self.scene_viz.set_text_color(new_text_color)

    def visualize_dataset_frame(self, in_dataset: NVDUDataset, in_frame_index: int = 0):
        frame_image_file_path, frame_data_file_path = in_dataset.get_frame_file_path_from_index(in_frame_index)
        if not path.exists(frame_image_file_path):
            print("Can't find image file for frame: {} - {}".format(in_frame_index, frame_image_file_path))
            return
        if not path.exists(frame_data_file_path):
            print("Can't find annotation file for frame: {} - {}".format(in_frame_index, frame_data_file_path))
            return

        print("visualize_dataset_frame: frame_image_file_path: {} - frame_data_file_path: {}".format(
            frame_image_file_path, frame_data_file_path))

        frame_scene_data = AnnotatedSceneInfo.create_from_file(self.dataset_settings,
                frame_data_file_path, frame_image_file_path)
        self.visualize_scene(frame_scene_data)

    def set_scene_data(self, new_scene_data):
        self.annotated_scene = new_scene_data
        self.scene_viz = AnnotatedSceneViz(self.annotated_scene)

    def visualize_scene(self, annotated_scene):
        self.set_scene_data(annotated_scene)
        self.draw()
