# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

import future
import os
from os import path
import pyglet
from pyglet.window import key

from nvdu.viz.nvdu_visualizer import *
from nvdu.core.nvdu_data import *

class NVDUVizWindow(pyglet.window.Window):
    DEFAULT_EXPORT_DIR = "viz"

    def __init__(self, width: int, height: int, caption: str =''):
        super(NVDUVizWindow, self).__init__(width, height, caption)
        
        self._org_caption = caption
        print('Window created: width = {} - height = {} - title = {}'.format(self.width, self.height, self.caption))
        # print('Window context: {} - config: {}'.format(self.context, self.context.config))

        self.frame_index = 0
        self.visualizer = NVDUVisualizer()

        self.auto_change_frame = False
        self.auto_fps = 0

        self._dataset: NVDUDataset = None
        self.export_dir: str = ""
        self._should_export: bool = False

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, new_dataset: NVDUDataset):
        self._dataset = new_dataset
        frame_count = self._dataset.scan()
        print("Number of frames in the dataset: {}".format(frame_count))

    @property
    def should_export(self) -> bool:
        # Can export if the export directory is valid
        # return not (not self.export_dir)
        return self._should_export and self.export_dir

    @should_export.setter
    def should_export(self, new_export: bool):
        self._should_export = new_export

    def set_caption_postfix(self, postfix: str):
        self.set_caption(self._org_caption + postfix)

    def setup(self):
        glClearColor(0, 0, 0, 1)
        glEnable(GL_DEPTH_TEST)
        # glEnable(GL_DEPTH_CLAMP)
        glFrontFace(GL_CCW)
        # glFrontFace(GL_CW)
        glEnable(GL_BLEND) 
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) 

        self.visualize_current_frame()

    def on_draw(self):
        # Clear the current GL Window
        self.clear()

        self.update_text_color()
        
        if (self.visualizer):
            self.visualizer.draw()
        if (self.should_export):
            self.save_current_viz_frame()

    def on_resize(self, width, height):
        super(NVDUVizWindow, self).on_resize(width, height)
        # set the Viewport
        glViewport(0, 0, width, height)

        self.visualizer.viewport.size = [width, height]

        # new_cam_intrinsic_settings = CameraIntrinsicSettings.from_perspective_fov_horizontal(width, height, CAMERA_FOV_HORIZONTAL)
        # self.visualizer.camera.set_instrinsic_settings(new_cam_intrinsic_settings)

    def set_camera_intrinsic_settings(self, new_cam_intrinsic_settings):
        # print("set_camera_intrinsic_settings: {}".format(new_cam_intrinsic_settings))
        self.visualizer.camera.set_instrinsic_settings(new_cam_intrinsic_settings)

    # Save the current screenshot to a file
    def save_screenshot(self, export_path: str):
        screen_image = pyglet.image.get_buffer_manager().get_color_buffer()
        screen_image.save(export_path)
        print("save_screenshot: {}".format(export_path))

    def save_current_viz_frame(self):
        # TODO: Should ignore? if the visualized frame already exist
        current_frame_name: str = self.dataset.get_frame_name_from_index(self.frame_index)
        # TODO: May need to add config to control the viz postfix
        viz_frame_file_name: str = current_frame_name + "_viz.png"
        export_viz_path: str = path.join(self.export_dir, viz_frame_file_name)
        if not path.exists(self.export_dir):
            os.makedirs(self.export_dir)

        self.save_screenshot(export_viz_path)

    # ========================== DATA PROCESSING ==========================
    def visualize_current_frame(self):
        print('Visualizing frame: {}'.format(self.frame_index))
        self.visualizer.visualize_dataset_frame(self.dataset, self.frame_index)

    def set_frame_index(self, new_frame_index: int):
        total_frame_count: int = self.dataset.frame_count
        if (new_frame_index < 0):
            new_frame_index += total_frame_count
        # TODO: May need to update the total frame count when it's invalid
        if (total_frame_count > 0):
            new_frame_index = new_frame_index % total_frame_count

        if (self.frame_index != new_frame_index):
            self.frame_index = new_frame_index
            self.visualize_current_frame()
    
    # ========================== INPUT CONTROL ==========================
    def on_key_press(self, symbol, modifiers):
        super(NVDUVizWindow, self).on_key_press(symbol, modifiers)
        if (symbol == key.F3):
            self.toggle_cuboid2d_overlay()
        if (symbol == key.F4):
            self.toggle_cuboid3d_overlay()
        elif (symbol == key.F5):
            self.toggle_object_overlay()
        elif (symbol == key.F6):
            self.toggle_pivot()
        elif (symbol == key.F7):
            self.toggle_info_overlay()
        elif (symbol == key.F12):
            self.toggle_export_viz_frame()
        elif (symbol == key._1):
            self.visualizer.set_render_mode(RenderMode.normal)
        elif (symbol == key._2):
            self.visualizer.set_render_mode(RenderMode.wire_frame)
        elif (symbol == key._3):
            self.visualizer.set_render_mode(RenderMode.point)
        elif (symbol == key.SPACE):
            self.toggle_auto_change_frame()

    def on_text_motion(self, motion):
        if motion == key.LEFT:
            self.set_frame_index(self.frame_index - 1)
        elif motion == key.RIGHT:
            self.visualize_next_frame()
        elif motion == key.UP:
            self.set_frame_index(self.frame_index + 100)
        elif motion == key.DOWN:
            self.set_frame_index(self.frame_index - 100)

    def visualize_next_frame(self, dt=0):
        self.set_frame_index(self.frame_index + 1)

    def toggle_export_viz_frame(self):
        self._should_export = not self._should_export
        if (self._should_export and not self.export_dir):
            self.export_dir = NVDUVizWindow.DEFAULT_EXPORT_DIR

        self.update_text_color()
    
    def update_text_color(self):
        # Use different color when we are exporting visualized frame
        if (self.should_export):
            self.set_caption_postfix(" - Exporting ...")
            self.visualizer.set_text_color((255, 0, 0, 255))
        else:
            self.set_caption_postfix("")
            self.visualizer.set_text_color((255, 255, 255, 255))

    def toggle_cuboid2d_overlay(self):
        self.visualizer.toggle_cuboid2d_overlay()

    def toggle_cuboid3d_overlay(self):
        self.visualizer.toggle_cuboid3d_overlay()
    
    def toggle_object_overlay(self):
        self.visualizer.toggle_object_overlay()
        
    def toggle_pivot(self):
        self.visualizer.toggle_pivot_axis()

    def toggle_info_overlay(self):
        self.visualizer.toggle_info_overlay()   

    def toggle_auto_change_frame(self):
        self.set_auto_change_frame(not self.auto_change_frame)
    
    def set_auto_fps(self, new_fps):
        self.auto_fps = new_fps
        if (new_fps <= 0):
            self.set_auto_change_frame(False)
        else:
            self.set_auto_change_frame(True)
            
    def set_auto_change_frame(self, new_bool):
        if (self.auto_change_frame == new_bool):
            return

        self.auto_change_frame = new_bool
        if (self.auto_change_frame):
            print("Start auto changing frame ...")
            wait_duration = 1.0 / self.auto_fps
            pyglet.clock.schedule_interval(self.visualize_next_frame, wait_duration)
        else:
            print("Stop auto changing frame ...")
            pyglet.clock.unschedule(self.visualize_next_frame)
            
