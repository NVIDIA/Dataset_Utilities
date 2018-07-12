# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

from pyrr import Quaternion, Matrix44, Vector3, euler
import numpy as np

from .utils3d import *

from nvdu.core import transform3d
from nvdu.core import scene_object

class SceneObjectViz(object):
    def __init__(self, scene_object):
        self.scene_object = scene_object
        self.render_mode = RenderMode.normal
        self._is_visible = True

    def draw(self):
        if ((self.scene_object is None) or (not self.is_visible())):
            return
        
        # transform_matrix = self.scene_object.relative_transform.to_matrix()
        # glMultMatrixf(get_opengl_matrixf(transform_matrix))      
        # for child_object in self.child_objects:
        #     if (child_object != None):
        #         child_object.draw()

        glPushMatrix()

        world_transform_matrix = self.scene_object.get_world_transform_matrix()
        # print("{} - draw - world_transform_matrix: {}".format(self, world_transform_matrix))
        glMultMatrixf(get_opengl_matrixf(world_transform_matrix))

        self.on_draw()

        glPopMatrix()

    def on_draw(self):
        pass

    def is_visible(self):
        return self._is_visible

    def set_visibility(self, should_visible):
        self._is_visible = should_visible

    def hide(self):
        self._is_visible = False
    
    def show(self):
        self._is_visible = True

    def toggle_visibility(self):
        self._is_visible = not self._is_visible