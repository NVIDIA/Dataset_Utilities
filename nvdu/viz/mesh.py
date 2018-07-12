# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

import os
from os import path
import asyncio
from pyrr import Quaternion, Matrix44, Vector3, euler
import numpy as np
from pyglet.gl import *
from pyglet.gl.gl import *
from pyglet.gl.glu import *
from ctypes import *
import pywavefront
import pywavefront.visualization

from nvdu.core.mesh import *
from .utils3d import *
from .scene_object import *
from .pivot_axis import *

class Model3dManager(object):
    def __init__(self):
        self.model_map = {}

    def get_model(self, model_path, auto_load = True):
        if (model_path in self.model_map):
            return self.model_map[model_path]

        if (auto_load):
            self.load_model(model_path)

        return None

    def load_model(self, model_path):
        new_model = self.load_model_from_file(model_path)
        self.model_map[model_path] = new_model
        return new_model
    
    # def load_model_list(self, model_paths):
    #     # print("load_model_list: {}".format(model_paths))
    #     tasks = []
    #     for check_path in model_paths:
    #         if not (check_path in self.model_map):
    #             tasks.append(asyncio.ensure_future(self.load_model(check_path)))
    #     if (len(tasks) > 0):
    #         # print("Load all the models: {}".format(model_paths))
    #         self.loop.run_until_complete(asyncio.wait(tasks))
        
    def load_model_from_file(self, model_file_path):
        if (path.exists(model_file_path)):
            print("Model3dManager::load_model_from_file: {}".format(model_file_path))
            return pywavefront.Wavefront(model_file_path)
        else:
            print("Model3dManager::load_model_from_file - can NOT find 3d model: {}".format(model_file_path))
        return None

GlobalModelManager = Model3dManager()

class MeshViz(SceneObjectViz):
    def __init__(self, mesh_obj):
        super(MeshViz, self).__init__(mesh_obj)

        self.mesh_obj = mesh_obj
        self.mesh_model = None

        # pivot_size = [10, 10, 10]
        # self.pivot_axis = PivotAxis(pivot_size)
        self.pivot_axis = None
        self.ignore_initial_matrix = False

    def on_draw(self):
        super(MeshViz, self).on_draw()

        if (self.mesh_model is None):
            self.mesh_model = GlobalModelManager.get_model(self.mesh_obj.source_file_path)

        if (self.mesh_model):
            if (self.pivot_axis):
                self.pivot_axis.draw()
            
            glPolygonMode(GL_FRONT_AND_BACK, self.render_mode)

            if (not self.ignore_initial_matrix):
                mesh_initial_matrix = self.mesh_obj.get_initial_matrix()
                # print("mesh_initial_matrix: {}".format(mesh_initial_matrix))
                glMultMatrixf(get_opengl_matrixf(mesh_initial_matrix))

            # TODO: Need to get the color from the object settings
            glColor4f(1.0, 1.0, 0.0, 0.5)
            self.mesh_model.draw()
            glColor4f(1.0, 1.0, 1.0, 1.0)
            glPolygonMode(GL_FRONT_AND_BACK, RenderMode.normal)
