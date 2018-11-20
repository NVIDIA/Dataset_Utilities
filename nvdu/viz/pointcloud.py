# Copyright (c) 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

from pyrr import Quaternion, Matrix44, Vector3, euler
import numpy as np
from pyglet.gl import *
from ctypes import *

from .scene_object import *
from nvdu.core.cuboid import *

# ========================= PointCloud2d =========================
class PointCloud2d(SceneObjectVizBase):
    # Create a box with a certain size
    def __init__(self, point2d_list=[], in_color=None):
        super(PointCloud2d, self).__init__(point2d_list)
        self.vertices = point2d_list
        self.color = in_color
        self.generate_vertexes_buffer()

    def generate_vertexes_buffer(self):
        self.vertices_gl = []
        vertex_count = len(self.vertices)
        for i in range(0, vertex_count):
            vertex = self.vertices[i]
            if (not vertex is None):
                self.vertices_gl.append(vertex[0])
                self.vertices_gl.append(vertex[1])
        
        self.indices_point_gl = []
        for i in range (len(self.vertices)):
            if (not self.vertices[i] is None):
                self.indices_point_gl.append(i)

        self.vertex_gl_array = (GLfloat* len(self.vertices_gl))(*self.vertices_gl)
        self.indices_point_gl_array = (GLubyte* len(self.indices_point_gl))(*self.indices_point_gl)

        # print("PointCloud2d: {}".format(self.vertices_gl))

    def on_draw(self):
        super(PointCloud2d, self).on_draw()

        # print("Drawing pointcloud: {}".format(self.vertices_gl))

        glEnableClientState(GL_VERTEX_ARRAY)
        # glEnableClientState(GL_COLOR_ARRAY)

        glPolygonMode(GL_FRONT_AND_BACK, RenderMode.normal)

        glVertexPointer(2, GL_FLOAT, 0, self.vertex_gl_array)

        # glColorPointer(4, GL_UNSIGNED_BYTE, 0, self.vertex_color_gl_array)
        glPointSize(10.0)
        glDrawElements(GL_POINTS, len(self.indices_point_gl_array), GL_UNSIGNED_BYTE, self.indices_point_gl_array)
        glPointSize(1.0)
      
        # Deactivate vertex arrays after drawing
        glDisableClientState(GL_VERTEX_ARRAY)
        # glDisableClientState(GL_COLOR_ARRAY)
