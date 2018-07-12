# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

import numpy as np
from pyrr import Quaternion, Matrix44, Vector3, euler
from pyglet.gl import *
from pyglet.gl.gl import *
from pyglet.gl.glu import *
from ctypes import *

from .scene_object import *

class PivotAxis(SceneObjectViz):
    # Create a pivot axis object with 3 axes each can have different length
    def __init__(self, in_pivot_axis_obj, in_line_width = 5.0):
        super(PivotAxis, self).__init__(in_pivot_axis_obj)
        self.pivot_obj = in_pivot_axis_obj
        self.origin_loc = self.pivot_obj.origin_loc
        self.line_width = in_line_width
        self.render_stipple_line = False
        # List of color for each vertices of the box
        self.colors = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ]

    def on_draw(self):
        super(PivotAxis, self).on_draw()
        
        glEnable(GL_LINE_SMOOTH)
        if (self.render_stipple_line):
            glEnable(GL_LINE_STIPPLE)
            line_pattern =  0x00ff
            line_tripple_factor = 1
            glLineStipple(line_tripple_factor, line_pattern);

        glLineWidth(self.line_width)
        glBegin(GL_LINES)
        # X axis
        glColor3ub(self.colors[0][0], self.colors[0][1], self.colors[0][2])
        glVertex3f(self.origin_loc[0], self.origin_loc[1], self.origin_loc[2])
        glVertex3f(self.pivot_obj.x_axis[0], self.pivot_obj.x_axis[1], self.pivot_obj.x_axis[2])

        # Y axis
        glColor3ub(self.colors[1][0], self.colors[1][1], self.colors[1][2])
        glVertex3f(self.origin_loc[0], self.origin_loc[1], self.origin_loc[2])
        glVertex3f(self.pivot_obj.y_axis[0], self.pivot_obj.y_axis[1], self.pivot_obj.y_axis[2])

        # Z axis
        glColor3ub(self.colors[2][0], self.colors[2][1], self.colors[2][2])
        glVertex3f(self.origin_loc[0], self.origin_loc[1], self.origin_loc[2])
        glVertex3f(self.pivot_obj.z_axis[0], self.pivot_obj.z_axis[1], self.pivot_obj.z_axis[2])
        glEnd()

        glColor3ub(255, 255, 255)
        glLineWidth(1.0)

        if (self.render_stipple_line):
            glDisable(GL_LINE_STIPPLE)
        glDisable(GL_LINE_SMOOTH)