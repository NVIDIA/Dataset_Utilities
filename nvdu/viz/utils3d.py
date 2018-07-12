# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

from pyrr import Quaternion, Matrix44, Vector3, euler
import numpy as np
from pyglet.gl.gl import *
from ctypes import *

# Get the openGL matrix (GLfloat* 16) from a Matrix44 type
# NOTE: OpenGL use column major while OpenCV use row major
def get_opengl_matrixf(in_mat44):
    return (GLfloat* 16)(
            in_mat44.m11, in_mat44.m12, in_mat44.m13, in_mat44.m14,
            in_mat44.m21, in_mat44.m22, in_mat44.m23, in_mat44.m24,
            in_mat44.m31, in_mat44.m32, in_mat44.m33, in_mat44.m34,
            in_mat44.m41, in_mat44.m42, in_mat44.m43, in_mat44.m44
        )

def convert_HFOV_to_VFOV(hfov, hw_ratio):
    # https://en.wikipedia.org/wiki/Field_of_view_in_video_games
    vfov = 2 * np.arctan(np.tan(np.deg2rad(hfov / 2)) * hw_ratio)

    return np.rad2deg(vfov)

opencv_to_opengl_matrix = Matrix44([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

class RenderMode():
    normal = GL_FILL
    wire_frame = GL_LINE
    point = GL_POINT
