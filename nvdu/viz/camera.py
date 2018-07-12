# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

from pyrr import Quaternion, Matrix44, Vector3, euler
from pyglet.gl import *
from pyglet.gl.gl import c_float, c_double, c_int, glGetFloatv, GL_MODELVIEW_MATRIX
from pyglet.gl.glu import *
import numpy as np

from .utils3d import *

from nvdu.core import transform3d
from nvdu.core import scene_object
from nvdu.core.camera import *
from .scene_object import *

# A Camera handle the perspective and frustrum of a scene
class Camera(SceneObjectViz):
    # DEFAULT_ZNEAR = 0.000001
    # DEFAULT_ZNEAR = 0.00001
    DEFAULT_ZNEAR = 1
    DEFAULT_ZFAR = 100000.0
    # Horizontal field of view of the camera (in degree)
    DEFAULT_FOVX = 90.0
    DEFAULT_ASPECT_RATIO_XY = 1920.0 / 1080.0
    
    def __init__(self, cam_intrinsic_settings = CameraIntrinsicSettings(), scene_object = None):
        super(Camera, self).__init__(scene_object)

        self.camera_matrix = Matrix44.identity()
        self.projection_matrix = Matrix44.identity()

        self.set_instrinsic_settings(cam_intrinsic_settings)

    def draw(self):
        # glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # TODO: Need to invert the CameraToWorld matrix => WorldToCamera matrix
        # transform_matrix = self.relative_transform.to_matrix()
        # glMultMatrixf(get_opengl_matrixf(transform_matrix))

        # self.build_perspective_projection_matrix()
        
        # print("on_draw - camera_fov: {}".format(self.camera_fov))
        # print("camera - on_draw - projection_matrix: {}".format(self.projection_matrix))

        glMultMatrixf(get_opengl_matrixf(self.projection_matrix))

        # Scale up the scene so the objects really close to the camera doesn't get clipped by the near plane
        # glScalef(100.0, 100.0, 100.0)

    # Set up the camera using its intrinsic parameters:
    # (fx, fy): focal length
    # (cx, cy): optical center
    def set_instrinsic_settings(self, cam_intrinsic_settings):
        self.intrinsic_settings = cam_intrinsic_settings
        if (cam_intrinsic_settings):
            self.projection_matrix = cam_intrinsic_settings.get_projection_matrix()

        # print("set_instrinsic_settings: {} - projection matrix: {}".format(str(self.intrinsic_settings), self.projection_matrix))

    def set_perspective_params(self, fovy, aspect_ratio_xy, znear = DEFAULT_ZNEAR, zfar = DEFAULT_ZFAR):
        self.fovy = np.deg2rad(fovy)
        self.aspect_ratio_xy = aspect_ratio_xy
        self.znear = znear
        self.zfar = zfar

        self.build_perspective_projection_matrix()

    def set_fovx(self, new_fovx):
        # new_fovy = convert_HFOV_to_VFOV(new_fovx, 1.0 / self.aspect_ratio_xy)
        # self.set_fovy(new_fovy)
        # self.build_perspective_projection_matrix()
        new_cam_instrinsics = CameraIntrinsicSettings.from_perspective_fov_horizontal(
            self.intrinsic_settings.res_width, self.intrinsic_settings.res_height, new_fovx)

    def build_perspective_projection_matrix(self):
        zdiff = float(self.znear - self.zfar)

        fovy_tan = np.tan(self.fovy / 2.0)
        # TODO: Handle fovy_tan = 0?
        a = 1.0 / (fovy_tan * self.aspect_ratio_xy)
        b = 1.0 / fovy_tan
        # print('a: {} - b: {}'.format(a, b))
        c = (self.znear + self.zfar) / zdiff
        d = 2 * (self.znear * self.zfar) / zdiff

        self.projection_matrix = Matrix44([
            [a, 0, 0, 0],
            [0, b, 0, 0],
            [0, 0, c, -1.0],
            [0, 0, d, 0]
        ])
        print('build_perspective_projection_matrix: {} - znear: {} - zfar: {} - aspect_ratio_xy: {} - fovy: {}'.format(
            self.projection_matrix, self.znear, self.zfar, self.aspect_ratio_xy, self.fovy))

    def set_viewport_size(self, viewport_size):
        self.viewport_size = viewport_size
