# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

import numpy as np
import json
from .transform3d import *
from .utils3d import *

class CameraIntrinsicSettings(object):
    DEFAULT_ZNEAR = 1
    # DEFAULT_ZFAR = 100000.0
    DEFAULT_ZFAR = DEFAULT_ZNEAR
    
    def __init__(self,
            res_width = 640.0, res_height = 480.0,
            fx = 640.0, fy = 640.0,
            cx = 320.0, cy = 240.0,
            projection_matrix = None):
        self.res_width = res_width
        self.res_height = res_height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.znear = self.DEFAULT_ZNEAR
        self.zfar = self.DEFAULT_ZFAR

        self.projection_matrix = projection_matrix

    @staticmethod
    def from_perspective_fov_horizontal(res_width = 640.0, res_height = 480.0, hfov = 90.0):
        '''
        Create camera intrinsics settings from 3d rendering horizontal field of view
        '''
        cx = res_width / 2.0
        cy = res_height / 2.0
        fx = cx / np.tan(np.deg2rad(hfov) / 2.0)
        fy = fx

        # print("CameraIntrinsicSettings: res_width = {} - res_height = {} - hfov = {} - cx = {} - cy = {} - fx = {} - fy = {}".format(
                # res_width, res_height, hfov, cx, cy, fx, fy))

        new_cam_instrinsics = CameraIntrinsicSettings(res_width, res_height, fx, fy, cx, cy)
        return new_cam_instrinsics

    @staticmethod
    def from_json_object(json_obj):
        intrinsic_settings = json_obj["intrinsic_settings"] if ("intrinsic_settings" in json_obj) else None
        if (intrinsic_settings is None):
            return None

        # print("intrinsic_settings: {}".format(intrinsic_settings))

        try:
            captured_image_size = json_obj['captured_image_size']
            res_width = captured_image_size['width']
            res_height = captured_image_size['height']
        except KeyError:
            print("*** Error ***:  'captured_image_size' is not present in camera settings file.  Using default 640 x 480.")
            res_width = 640
            res_height = 480

        fx = intrinsic_settings['fx'] if ('fx' in intrinsic_settings) else 640.0
        fy = intrinsic_settings['fy'] if ('fy' in intrinsic_settings) else 640.0
        cx = intrinsic_settings['cx'] if ('cx' in intrinsic_settings) else (res_width / 2.0)
        cy = intrinsic_settings['cy'] if ('cy' in intrinsic_settings) else (res_height / 2.0)

        projection_matrix_json = json_obj["cameraProjectionMatrix"] if ("cameraProjectionMatrix" in json_obj) else None
        projection_matrix = None
        if (not projection_matrix_json is None):
            projection_matrix = Matrix44(projection_matrix_json)
            projection_matrix[2, 0] = -projection_matrix[2, 0]
            projection_matrix[2, 1] = -projection_matrix[2, 1]
            projection_matrix[2, 3] = -projection_matrix[2, 3]
            projection_matrix[3, 2] = -projection_matrix[3, 2]

        # print("projection_matrix_json: {}".format(projection_matrix_json))
        # print("projection_matrix: {}".format(projection_matrix))
        
        return CameraIntrinsicSettings(res_width, res_height, fx, fy, cx, cy, projection_matrix)

    @staticmethod
    def from_json_file(json_file_path):
        with open(json_file_path, 'r') as json_file:
            json_obj = json.load(json_file)
            if ('camera_settings' in json_obj):
                viewpoint_list = json_obj['camera_settings']
                # TODO: Need to parse all the viewpoints information, right now we only parse the first viewpoint
                viewpoint_obj = viewpoint_list[0]
                return CameraIntrinsicSettings.from_json_object(viewpoint_obj)
        return None

    def get_intrinsic_matrix(self):
        """
        Get the camera intrinsic matrix as numpy array
        """
        intrinsic_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1.0]
            ], dtype='double')
        return intrinsic_matrix

    def get_projection_matrix(self):
        if (self.projection_matrix is None):
            self.calculate_projection_matrix()

        return self.projection_matrix

    def calculate_projection_matrix(self):
        zdiff = float(self.zfar - self.znear)
        a = (2.0 * self.fx) / float(self.res_width)
        b = (2.0 * self.fy) / float(self.res_height)
        # print('a: {} - b: {}'.format(a, b))
        c = -self.znear / zdiff if (zdiff > 0) else 0
        d = (self.znear * self.zfar) / zdiff if (zdiff > 0) else (-self.znear)
        c1 = 1.0 - (2.0 * self.cx) / self.res_width
        c2 = (2.0 * self.cy) / self.res_height - 1.0

        self.projection_matrix = Matrix44([
            [a, 0, 0, 0],
            [0, b, 0, 0],
            [c1, c2, c, d],
            [0, 0, -1.0, 0]
        ])

    def str():
        return "{}".format(self.get_intrinsic_matrix())