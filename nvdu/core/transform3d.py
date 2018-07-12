# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

from pyrr import Quaternion, Matrix44, Vector3, euler
import numpy as np

from .utils3d import *

class transform3d():
    def __init__(self):
        self.location = Vector3()
        self.scale = Vector3([1, 1, 1])
        self.rotation = Rotator([0.0, 0.0, 0.0])
        self.quaternion = Quaternion([0, 0, 0, 1])
        self.initial_matrix = Matrix44.identity()
        
        self.transform_matrix = Matrix44.identity()
        # Flag indicate whether the transformation is modified or not
        self.is_changed = False

    def to_matrix(self):
        if self.is_changed:
            self.update_transform_matrix()

        return self.transform_matrix
    
    def update_transform_matrix(self):
        scale_matrix = Matrix44.from_scale(self.scale)
        translation_matrix = Matrix44.from_translation(self.location)
        
        # TODO: For some reason the qx, qy, qz part of the quaternion must be flipped
        # Need to understand why and fix it
        # The change need to be made together with the coordinate conversion in NDDS

        # rotation_matrix = Matrix44.from_quaternion(self.quaternion)
        qx, qy, qz, qw = self.quaternion
        test_quaternion = Quaternion([-qx, -qy, -qz, qw])
        rotation_matrix = Matrix44.from_quaternion(test_quaternion)

        # print('update_transform_matrix: rotation_matrix = {}'.format(rotation_matrix))

        relative_matrix = (translation_matrix * scale_matrix * rotation_matrix)
        # self.transform_matrix =  relative_matrix * self.initial_matrix
        self.transform_matrix = relative_matrix
        # print('update_transform_matrix: transform_matrix = {}'.format(self.transform_matrix))

    def mark_changed(self):
        self.is_changed = True

    # ======================== Rotation ========================
    def set_euler_rotation(self, new_rotation):
        self.rotation = new_rotation
        new_quaternion = self.rotation.to_quaternion()
        self.set_quaternion(new_quaternion)

    def rotate(self, angle_rotator):
        # new_rotator = self.rotation + angle_rotator
        new_rotator = self.rotation.add(angle_rotator)
        # print("New rotation: {}".format(new_rotator))
        self.set_euler_rotation(new_rotator)

    def set_quaternion(self, new_quaternion):
        # print("New quaternion: {}".format(new_quaternion))
        self.quaternion = new_quaternion
        self.mark_changed()

    # ======================== Scale ========================
    #  Scale the mesh with the same amount between all the axis
    def set_scale_uniform(self, uniform_scale):
        self.scale *= uniform_scale
        self.mark_changed()
    
    def set_scale(self, new_scale):
        self.scale = new_scale
        self.mark_changed()

    # ======================== Translation ========================
    def set_location(self, new_location):
        self.location = new_location
        self.mark_changed()

    def move(self, move_vector):
        new_location = self.location + move_vector
        self.set_location(new_location)

    # ======================== Others ========================
    def set_initial_matrix(self, new_initial_matrix):
        self.initial_matrix = new_initial_matrix
        self.mark_changed()

    def reset_transform(self):
        self.set_location(Vector3())
        # self.set_rotation(euler.create(0.0, 0.0, 0.0))
        self.set_quaternion(Quaternion([0, 0, 0, 1]))
        self.mark_changed()
