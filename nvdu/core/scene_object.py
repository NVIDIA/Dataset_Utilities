# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

from pyrr import Quaternion, Matrix44, Vector3, euler
import numpy as np

from .utils3d import *
from .transform3d import *

class SceneObject(object):
    def __init__(self, in_parent_object = None):
        # Relative transform relate to the parent object
        self._relative_transform = transform3d()

        # List of child object attached to this scene object
        self.child_objects = []
        # The object which this object is attached to
        self.parent_object = in_parent_object
        self.attach_to_object(in_parent_object)

    def attach_to_object(self, new_parent_object):
        if (self.parent_object):
            self.parent_object.remove_child_object(self)
        
        self.parent_object = new_parent_object
        if (self.parent_object):
            self.parent_object.child_objects.append(self)
    
    def remove_child_object(self, child_object_to_be_removed):
        print('***********************************')
        if (child_object_to_be_removed):
            try:
                self.child_objects.remove(child_object_to_be_removed)
            except ValueError:
                pass        

    def set_relative_transform(self, new_location, new_quaternion):
        self._relative_transform.set_location(new_location)
        self._relative_transform.set_quaternion(new_quaternion)

    def get_relative_transform(self):
        return self._relative_transform

    def get_relative_transform_matrix(self):
        return self._relative_transform.to_matrix()

    def get_world_transform_matrix(self):
        """Get the World-to-Object transform matrix"""
        if (self.parent_object is None):
            return self.get_relative_transform_matrix()       
        parent_world_matrix = self.parent_object.get_world_transform_matrix()
        world_transform_matrix = parent_world_matrix * self.get_relative_transform_matrix()
        return world_transform_matrix
