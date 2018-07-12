# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

from .scene_object import *

# ========================= Cuboid2d =========================
class PivotAxis(SceneObject):
    """Object represent a pivot axis in 3d space"""
    def __init__(self, in_size3d = [1.0, 1.0, 1.0], in_parent_object = None):
        super(PivotAxis, self).__init__(in_parent_object)

        self.size3d = in_size3d
        self.generate_vertexes()

    def generate_vertexes(self):
        self.origin_loc = [0.0, 0.0, 0.0]
        x, y, z = self.size3d
        self.x_axis = [x, 0.0, 0.0]
        self.y_axis = [0.0, y, 0.0]
        self.z_axis = [0.0, 0.0, z]

