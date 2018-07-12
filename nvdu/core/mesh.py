# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

from .scene_object import *

# ========================= Cuboid2d =========================
class Mesh(SceneObject):
    """Container for a 3d model"""
    def __init__(self, mesh_file_path=None, in_parent_object = None):
        super(Mesh, self).__init__(in_parent_object)

        self.source_file_path = mesh_file_path

    def set_initial_matrix(self, new_initial_matrix):
        self._relative_transform.set_initial_matrix(new_initial_matrix)

    def get_initial_matrix(self):
        return self._relative_transform.initial_matrix
