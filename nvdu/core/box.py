# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

from .scene_object import *

class Box2d(SceneObject):
    # Create a box from its border
    def __init__(self, left, right, top, bottom):
        super(BoundingBox2d, self).__init__()
        
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

        self.generate_vertexes()

    def get_width():
        return (self.right - self.left)

    def get_height():
        return (self.bottom - self.top)

    def get_size():
        return [self.get_width(), self.get_height()]