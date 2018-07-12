# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

from .utils3d import *
from .scene import *

class Viewport(object):
    def __init__(self, context):
        self._context = context
        self._size = [0, 0]

        self.scene_bg = Scene2d(self)
        self.scene3d = Scene3d(self)
        self.scene_overlay = Scene2d(self)
        
        self.scenes = [
            self.scene_bg,
            self.scene3d,
            self.scene_overlay
        ]

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, new_size):
        self._size[0] = new_size[0]
        self._size[1] = new_size[1]

    def clear(self):
        for scene in self.scenes:
            if (scene):
                scene.clear()

    def draw(self):
        for scene in self.scenes:
            if (scene):
                scene.draw()
