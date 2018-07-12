# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

from pyrr import Quaternion, Matrix44, Vector3, euler
import numpy as np

from .utils3d import *
from nvdu.core import transform3d
from .camera import *

# A Scene manage all of the objects need to be rendered
class Scene(object):
    def __init__(self, owner_viewport):
        self.viewport = owner_viewport
        # List of the objects in the scene - SceneObject type
        self.objects = []
        # TODO: May need to derive from SceneObject
        # self._is_visible = True

    def clear(self):
        self.objects = []

    def add_object(self, new_obj):
        # TODO: Make sure the new_obj is new and not already in the objects list?
        self.objects.append(new_obj)

    def draw(self):
        pass

# ================================= Scene2d =================================
class Scene2d(Scene):
    def __init__(self, owner_viewport):
        super(Scene2d, self).__init__(owner_viewport)
    
    def draw(self):
        super(Scene2d, self).draw()

        viewport_width, viewport_height = self.viewport.size
        # print('Scene2d - {} - draw'.format(self))
        
        # Disable depth when rendering 2d scene
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDepthMask(False)
        # Use Orthographic camera in full viewport size for 2d scene
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0.0, viewport_width, 0.0, viewport_height)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        # NOTE: OpenCV 2d image coordinate system have Y going down
        # while OpenGL have Y going up => need to flip the Y axis
        # and add the viewport_height so the OpenCV coordinate appear right
        glTranslatef(0.0, viewport_height, 0.0)
        glMultMatrixf(get_opengl_matrixf(opencv_to_opengl_matrix))

        # Render the objects in the scene
        for obj in self.objects:
            if (obj):
                obj.draw()

        glPopMatrix()
        glDepthMask(True)

# ================================= Scene3d =================================
class Scene3d(Scene):
    def __init__(self, owner_viewport):
        super(Scene3d, self).__init__(owner_viewport)
        self.camera = None
    
    def draw(self):
        super(Scene3d, self).draw()

        # print('Scene3d - {} - draw'.format(self))

        glPushMatrix()
        self.camera.draw()
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glPushMatrix()
        glMultMatrixf(get_opengl_matrixf(opencv_to_opengl_matrix))

        # TODO: Sort the 3d objects in the scene from back to front (Z reducing)
        for child_object in self.objects:
            if child_object:
                child_object.draw()

        glPopMatrix()
        glPopMatrix()
