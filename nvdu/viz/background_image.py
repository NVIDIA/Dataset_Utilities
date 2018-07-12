# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

import cv2
import numpy as np
from pyglet.gl import *

class BackgroundImage(object):
    def __init__(self, width = 0, height = 0):
        self.width = width
        self.height = height
        self.location = [0, 0, 0]
        self.vlist = pyglet.graphics.vertex_list(4,
                ('v2f', [0,0, width,0, 0,height, width,height]),
                ('t2f', [0,0, width,0, 0,height, width,height]))

        self.scale = [self.width, self.height, 1.0]

    @classmethod
    def create_from_numpy_image_data(cls, numpy_image_data, width = 0, height = 0):
        img_width = numpy_image_data.shape[0] if (width == 0) else width
        img_height = numpy_image_data.shape[1] if (height == 0) else height
        
        new_image = cls(img_width, img_height)
        new_image.load_image_data_from_numpy(numpy_image_data)
        return new_image

    @classmethod
    def create_from_file_path(cls, image_file_path, width = 0, height = 0):
        image_np = np.array(cv2.imread(image_file_path))
        image_np = image_np[:,:,::-1]  # Convert BGR to RGB format.  Alternatively, use cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cls.create_from_numpy_image_data(image_np, width, height)
    
    def load_image_data_from_numpy(self, numpy_image_data):
        width = numpy_image_data.shape[1]
        height = numpy_image_data.shape[0]
        color_channel_count = numpy_image_data.shape[2]
        pitch = -width * color_channel_count
        # print('numpy_image_data.shape: {}'.format(numpy_image_data.shape))
        img_data = numpy_image_data
        img_data = img_data.tostring()
        self.image = pyglet.image.ImageData(width, height, 'RGB', img_data, pitch)
        self.texture = self.image.get_texture(True, True)

    def load_image_from_file(self, image_file_path):
        self.image = pyglet.image.load(image_file_path)
        self.texture = self.image.get_texture(True, True)

    def load_new_image(self, image_file_path):
        self.image = pyglet.image.load(image_file_path)
        self.texture = self.image.get_texture(True, True)
        # print('Texture: {} - id: {} - target:{} - width: {} - height: {}'.format(
        #     self.texture, self.texture.id, self.texture.target, self.texture.width, self.texture.height))
        # print('GL_TEXTURE_RECTANGLE_ARB: {} - GL_TEXTURE_RECTANGLE_NV: {} - GL_TEXTURE_2D: {}'.format(
        #     GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_2D))

    def draw(self):
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        x, y, z = self.location
        glTranslatef(x, y, z)
        glColor3f(1, 1, 1)

        texture_target = self.texture.target
        glEnable(texture_target)
        glBindTexture(texture_target, self.texture.id)
        self.vlist.draw(GL_TRIANGLE_STRIP)
        glDisable(texture_target)

        glPopMatrix()
