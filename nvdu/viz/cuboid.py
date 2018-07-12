# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

from pyrr import Quaternion, Matrix44, Vector3, euler
import numpy as np
from pyglet.gl import *
from ctypes import *

from .scene_object import *
from nvdu.core.cuboid import *

# ========================= Cuboid3d =========================
# TODO: Should merge Cuboid and Box3d
class Cuboid3dViz(SceneObjectViz):
    # Create a box with a certain size
    def __init__(self, cuboid3d, in_color=None):
        super(Cuboid3dViz, self).__init__(cuboid3d)
        self.cuboid3d = cuboid3d
        self.vertices = self.cuboid3d.get_vertices()

        self.render_line = True
        self.color = in_color
        self.face_alpha = 128
        # self.render_line = False

        self.generate_vertex_buffer()

    def generate_vertex_buffer(self):
        self.vertices_gl = []
        for i in range(0, CuboidVertexType.TotalCornerVertexCount):
            self.vertices_gl.append(self.vertices[i][0])
            self.vertices_gl.append(self.vertices[i][1])
            self.vertices_gl.append(self.vertices[i][2])

        # List of color for each vertices of the box
        if (self.color is None):
            self.colors_gl = [
                0, 0, 255, 255,     # Front Top Right
                0, 0, 255, 255,     # Front Top Left
                255, 0, 255, 255,   # Front Bottom Left
                255, 0, 255, 255,   # Front Bottom Right
                0, 255, 0, 255,     # Rear Top Right
                0, 255, 0, 255,     # Rear Top Left
                255, 255, 0, 255,   # Rear Bottom Left
                255, 255, 0, 255,   # Rear Bottom Right
            ]
        else:
            self.colors_gl = []
            for i in range(0, CuboidVertexType.TotalCornerVertexCount):
                for color_channel in self.color:
                    self.colors_gl.append(color_channel)

        # Reduce the alpha of the vertex colors when we rendering triangles
        self.colors_tri_gl = list(int(color / 4) for color in self.colors_gl)

        cvt = CuboidVertexType
        # Counter-Clockwise order triangle indices
        self.indices_tri_gl = [
            # Front face
            cvt.FrontBottomLeft, cvt.FrontTopLeft, cvt.FrontTopRight,
            cvt.FrontTopRight, cvt.FrontBottomRight, cvt.FrontBottomLeft,
            # Right face
            cvt.FrontBottomRight, cvt.FrontTopRight, cvt.RearBottomRight,
            cvt.RearTopRight, cvt.RearBottomRight, cvt.FrontTopRight,
            # Back face
            cvt.RearBottomLeft, cvt.RearBottomRight, cvt.RearTopRight,
            cvt.RearTopRight, cvt.RearTopLeft, cvt.RearBottomLeft,
            # Left face
            cvt.FrontTopLeft, cvt.FrontBottomLeft, cvt.RearBottomLeft,
            cvt.RearBottomLeft, cvt.RearTopLeft, cvt.FrontTopLeft,
            # Top face
            cvt.RearTopLeft, cvt.RearTopRight, cvt.FrontTopRight,
            cvt.FrontTopRight, cvt.FrontTopLeft, cvt.RearTopLeft,
            # Bottom face
            cvt.RearBottomLeft, cvt.FrontBottomLeft, cvt.FrontBottomRight,
            cvt.FrontBottomRight, cvt.RearBottomRight, cvt.RearBottomLeft,
        ]
        self.indices_line_gl = np.array(CuboidLineIndexes).flatten()
        # print("indices_line_gl: {}".format(self.indices_line_gl))

        self.indices_point_gl = list(range(0, len(self.vertices)))
        # print('indices_point_gl: {}'.format(self.indices_point_gl))

        self.vertex_gl_array = (GLfloat* len(self.vertices_gl))(*self.vertices_gl)
        self.color_gl_array = (GLubyte* len(self.colors_gl))(*self.colors_gl)
        self.colors_tri_gl_array = (GLubyte* len(self.colors_tri_gl))(*self.colors_tri_gl)
        self.indices_tri_gl_array = (GLubyte* len(self.indices_tri_gl))(*self.indices_tri_gl)
        self.indices_line_gl_array = (GLubyte* len(self.indices_line_gl))(*self.indices_line_gl)
        self.indices_point_gl_array = (GLubyte* len(self.indices_point_gl))(*self.indices_point_gl)

    def on_draw(self):
        super(Cuboid3dViz, self).on_draw()
        # print('Cuboid3dViz - on_draw - vertices: {}'.format(self.vertices))

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        # glEnable(GL_POLYGON_SMOOTH)

        glPolygonMode(GL_FRONT_AND_BACK, RenderMode.normal)

        glVertexPointer(3, GL_FLOAT, 0, self.vertex_gl_array)
        
        # Render each faces of the cuboid
        glColorPointer(4, GL_UNSIGNED_BYTE, 0, self.colors_tri_gl_array)
        glDrawElements(GL_TRIANGLES, len(self.indices_tri_gl), GL_UNSIGNED_BYTE, self.indices_tri_gl_array)

        # Render each edge lines
        glEnable(GL_LINE_SMOOTH)
        glLineWidth(3.0)
        glColorPointer(4, GL_UNSIGNED_BYTE, 0, self.color_gl_array)
        # TODO: May want to use GL_LINE_STRIP or GL_LINE_LOOP
        glDrawElements(GL_LINES, len(self.indices_line_gl), GL_UNSIGNED_BYTE, self.indices_line_gl_array)
        glDisable(GL_LINE_SMOOTH)

        # Render each corner vertices in POINTS mode
        glPointSize(10.0)
        glDrawElements(GL_POINTS, len(self.indices_point_gl_array), GL_UNSIGNED_BYTE, self.indices_point_gl_array)
        glPointSize(1.0)
      
        # Deactivate vertex arrays after drawing
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        # glDisable(GL_POLYGON_SMOOTH)

# ========================= Cuboid2d =========================
class Cuboid2dViz(SceneObjectViz):
    # Create a box with a certain size
    def __init__(self, cuboid2d, in_color=None):
        super(Cuboid2dViz, self).__init__(cuboid2d)
        self.cuboid2d = cuboid2d
        self.color = in_color
        if not (self.cuboid2d is None):
            self.vertices = self.cuboid2d.get_vertices()
            self.generate_vertexes_buffer()
            # self.render_line = False

    def generate_vertexes_buffer(self):
        self.vertices_gl = []
        max_vertex_count = min(CuboidVertexType.TotalVertexCount, len(self.vertices))
        for i in range(0, max_vertex_count):
            vertex = self.vertices[i]
            if (not vertex is None):
                self.vertices_gl.append(self.vertices[i][0])
                self.vertices_gl.append(self.vertices[i][1])
            else:
                self.vertices_gl.append(0.0)
                self.vertices_gl.append(0.0)
                
        # List of color for each vertices of the box
        self.vertex_colors_gl = [
            0, 0, 255, 255,     # Front Top Right
            0, 0, 255, 255,     # Front Top Left
            255, 0, 255, 255,   # Front Bottom Left
            255, 0, 255, 255,   # Front Bottom Right
            0, 255, 0, 255,     # Rear Top Right
            0, 255, 0, 255,     # Rear Top Left
            255, 255, 0, 255,   # Rear Bottom Left
            255, 255, 0, 255,   # Rear Bottom Right
        ]

        # List of color for each vertices of the box
        if (self.color is None):
            self.edge_colors_gl = self.vertex_colors_gl
        else:
            self.edge_colors_gl = []
            for i in range(0, CuboidVertexType.TotalCornerVertexCount):
                for color_channel in self.color:
                    self.edge_colors_gl.append(color_channel)

        # NOTE: Only add valid lines:
        self.indices_line_gl = []
        for line in CuboidLineIndexes:
            vi0, vi1 = line
            v0 = self.vertices[vi0]
            v1 = self.vertices[vi1]
            if not (v0 is None) and not (v1 is None):
                self.indices_line_gl.append(vi0)
                self.indices_line_gl.append(vi1)
        # print('indices_line_gl: {}'.format(self.indices_line_gl))

        # self.indices_line_gl = [
        #     # Front face
        #     cvt.FrontTopLeft, cvt.FrontTopRight,
        #     cvt.FrontTopRight, cvt.FrontBottomRight,
        #     cvt.FrontBottomRight, cvt.FrontBottomLeft,
        #     cvt.FrontBottomLeft, cvt.FrontTopLeft,
        #     # Back face
        #     cvt.RearTopLeft, cvt.RearTopRight,
        #     cvt.RearTopRight, cvt.RearBottomRight,
        #     cvt.RearBottomRight, cvt.RearBottomLeft,
        #     cvt.RearBottomLeft, cvt.RearTopLeft,
        #     # Left face
        #     cvt.FrontBottomLeft, cvt.RearBottomLeft,
        #     cvt.FrontTopLeft, cvt.RearTopLeft,
        #     # Right face
        #     cvt.FrontBottomRight, cvt.RearBottomRight,
        #     cvt.FrontTopRight, cvt.RearTopRight,
        # ]

        # self.indices_point_gl = list(i for i in range(0, len(self.vertices)))
        # NOTE: Only add valid points:
        self.indices_point_gl = []
        for i in range (len(self.vertices)):
            if (not self.vertices[i] is None):
                self.indices_point_gl.append(i)
        # print('indices_point_gl: {}'.format(self.indices_point_gl))

        self.vertex_gl_array = (GLfloat* len(self.vertices_gl))(*self.vertices_gl)
        self.vertex_color_gl_array = (GLubyte* len(self.vertex_colors_gl))(*self.vertex_colors_gl)
        self.edge_colors_gl_array = (GLubyte* len(self.edge_colors_gl))(*self.edge_colors_gl)
        self.indices_line_gl_array = (GLubyte* len(self.indices_line_gl))(*self.indices_line_gl)
        self.indices_point_gl_array = (GLubyte* len(self.indices_point_gl))(*self.indices_point_gl)

    def on_draw(self):
        if (self.cuboid2d is None):
            return

        super(Cuboid2dViz, self).on_draw()
        # print('Cuboid2dViz - on_draw - vertices: {}'.format(self.vertices))

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        # glEnable(GL_POLYGON_SMOOTH)

        glPolygonMode(GL_FRONT_AND_BACK, RenderMode.normal)

        glVertexPointer(2, GL_FLOAT, 0, self.vertex_gl_array)

        glEnable(GL_LINE_SMOOTH)
        glLineWidth(3.0)
        glColorPointer(4, GL_UNSIGNED_BYTE, 0, self.edge_colors_gl_array)
        # TODO: May want to use GL_LINE_STRIP or GL_LINE_LOOP
        glDrawElements(GL_LINES, len(self.indices_line_gl), GL_UNSIGNED_BYTE, self.indices_line_gl_array)
        glDisable(GL_LINE_SMOOTH)

        glColorPointer(4, GL_UNSIGNED_BYTE, 0, self.vertex_color_gl_array)
        glPointSize(10.0)
        glDrawElements(GL_POINTS, len(self.indices_point_gl_array), GL_UNSIGNED_BYTE, self.indices_point_gl_array)
        glPointSize(1.0)
      
        # Deactivate vertex arrays after drawing
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        # glDisable(GL_POLYGON_SMOOTH)
