# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

from enum import IntEnum, unique
import numpy as np
import cv2
from .scene_object import *

# Related to the object's local coordinate system
# @unique
class CuboidVertexType(IntEnum):
    FrontTopRight = 0
    FrontTopLeft = 1
    FrontBottomLeft = 2
    FrontBottomRight = 3
    RearTopRight = 4
    RearTopLeft = 5
    RearBottomLeft = 6
    RearBottomRight = 7
    Center = 8
    TotalCornerVertexCount = 8 # Corner vertexes doesn't include the center point
    TotalVertexCount = 9

# List of the vertex indexes in each line edges of the cuboid
CuboidLineIndexes = [
    # Front face
    [ CuboidVertexType.FrontTopLeft,      CuboidVertexType.FrontTopRight ],
    [ CuboidVertexType.FrontTopRight,     CuboidVertexType.FrontBottomRight ],
    [ CuboidVertexType.FrontBottomRight,  CuboidVertexType.FrontBottomLeft ],
    [ CuboidVertexType.FrontBottomLeft,   CuboidVertexType.FrontTopLeft ],
    # Back face
    [ CuboidVertexType.RearTopLeft,       CuboidVertexType.RearTopRight ],
    [ CuboidVertexType.RearTopRight,      CuboidVertexType.RearBottomRight ],
    [ CuboidVertexType.RearBottomRight,   CuboidVertexType.RearBottomLeft ],
    [ CuboidVertexType.RearBottomLeft,    CuboidVertexType.RearTopLeft ],
    # Left face
    [ CuboidVertexType.FrontBottomLeft,   CuboidVertexType.RearBottomLeft ],
    [ CuboidVertexType.FrontTopLeft,      CuboidVertexType.RearTopLeft ],
    # Right face
    [ CuboidVertexType.FrontBottomRight,  CuboidVertexType.RearBottomRight ],
    [ CuboidVertexType.FrontTopRight,     CuboidVertexType.RearTopRight ],
]

# ========================= Cuboid2d =========================
class Cuboid2d(SceneObject):
    """Container for 2d projected points of a cuboid on an image"""
    def __init__(self, vertices=[]):
        """Create a cuboid 2d from a list of 2d points
        Args:
            vertices - numpy array ([8, 9] * 2)
        """
        super(Cuboid2d, self).__init__()

        self._vertices = np.array(vertices)
        # print('Cuboid2d - vertices: {}'.format(self._vertices))

    def get_vertex(self, vertex_type):
        """Get the location of a vertex
        Args:
            vertex_type: enum of type CuboidVertexType
        Return:
            Numpy array(3) - Location of the vertex type in the cuboid
        """
        return self._vertices[vertex_type]

    def get_vertices(self):
        return self._vertices

# ========================= Cuboid3d =========================
class Cuboid3d(SceneObject):
    # Create a box with a certain size
    # TODO: Instead of using center_location and coord_system, should pass in a Transform3d
    def __init__(self, size3d = [1.0, 1.0, 1.0], center_location = [0, 0, 0],
        coord_system = None, parent_object = None):
        super(Cuboid3d, self).__init__(parent_object)

        # NOTE: This local coordinate system is similar
        # to the intrinsic transform matrix of a 3d object
        self.center_location = center_location
        self.coord_system = coord_system
        self.size3d = size3d
        self._vertices = [0, 0, 0] * CuboidVertexType.TotalVertexCount

        self.generate_vertexes()

    def get_vertex(self, vertex_type):
        """Get the location of a vertex
        Args:
            vertex_type: enum of type CuboidVertexType
        Return:
            Numpy array(3) - Location of the vertex type in the cuboid
        """
        return self._vertices[vertex_type]

    def get_vertices(self):
        return self._vertices

    def generate_vertexes(self):
        width, height, depth = self.size3d

        # By default just use the normal OpenCV coordinate system
        if (self.coord_system is None):
            cx, cy, cz = self.center_location
            # X axis point to the right
            right = cx + width / 2.0
            left = cx - width / 2.0
            # Y axis point downward
            top = cy - height / 2.0
            bottom = cy + height / 2.0
            # Z axis point forward
            front = cz + depth / 2.0
            rear = cz - depth / 2.0

            # List of 8 vertices of the box       
            self._vertices = [
                [right, top, front],    # Front Top Right
                [left, top, front],     # Front Top Left
                [left, bottom, front],  # Front Bottom Left
                [right, bottom, front], # Front Bottom Right
                [right, top, rear],     # Rear Top Right
                [left, top, rear],      # Rear Top Left
                [left, bottom, rear],   # Rear Bottom Left
                [right, bottom, rear],  # Rear Bottom Right
                self.center_location,   # Center
            ]
        else:
            # NOTE: should use quaternion for initial transform
            sx, sy, sz = self.size3d
            forward = np.array(self.coord_system.forward, dtype=float) * sy * 0.5
            up = np.array(self.coord_system.up, dtype=float) * sz * 0.5
            right = np.array(self.coord_system.right, dtype=float) * sx * 0.5
            center = np.array(self.center_location, dtype=float)
            self._vertices = [
                center + forward + up + right,      # Front Top Right
                center + forward + up - right,      # Front Top Left
                center + forward - up - right,      # Front Bottom Left
                center + forward - up + right,      # Front Bottom Right
                center - forward + up  + right,     # Rear Top Right
                center - forward + up - right,      # Rear Top Left
                center - forward - up - right,      # Rear Bottom Left
                center - forward - up + right,      # Rear Bottom Right
                self.center_location,               # Center
            ]
            # print("cuboid3d - forward: {} - up: {} - right: {}".format(forward, up, right))

        # print("cuboid3d - size3d: {}".format(self.size3d))
        # print("cuboid3d - depth: {} - width: {} - height: {}".format(depth, width, height))
        # print("cuboid3d - vertices: {}".format(self._vertices))

    def get_projected_cuboid2d(self, cuboid_transform, camera_intrinsic_matrix):
        """
        Project the cuboid into the projection plane using CameraIntrinsicSettings to get a cuboid 2d
        Args:
            cuboid_transform: the world transform of the cuboid
            camera_intrinsic_matrix: camera intrinsic matrix
        Return:
            Cuboid2d - the projected cuboid points
        """

        # projected_vertices = [0, 0] * CuboidVertexType.TotalVertexCount
        # world_transform_matrix = self.get_world_transform_matrix()
        world_transform_matrix = cuboid_transform
        rvec = [0, 0, 0]
        tvec = [0, 0, 0]
        dist_coeffs = np.zeros((4, 1))

        transformed_vertices = [0, 0, 0] * CuboidVertexType.TotalVertexCount
        for vertex_index in range(CuboidVertexType.TotalVertexCount):
            vertex3d = self._vertices[vertex_index]
            transformed_vertices[vertex_index] = world_transform_matrix * vertex3d

        projected_vertices = cv2.projectPoints(transformed_vertices, rvec, tvec, 
                                camera_intrinsic_matrix, dist_coeffs)

        return Cuboid2d(projected_vertices)
