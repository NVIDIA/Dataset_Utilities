# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

import math
import sys
import cv2

from nvdu.core.cuboid import *

def is_point_valid(point):
    if (point is None):
        return False
    if (math.isnan(point[0]) or math.isnan(point[1])):
        return False
    # NOTE: sometime the value get too big and we run into error:
    # OverflowError: Python int too large to convert to C long
    # if (math.fabs(point[0]) >= sys.maxsize) or (math.fabs(point[1]) >= sys.maxsize):
    if (math.fabs(point[0]) >= 10000) or (math.fabs(point[1]) >= 10000):
        return False
    return True

# This module contains all the functions related to drawing on image
def draw_cuboid2d(image, cuboid2d, color, line_thickness=1, point_size=1):
    if (image is None) or (cuboid2d is None):
        return

    # print("image: {} - image.shape: {}".format(image, image.shape))

    line_type = cv2.LINE_AA
    # Draw the lines edge of the cuboid
    for line in CuboidLineIndexes:
        vi0, vi1 = line
        v0 = cuboid2d.get_vertex(vi0)
        v1 = cuboid2d.get_vertex(vi1)
        # print("draw line - v0: {} - v1: {}".format(v0, v1))
        if (is_point_valid(v0) and is_point_valid(v1)):
            v0 = (int(v0[0]), int(v0[1]))
            v1 = (int(v1[0]), int(v1[1]))
            # print("draw line - v0: {} - v1: {}".format(v0, v1))

            cv2.line(image, v0, v1, color, line_thickness, line_type)

    # Draw circle at each corner vertices of the cuboid
    thickness = -1
    # TODO: Highlight the top front vertices
    for vertex_index in range(CuboidVertexType.TotalVertexCount):
        vertex = cuboid2d.get_vertex(vertex_index)
        if (not is_point_valid(vertex)):
            continue

        point = (int(vertex[0]), int(vertex[1]))
        cv2.circle(image, point, point_size, color, thickness, line_type)

        if (vertex_index == CuboidVertexType.FrontTopRight):
            cv2.circle(image, point, point_size, (0,0,0), int(point_size / 2), line_type)
        elif (vertex_index == CuboidVertexType.FrontTopLeft):
            cv2.circle(image, point, point_size, (0,0,0), 1, line_type)

            
        