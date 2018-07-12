# Copyright © 2018 NVIDIA Corporation.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

from pyrr import Quaternion, Matrix44, Vector3, euler
import numpy as np

# =============================== Data parsing ===============================

# ========================= CoordinateSystem =========================
# TODO: May just want to use the Transform3d class
class CoordinateSystem(object):
    """This class present a coordinate system with 3 main directions
    Each direction is represent by a vector in OpenCV coordinate system
    By default in OpenCV coordinate system:
    Forward: Z - [0, 0, 1]
    Right: X - [1, 0, 0]
    Up: -Y - [0, -1, 0]
    """
    def __init__(self, 
        forward = [0, 0, 1],
        right = [1, 0, 0],
        up = [0, -1, 0]):
        self.forward = forward
        self.right = right
        self.up = up

    # TODO: Build the transform matrix to convert from OpenCV to this coordinate system

# ========================= Rotator =========================
class Rotator():
    def __init__(self, angles = [0, 0, 0]):
        # Store the angle (in radian) rotated in each axis: X, Y, Z
        self.angles = angles

    # NOTE: All the calculation use the OpenCV coordinate system
    # http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/OWENS/LECT9/node2.html

    def __str__(self):
        return "({}, {}, {})".format(self.angles[0], self.angles[1], self.angles[2])

    @property
    def yaw(self):
        return self.angle[1]

    @property
    def pitch(self):
        return self.angle[0]
    
    @property
    def roll(self):
        return self.angle[2]

    @staticmethod
    def create_from_yaw_pitch_roll(yaw = 0, pitch = 0, roll = 0):
        return rotator([pitch, yaw, roll])

    @staticmethod
    def create_from_yaw_pitch_roll_degree(yaw = 0, pitch = 0, roll = 0):
        return rotator([np.deg2rad(pitch), np.deg2rad(yaw), np.deg2rad(roll)])

    def add(self, other_rotator):
        return rotator([
            self.angles[0] + other_rotator.angles[0],
            self.angles[1] + other_rotator.angles[1],
            self.angles[2] + other_rotator.angles[2],
            ])
    
    # https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    # NOTE: The Tait-Bryan angles result is reverted so instead of ZXY we must use the YXZ
    # NOTE: The rotation order is Yaw Pitch Roll or Y X Z
    # Output: 3x3 rotation row-major matrix
    def to_rotation_matrix(self):
        x, y, z = self.angles

        # Z1 X2 Y3
        c1 = np.cos(z)
        s1 = np.sin(z)
        c2 = np.cos(x)
        s2 = np.sin(x)
        c3 = np.cos(y)
        s3 = np.sin(y)

        return np.array(
            [
                [
                    c1 * c3 - s1 * s2 * s3,
                    -c2 * s1,
                    c1 * s3 + c3 * s1 * s2
                ],
                [
                    c3 * s1 + c1 * s2 * s3,
                    c1 * c2,
                    s1 * s3 - c1 * c3 * s2
                ],
                [
                    -c2 * s3,
                    s2,
                    c2 * c3
                ]
            ]
        )

    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_Angles_to_Quaternion_Conversion
    # NOTE: The rotation order is Yaw Pitch Roll or Y X Z
    def to_quaternion(self):
        x, y, z = self.angles

        halfX = x * 0.5
        halfY = y * 0.5
        halfZ = z * 0.5

        cX = np.cos(halfX)
        sX = np.sin(halfX)
        cY = np.cos(halfY)
        sY = np.sin(halfY)
        cZ = np.cos(halfZ)
        sZ = np.sin(halfZ)

        return np.array(
            [
                -sX * cY * cZ - cX * sY * sZ,
                sX * cY * sZ - cX * sY * cZ,
                sX * sY * cZ - cX * cY * sZ,
                sX * sY * sZ + cX * cY * cZ
            ]
        )