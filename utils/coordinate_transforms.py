# coordinate_transforms.py
"""
Coordinate Transformation Utilities

This module implements coordinate transformations used in
satellite navigation, orbital mechanics, and vision-based
space object detection.

Frames supported:

ECI  : Earth Centered Inertial
ECEF : Earth Centered Earth Fixed
LVLH : Local Vertical Local Horizontal
Body : Satellite body frame
Camera : Camera frame mounted on CubeSat

These transformations allow detected objects from camera
images to be mapped into orbital coordinates.
"""

import numpy as np
from numpy.linalg import norm
from datetime import datetime

# Earth rotation rate (rad/s)
OMEGA_EARTH = 7.2921159e-5


# ---------------------------------------------------------
# Rotation Matrix Utilities
# ---------------------------------------------------------

def rot_x(angle):
    """Rotation matrix about X axis."""
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rot_y(angle):
    """Rotation matrix about Y axis."""
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rot_z(angle):
    """Rotation matrix about Z axis."""
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


# ---------------------------------------------------------
# Julian Date Conversion
# ---------------------------------------------------------

def datetime_to_julian(date: datetime):
    """
    Convert datetime to Julian Date.
    """

    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    minute = date.minute
    second = date.second

    if month <= 2:
        year -= 1
        month += 12

    A = int(year / 100)
    B = 2 - A + int(A / 4)

    JD = int(365.25 * (year + 4716)) \
         + int(30.6001 * (month + 1)) \
         + day + B - 1524.5

    frac_day = (hour + minute/60 + second/3600) / 24

    return JD + frac_day


# ---------------------------------------------------------
# Greenwich Sidereal Time
# ---------------------------------------------------------

def greenwich_sidereal_time(date: datetime):
    """
    Compute Greenwich Mean Sidereal Time (GMST).
    """

    JD = datetime_to_julian(date)

    T = (JD - 2451545.0) / 36525.0

    GMST = 280.46061837 \
           + 360.98564736629 * (JD - 2451545) \
           + 0.000387933 * T**2 \
           - T**3 / 38710000

    GMST = np.radians(GMST % 360)

    return GMST


# ---------------------------------------------------------
# ECI ↔ ECEF Transformations
# ---------------------------------------------------------

def eci_to_ecef(position_eci, date):
    """
    Convert position from ECI to ECEF frame.
    """

    theta = greenwich_sidereal_time(date)

    R = rot_z(theta)

    position_ecef = R @ position_eci

    return position_ecef


def ecef_to_eci(position_ecef, date):
    """
    Convert position from ECEF to ECI frame.
    """

    theta = greenwich_sidereal_time(date)

    R = rot_z(-theta)

    position_eci = R @ position_ecef

    return position_eci


# ---------------------------------------------------------
# LVLH Frame Transformation
# ---------------------------------------------------------

def eci_to_lvlh(r_sat, v_sat):
    """
    Construct transformation matrix from ECI to LVLH frame.

    LVLH axes:
    x -> radial
    y -> along-track
    z -> cross-track
    """

    r = r_sat / norm(r_sat)

    h = np.cross(r_sat, v_sat)
    z = h / norm(h)

    y = np.cross(z, r)

    T = np.vstack([r, y, z]).T

    return T


def eci_vector_to_lvlh(vector_eci, r_sat, v_sat):
    """
    Transform vector from ECI to LVLH frame.
    """

    T = eci_to_lvlh(r_sat, v_sat)

    return T.T @ vector_eci


def lvlh_to_eci(vector_lvlh, r_sat, v_sat):
    """
    Transform vector from LVLH to ECI.
    """

    T = eci_to_lvlh(r_sat, v_sat)

    return T @ vector_lvlh


# ---------------------------------------------------------
# Satellite Body Frame
# ---------------------------------------------------------

def body_to_lvlh(attitude_matrix, vector_body):
    """
    Convert vector from satellite body frame to LVLH.
    """

    return attitude_matrix @ vector_body


def lvlh_to_body(attitude_matrix, vector_lvlh):
    """
    Convert LVLH vector into satellite body frame.
    """

    return attitude_matrix.T @ vector_lvlh


# ---------------------------------------------------------
# Camera Projection Model
# ---------------------------------------------------------

def camera_to_body(pixel, camera_intrinsics):
    """
    Convert pixel coordinates into a unit vector
    in the camera frame.

    Uses pinhole camera model.
    """

    fx = camera_intrinsics["fx"]
    fy = camera_intrinsics["fy"]
    cx = camera_intrinsics["cx"]
    cy = camera_intrinsics["cy"]

    x = (pixel[0] - cx) / fx
    y = (pixel[1] - cy) / fy

    direction = np.array([x, y, 1.0])

    return direction / norm(direction)


def camera_ray_to_eci(
    pixel,
    camera_intrinsics,
    attitude_matrix,
    r_sat,
    v_sat
):
    """
    Convert detected pixel to an ECI direction vector.

    Steps:
    Camera → Body → LVLH → ECI
    """

    ray_camera = camera_to_body(pixel, camera_intrinsics)

    ray_body = ray_camera

    ray_lvlh = body_to_lvlh(attitude_matrix, ray_body)

    ray_eci = lvlh_to_eci(ray_lvlh, r_sat, v_sat)

    return ray_eci / norm(ray_eci)


# ---------------------------------------------------------
# Line of Sight Distance Estimation
# ---------------------------------------------------------

def line_of_sight_intersection(
    r_sat,
    direction,
    target_distance
):
    """
    Estimate object position along a viewing ray.

    r_sat : satellite position
    direction : viewing direction
    target_distance : estimated range
    """

    return r_sat + direction * target_distance