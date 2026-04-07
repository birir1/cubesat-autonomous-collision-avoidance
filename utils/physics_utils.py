"""
Physics Utilities

Fundamental physics calculations for collision avoidance.
"""

import numpy as np
from typing import Tuple, Optional, Union
import logging

# Physical constants
MU_EARTH = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
EARTH_RADIUS = 6371000  # Earth radius (m)
J2 = 0.0010826269  # Earth's J2 coefficient
EARTH_ROTATION_RATE = 7.2921159e-5  # Earth rotation rate (rad/s)

class PhysicsUtils:
    """
    Collection of physics utility functions.
    """

    @staticmethod
    def gravitational_acceleration(position: np.ndarray) -> np.ndarray:
        """
        Calculate gravitational acceleration at a position.

        Args:
            position: Position vector [x, y, z] (m)

        Returns:
            Acceleration vector [ax, ay, az] (m/s^2)
        """
        r = np.linalg.norm(position)
        if r == 0:
            return np.zeros(3)

        # Basic inverse square law
        acceleration = -MU_EARTH / (r ** 3) * position

        return acceleration

    @staticmethod
    def orbital_velocity(position: np.ndarray, velocity: np.ndarray) -> float:
        """
        Calculate orbital velocity.

        Args:
            position: Position vector [x, y, z] (m)
            velocity: Velocity vector [vx, vy, vz] (m/s)

        Returns:
            Orbital speed (m/s)
        """
        return np.linalg.norm(velocity)

    @staticmethod
    def orbital_period(semi_major_axis: float) -> float:
        """
        Calculate orbital period using Kepler's third law.

        Args:
            semi_major_axis: Semi-major axis (m)

        Returns:
            Orbital period (seconds)
        """
        return 2 * np.pi * np.sqrt(semi_major_axis ** 3 / MU_EARTH)

    @staticmethod
    def vis_viva_equation(semi_major_axis: float, radius: float) -> float:
        """
        Calculate orbital speed using vis-viva equation.

        Args:
            semi_major_axis: Semi-major axis (m)
            radius: Current orbital radius (m)

        Returns:
            Orbital speed (m/s)
        """
        return np.sqrt(MU_EARTH * (2 / radius - 1 / semi_major_axis))

    @staticmethod
    def collision_probability_cylindrical(miss_distance: float,
                                        relative_velocity: float,
                                        combined_radius: float = 100.0) -> float:
        """
        Calculate collision probability using cylindrical approximation.

        Args:
            miss_distance: Miss distance (m)
            relative_velocity: Relative velocity (m/s)
            combined_radius: Combined radius of objects (m)

        Returns:
            Collision probability
        """
        if miss_distance <= combined_radius:
            return 1.0

        # Cylindrical collision probability
        # Pc = 2 * R / (pi * miss_distance) for high relative velocities
        # Simplified calculation
        from scipy.stats import norm

        sigma = combined_radius / 3  # Assume 3-sigma containment
        pc = 2 * (1 - norm.cdf(miss_distance / sigma))

        return min(pc, 1.0)

    @staticmethod
    def mahalanobis_distance(point: np.ndarray, mean: np.ndarray,
                           covariance: np.ndarray) -> float:
        """
        Calculate Mahalanobis distance.

        Args:
            point: Point vector
            mean: Mean vector
            covariance: Covariance matrix

        Returns:
            Mahalanobis distance
        """
        diff = point - mean
        inv_covariance = np.linalg.inv(covariance)
        distance = np.sqrt(np.dot(np.dot(diff.T, inv_covariance), diff))

        return distance

    @staticmethod
    def relative_velocity(position1: np.ndarray, velocity1: np.ndarray,
                         position2: np.ndarray, velocity2: np.ndarray) -> np.ndarray:
        """
        Calculate relative velocity between two objects.

        Args:
            position1: Position of object 1 [x, y, z] (m)
            velocity1: Velocity of object 1 [vx, vy, vz] (m/s)
            position2: Position of object 2 [x, y, z] (m)
            velocity2: Velocity of object 2 [vx, vy, vz] (m/s)

        Returns:
            Relative velocity vector [dvx, dvy, dvz] (m/s)
        """
        return velocity1 - velocity2

    @staticmethod
    def relative_position(position1: np.ndarray, position2: np.ndarray) -> np.ndarray:
        """
        Calculate relative position between two objects.

        Args:
            position1: Position of object 1 [x, y, z] (m)
            position2: Position of object 2 [x, y, z] (m)

        Returns:
            Relative position vector [dx, dy, dz] (m)
        """
        return position1 - position2

    @staticmethod
    def hohmann_transfer_delta_v(semi_major_axis1: float,
                               semi_major_axis2: float) -> Tuple[float, float]:
        """
        Calculate delta-V for Hohmann transfer between two circular orbits.

        Args:
            semi_major_axis1: Initial semi-major axis (m)
            semi_major_axis2: Final semi-major axis (m)

        Returns:
            Tuple of (delta_v1, delta_v2) (m/s)
        """
        # Velocity at initial orbit
        v1 = np.sqrt(MU_EARTH / semi_major_axis1)

        # Velocity at transfer orbit perigee
        v_transfer_perigee = np.sqrt(MU_EARTH * (2 / semi_major_axis1 - 1 / ((semi_major_axis1 + semi_major_axis2) / 2)))

        # First delta-V
        delta_v1 = v_transfer_perigee - v1

        # Velocity at transfer orbit apogee
        v_transfer_apogee = np.sqrt(MU_EARTH * (2 / semi_major_axis2 - 1 / ((semi_major_axis1 + semi_major_axis2) / 2)))

        # Velocity at final orbit
        v2 = np.sqrt(MU_EARTH / semi_major_axis2)

        # Second delta-V
        delta_v2 = v2 - v_transfer_apogee

        return delta_v1, delta_v2

    @staticmethod
    def atmospheric_drag_acceleration(velocity: np.ndarray, altitude: float,
                                    mass: float = 1.0, area: float = 1.0,
                                    cd: float = 2.2) -> np.ndarray:
        """
        Calculate atmospheric drag acceleration.

        Args:
            velocity: Velocity vector [vx, vy, vz] (m/s)
            altitude: Altitude above Earth surface (m)
            mass: Mass (kg)
            area: Cross-sectional area (m^2)
            cd: Drag coefficient

        Returns:
            Drag acceleration vector [ax, ay, az] (m/s^2)
        """
        # Simplified atmospheric density model (exponential)
        rho0 = 1.225  # Sea level density (kg/m^3)
        h_scale = 8500  # Scale height (m)

        if altitude < 0:
            altitude = 0

        density = rho0 * np.exp(-altitude / h_scale)

        # Drag force
        speed = np.linalg.norm(velocity)
        if speed == 0:
            return np.zeros(3)

        drag_force = -0.5 * density * speed * cd * area * velocity / speed

        # Acceleration
        acceleration = drag_force / mass

        return acceleration

    @staticmethod
    def j2_perturbation_acceleration(position: np.ndarray) -> np.ndarray:
        """
        Calculate J2 perturbation acceleration.

        Args:
            position: Position vector [x, y, z] (m)

        Returns:
            J2 acceleration vector [ax, ay, az] (m/s^2)
        """
        x, y, z = position
        r = np.linalg.norm(position)

        if r == 0:
            return np.zeros(3)

        # J2 acceleration components
        factor = 3 * J2 * MU_EARTH * EARTH_RADIUS**2 / (2 * r**5)

        ax = factor * x * (1 - 5 * z**2 / r**2)
        ay = factor * y * (1 - 5 * z**2 / r**2)
        az = factor * z * (3 - 5 * z**2 / r**2)

        return np.array([ax, ay, az])

    @staticmethod
    def solar_radiation_pressure_acceleration(position: np.ndarray,
                                           velocity: np.ndarray,
                                           mass: float = 1.0,
                                           area: float = 1.0,
                                           reflectivity: float = 1.4) -> np.ndarray:
        """
        Calculate solar radiation pressure acceleration.

        Args:
            position: Position vector [x, y, z] (m)
            velocity: Velocity vector [vx, vy, vz] (m/s)
            mass: Mass (kg)
            area: Cross-sectional area (m^2)
            reflectivity: Reflectivity coefficient

        Returns:
            SRP acceleration vector [ax, ay, az] (m/s^2)
        """
        # Simplified: assume sun at origin, constant solar flux
        solar_constant = 1366  # W/m^2
        c = 299792458  # Speed of light (m/s)
        au = 149597870700  # Astronomical unit (m)

        # Solar flux at 1 AU
        solar_flux = solar_constant / c

        # Direction to sun (simplified)
        sun_direction = -position / np.linalg.norm(position)

        # SRP force
        srp_force = solar_flux * reflectivity * area * sun_direction

        # Acceleration
        acceleration = srp_force / mass

        return acceleration

    @staticmethod
    def lambert_solver(position1: np.ndarray, position2: np.ndarray,
                      time_of_flight: float, mu: float = MU_EARTH) -> Optional[np.ndarray]:
        """
        Solve Lambert's problem for orbital transfer.

        Args:
            position1: Initial position [x, y, z] (m)
            position2: Final position [x, y, z] (m)
            time_of_flight: Time of flight (s)
            mu: Gravitational parameter (m^3/s^2)

        Returns:
            Initial velocity vector or None if no solution
        """
        # Simplified Lambert solver (placeholder)
        # Real implementation would use universal variables or other methods

        r1 = np.linalg.norm(position1)
        r2 = np.linalg.norm(position2)

        # Angular separation
        cos_angle = np.dot(position1, position2) / (r1 * r2)
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        # Semi-major axis estimate
        a = (r1 + r2) / 2

        # Time of flight for parabolic orbit
        t_parabolic = np.pi / np.sqrt(mu) * (a**(3/2))

        if time_of_flight < t_parabolic:
            # Elliptical orbit
            # Simplified calculation
            v1 = np.sqrt(mu * (2 / r1 - 1 / a))
            v2 = np.sqrt(mu * (2 / r2 - 1 / a))

            # Direction (simplified)
            direction = (position2 - position1) / np.linalg.norm(position2 - position1)
            velocity = direction * v1

            return velocity
        else:
            return None  # No solution for given time


# Convenience functions
def calculate_collision_probability(miss_distance: float, relative_velocity: float,
                                  combined_radius: float = 100.0) -> float:
    """
    Calculate collision probability.

    Args:
        miss_distance: Miss distance (m)
        relative_velocity: Relative velocity (m/s)
        combined_radius: Combined radius (m)

    Returns:
        Collision probability
    """
    return PhysicsUtils.collision_probability_cylindrical(
        miss_distance, relative_velocity, combined_radius
    )


def orbital_elements_to_cartesian(a: float, e: float, i: float, raan: float,
                                arg_p: float, true_anomaly: float,
                                mu: float = MU_EARTH) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert orbital elements to Cartesian position and velocity.

    Args:
        a: Semi-major axis (m)
        e: Eccentricity
        i: Inclination (rad)
        raan: Right ascension of ascending node (rad)
        arg_p: Argument of perigee (rad)
        true_anomaly: True anomaly (rad)
        mu: Gravitational parameter

    Returns:
        Tuple of (position, velocity) vectors
    """
    # Position in orbital plane
    r = a * (1 - e**2) / (1 + e * np.cos(true_anomaly))
    x_orb = r * np.cos(true_anomaly)
    y_orb = r * np.sin(true_anomaly)
    z_orb = 0

    # Velocity in orbital plane
    h = np.sqrt(mu * a * (1 - e**2))  # Angular momentum
    vx_orb = - (mu / h) * np.sin(true_anomaly)
    vy_orb = (mu / h) * (e + np.cos(true_anomaly))
    vz_orb = 0

    # Rotation matrices
    # Rotate by argument of perigee
    x_argp = x_orb * np.cos(arg_p) - y_orb * np.sin(arg_p)
    y_argp = x_orb * np.sin(arg_p) + y_orb * np.cos(arg_p)
    z_argp = z_orb

    vx_argp = vx_orb * np.cos(arg_p) - vy_orb * np.sin(arg_p)
    vy_argp = vx_orb * np.sin(arg_p) + vy_orb * np.cos(arg_p)
    vz_argp = vz_orb

    # Rotate by inclination
    x_inc = x_argp
    y_inc = y_argp * np.cos(i) - z_argp * np.sin(i)
    z_inc = y_argp * np.sin(i) + z_argp * np.cos(i)

    vx_inc = vx_argp
    vy_inc = vy_argp * np.cos(i) - vz_argp * np.sin(i)
    vz_inc = vy_argp * np.sin(i) + vz_argp * np.cos(i)

    # Rotate by RAAN
    x = x_inc * np.cos(raan) - y_inc * np.sin(raan)
    y = x_inc * np.sin(raan) + y_inc * np.cos(raan)
    z = z_inc

    vx = vx_inc * np.cos(raan) - vy_inc * np.sin(raan)
    vy = vx_inc * np.sin(raan) + vy_inc * np.cos(raan)
    vz = vz_inc

    position = np.array([x, y, z])
    velocity = np.array([vx, vy, vz])

    return position, velocity


if __name__ == "__main__":
    # Example usage
    print("Physics Utilities Test")

    # Test gravitational acceleration
    pos = np.array([7000000, 0, 0])  # 7000 km altitude
    acc = PhysicsUtils.gravitational_acceleration(pos)
    print(f"Gravitational acceleration at {pos/1000} km: {acc} m/s^2")

    # Test orbital period
    a = 7000000  # 7000 km semi-major axis
    period = PhysicsUtils.orbital_period(a)
    print(f"Orbital period for a={a/1000} km: {period/3600:.2f} hours")

    # Test collision probability
    pc = PhysicsUtils.collision_probability_cylindrical(1000, 10)  # 1km miss, 10 m/s relative
    print(f"Collision probability: {pc:.6f}")

    # Test Hohmann transfer
    a1, a2 = 7000000, 8000000  # 7000 km to 8000 km
    dv1, dv2 = PhysicsUtils.hohmann_transfer_delta_v(a1, a2)
    print(f"Hohmann transfer delta-V: {dv1:.2f} m/s, {dv2:.2f} m/s")