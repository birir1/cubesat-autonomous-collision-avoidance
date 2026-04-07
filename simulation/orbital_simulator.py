import numpy as np

MU_EARTH = 398600.4418  # km^3/s^2

class OrbitalSimulator:
    def __init__(self, dt=1.0, total_time=1000):
        self.dt = dt
        self.total_time = total_time

    def gravitational_acceleration(self, position):
        """
        Compute acceleration due to Earth's gravity
        """
        r = np.linalg.norm(position)
        acc = -MU_EARTH * position / (r**3 + 1e-8)
        return acc

    def propagate_orbit(self, position, velocity):
        """
        Simple numerical integration (Euler method)
        """
        positions = []
        velocities = []

        pos = position.copy()
        vel = velocity.copy()

        for _ in range(int(self.total_time / self.dt)):
            acc = self.gravitational_acceleration(pos)

            # Update state
            vel = vel + acc * self.dt
            pos = pos + vel * self.dt

            positions.append(pos.copy())
            velocities.append(vel.copy())

        return np.array(positions), np.array(velocities)

    def simulate_two_satellites(self, state1, state2):
        """
        Simulate relative motion between two satellites
        """
        pos1, vel1 = state1
        pos2, vel2 = state2

        traj1_pos, traj1_vel = self.propagate_orbit(pos1, vel1)
        traj2_pos, traj2_vel = self.propagate_orbit(pos2, vel2)

        relative_positions = traj1_pos - traj2_pos
        relative_velocities = traj1_vel - traj2_vel

        return relative_positions, relative_velocities


if __name__ == "__main__":
    simulator = OrbitalSimulator(dt=1.0, total_time=500)

    # Example initial states (km, km/s)
    state1 = (
        np.array([7000, 0, 0]),
        np.array([0, 7.5, 0])
    )

    state2 = (
        np.array([7005, 0, 0]),
        np.array([0, 7.4, 0])
    )

    rel_pos, rel_vel = simulator.simulate_two_satellites(state1, state2)

    print("Relative position shape:", rel_pos.shape)
    print("Relative velocity shape:", rel_vel.shape)