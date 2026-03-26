"""
Trajectory Visualization for Satellite Collision Prediction
"""

import matplotlib.pyplot as plt
import numpy as np
from skyfield.api import load
from datetime import timedelta


def visualize_pair(sat1, sat2, time_steps=10, step_minutes=5):
    """
    Plot 3D trajectories of two satellites
    """

    ts = load.timescale()
    t0 = ts.now()

    traj1 = []
    traj2 = []

    for step in range(time_steps):

        t = ts.utc(t0.utc_datetime() + timedelta(minutes=step * step_minutes))

        s1 = sat1.at(t)
        s2 = sat2.at(t)

        traj1.append(s1.position.km)
        traj2.append(s2.position.km)

    traj1 = np.array(traj1)
    traj2 = np.array(traj2)

    # ---- plot ----
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(traj1[:, 0], traj1[:, 1], traj1[:, 2], label=sat1.name)
    ax.plot(traj2[:, 0], traj2[:, 1], traj2[:, 2], label=sat2.name)

    ax.scatter(traj1[0, 0], traj1[0, 1], traj1[0, 2], marker='o')
    ax.scatter(traj2[0, 0], traj2[0, 1], traj2[0, 2], marker='x')

    ax.set_title("Satellite Trajectories")
    ax.legend()

    plt.show()