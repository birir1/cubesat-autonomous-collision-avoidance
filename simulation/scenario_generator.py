import numpy as np

class ScenarioGenerator:
    def __init__(self, num_samples=1000, collision_threshold=0.1):
        self.num_samples = num_samples
        self.collision_threshold = collision_threshold

    def generate_relative_state(self):
        """
        Generate synthetic relative position (km) and velocity (km/s)
        """
        # Position: +/- 10 km range
        rel_pos = np.random.uniform(-10, 10, size=3)

        # Velocity: +/- 1 km/s range
        rel_vel = np.random.uniform(-1, 1, size=3)

        return rel_pos, rel_vel

    def compute_risk_score(self, rel_pos, rel_vel):
        """
        Simple proxy for collision risk:
        - Smaller distance + converging velocity → higher risk
        """
        distance = np.linalg.norm(rel_pos)
        speed = np.linalg.norm(rel_vel)

        # Direction alignment (approaching vs separating)
        cos_theta = np.dot(rel_pos, rel_vel) / (distance * speed + 1e-8)

        # Risk heuristic
        risk = (1 / (distance + 1e-3)) * (1 - cos_theta)

        return risk

    def label_risk(self, risk):
        """
        Binary classification label
        """
        return 1 if risk >= self.collision_threshold else 0

    def generate_dataset(self):
        """
        Generate dataset:
        Features = [rx, ry, rz, vx, vy, vz]
        Labels = collision risk (0 or 1)
        """
        X = []
        y = []

        for _ in range(self.num_samples):
            rel_pos, rel_vel = self.generate_relative_state()
            risk = self.compute_risk_score(rel_pos, rel_vel)
            label = self.label_risk(risk)

            features = np.concatenate([rel_pos, rel_vel])

            X.append(features)
            y.append(label)

        return np.array(X), np.array(y)


if __name__ == "__main__":
    generator = ScenarioGenerator(num_samples=5000)
    X, y = generator.generate_dataset()

    print("Dataset shape:", X.shape)
    print("Labels distribution:", np.bincount(y))