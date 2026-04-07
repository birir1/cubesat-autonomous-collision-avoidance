"""
Data Processing for Multimodal Satellite Collision Prediction

This module handles the creation and preprocessing of multimodal datasets
combining trajectory data, graph structures, and visual observations.
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import os
import yaml
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from PIL import Image
import cv2


class MultimodalDataProcessor:
    """
    Processes and creates multimodal datasets for satellite collision prediction.
    """

    def __init__(self, config_path=None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self.get_default_config()

        self.scalers = {}

    def get_default_config(self):
        """Default configuration for data processing."""
        return {
            'sequence_length': 20,
            'num_satellites': 10,
            'communication_range': 500.0,  # km
            'image_size': (224, 224),
            'feature_scaler': 'standard',
            'augmentation': {
                'noise_factor': 0.01,
                'rotation_range': 10,
                'brightness_range': [0.8, 1.2]
            }
        }

    def create_trajectory_sequences(self, raw_trajectory_data, sequence_length=20):
        """
        Create temporal sequences from trajectory data.

        Args:
            raw_trajectory_data: (num_samples, time_steps, 6) - [x,y,z,vx,vy,vz]
            sequence_length: length of sequences to create

        Returns:
            sequences: (num_sequences, sequence_length, 6)
        """
        num_samples, total_steps, features = raw_trajectory_data.shape

        sequences = []
        for i in range(num_samples):
            traj = raw_trajectory_data[i]  # (total_steps, 6)

            # Create sliding windows
            for start in range(0, total_steps - sequence_length + 1, sequence_length // 2):
                end = start + sequence_length
                seq = traj[start:end]  # (sequence_length, 6)
                sequences.append(seq)

        return np.array(sequences)

    def create_graph_data(self, positions, velocities, communication_range=500.0):
        """
        Create graph data from satellite positions and velocities.

        Args:
            positions: (num_samples, num_satellites, 3)
            velocities: (num_samples, num_satellites, 3)
            communication_range: maximum distance for edge creation

        Returns:
            graph_data_list: list of PyG Data objects
        """
        graph_data_list = []

        for i in tqdm(range(len(positions)), desc="Creating graph data"):
            pos = positions[i]  # (num_satellites, 3)
            vel = velocities[i]  # (num_satellites, 3)

            # Create node features: [x, y, z, vx, vy, vz]
            node_features = np.concatenate([pos, vel], axis=1)

            # Create edges based on communication range
            num_satellites = pos.shape[0]
            edges = []

            for j in range(num_satellites):
                for k in range(j + 1, num_satellites):
                    distance = np.linalg.norm(pos[j] - pos[k])
                    if distance <= communication_range:
                        edges.append([j, k])
                        edges.append([k, j])  # Undirected graph

            if edges:
                edge_index = np.array(edges).T
                edge_attr = self.compute_edge_attributes(pos, vel, edge_index)
            else:
                # No edges - create self-loops or empty
                edge_index = np.array([[i for i in range(num_satellites)],
                                      [i for i in range(num_satellites)]])
                edge_attr = np.zeros((num_satellites, 3))

            # Create PyG Data object
            data = Data(
                x=torch.FloatTensor(node_features),
                edge_index=torch.LongTensor(edge_index),
                edge_attr=torch.FloatTensor(edge_attr),
                pos=torch.FloatTensor(pos),
                vel=torch.FloatTensor(vel)
            )

            graph_data_list.append(data)

        return graph_data_list

    def compute_edge_attributes(self, positions, velocities, edge_index):
        """
        Compute edge attributes: relative distance, velocity, and angle.

        Args:
            positions: (num_satellites, 3)
            velocities: (num_satellites, 3)
            edge_index: (2, num_edges)

        Returns:
            edge_attr: (num_edges, 3)
        """
        num_edges = edge_index.shape[1]
        edge_attr = []

        for e in range(num_edges):
            i, j = edge_index[0, e], edge_index[1, e]

            # Relative position and distance
            rel_pos = positions[j] - positions[i]
            distance = np.linalg.norm(rel_pos)

            # Relative velocity
            rel_vel = velocities[j] - velocities[i]
            rel_speed = np.linalg.norm(rel_vel)

            # Angle between relative position and velocity
            if distance > 0 and rel_speed > 0:
                cos_angle = np.dot(rel_pos, rel_vel) / (distance * rel_speed)
            else:
                cos_angle = 0.0

            edge_attr.append([
                distance / self.config['communication_range'],  # normalized distance
                rel_speed / 10.0,  # normalized relative speed
                cos_angle
            ])

        return np.array(edge_attr)

    def process_vision_data(self, image_paths, augment=True):
        """
        Process and augment vision data.

        Args:
            image_paths: list of paths to images
            augment: whether to apply data augmentation

        Returns:
            processed_images: list of processed PIL images
        """
        processed_images = []

        for img_path in tqdm(image_paths, desc="Processing vision data"):
            # Load image
            image = Image.open(img_path).convert('RGB')

            # Resize
            image = image.resize(self.config['image_size'])

            if augment:
                image = self.augment_image(image)

            processed_images.append(image)

        return processed_images

    def augment_image(self, image):
        """
        Apply data augmentation to image.

        Args:
            image: PIL Image

        Returns:
            augmented_image: PIL Image
        """
        # Convert to numpy for augmentation
        img_array = np.array(image)

        # Random rotation
        if np.random.random() < 0.5:
            angle = np.random.uniform(-self.config['augmentation']['rotation_range'],
                                     self.config['augmentation']['rotation_range'])
            h, w = img_array.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            img_array = cv2.warpAffine(img_array, M, (w, h))

        # Random brightness
        if np.random.random() < 0.5:
            factor = np.random.uniform(*self.config['augmentation']['brightness_range'])
            img_array = np.clip(img_array * factor, 0, 255).astype(np.uint8)

        # Add noise
        if np.random.random() < 0.5:
            noise = np.random.normal(0, self.config['augmentation']['noise_factor'] * 255,
                                   img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def create_collision_labels(self, trajectories, positions, risk_threshold=0.1):
        """
        Create collision risk labels from trajectory and position data.

        Args:
            trajectories: (num_samples, sequence_length, 6)
            positions: (num_samples, num_satellites, 3)
            risk_threshold: threshold for positive collision risk

        Returns:
            labels: (num_samples,) binary labels
        """
        labels = []

        for i in range(len(trajectories)):
            # Simple risk calculation based on minimum distance
            pos = positions[i]
            min_distance = np.min([
                np.linalg.norm(pos[j] - pos[k])
                for j in range(len(pos))
                for k in range(j + 1, len(pos))
            ])

            # Risk based on distance (closer = higher risk)
            risk = 1 / (1 + np.exp(-(1000 - min_distance) / 200))

            # Also consider velocity convergence
            vel = trajectories[i, -1, 3:6]  # Final velocity
            speed = np.linalg.norm(vel)
            risk *= (1 + speed / 10.0)  # Higher speed = higher risk

            label = 1 if risk > risk_threshold else 0
            labels.append(label)

        return np.array(labels)

    def fit_scalers(self, trajectory_data, position_data, velocity_data):
        """
        Fit feature scalers on training data.

        Args:
            trajectory_data: (num_samples, sequence_length, 6)
            position_data: (num_samples, num_satellites, 3)
            velocity_data: (num_samples, num_satellites, 3)
        """
        # Flatten for scaling
        traj_flat = trajectory_data.reshape(-1, 6)
        pos_flat = position_data.reshape(-1, 3)
        vel_flat = velocity_data.reshape(-1, 3)

        # Fit scalers
        self.scalers['trajectory'] = StandardScaler()
        self.scalers['position'] = StandardScaler()
        self.scalers['velocity'] = StandardScaler()

        self.scalers['trajectory'].fit(traj_flat)
        self.scalers['position'].fit(pos_flat)
        self.scalers['velocity'].fit(vel_flat)

    def transform_features(self, trajectory_data, position_data, velocity_data):
        """
        Transform features using fitted scalers.

        Args:
            Same as fit_scalers

        Returns:
            transformed data
        """
        # Apply scaling
        traj_shape = trajectory_data.shape
        pos_shape = position_data.shape
        vel_shape = velocity_data.shape

        trajectory_scaled = self.scalers['trajectory'].transform(
            trajectory_data.reshape(-1, 6)
        ).reshape(traj_shape)

        position_scaled = self.scalers['position'].transform(
            position_data.reshape(-1, 3)
        ).reshape(pos_shape)

        velocity_scaled = self.scalers['velocity'].transform(
            velocity_data.reshape(-1, 3)
        ).reshape(vel_shape)

        return trajectory_scaled, position_scaled, velocity_scaled

    def save_processed_data(self, output_dir, train_data, val_data=None, test_data=None):
        """
        Save processed multimodal data to disk.

        Args:
            output_dir: directory to save data
            train_data: dict with 'trajectory', 'positions', 'velocities', 'labels', 'graphs', 'images'
            val_data: same as train_data
            test_data: same as train_data
        """
        os.makedirs(output_dir, exist_ok=True)

        splits = [('train', train_data)]
        if val_data:
            splits.append(('val', val_data))
        if test_data:
            splits.append(('test', test_data))

        for split_name, data in splits:
            # Save numpy arrays
            np.save(os.path.join(output_dir, f'{split_name}_trajectory.npy'), data['trajectory'])
            np.save(os.path.join(output_dir, f'{split_name}_positions.npy'), data['positions'])
            np.save(os.path.join(output_dir, f'{split_name}_velocities.npy'), data['velocities'])
            np.save(os.path.join(output_dir, f'{split_name}_labels.npy'), data['labels'])

            # Save graph data
            if 'graphs' in data:
                torch.save(data['graphs'], os.path.join(output_dir, f'{split_name}_graphs.pt'))

            # Save vision data (if available)
            if 'images' in data and data['images']:
                # For now, save as list - in practice, might want to save as tensors
                torch.save(data['images'], os.path.join(output_dir, f'{split_name}_images.pt'))

        # Save scalers
        torch.save(self.scalers, os.path.join(output_dir, 'scalers.pt'))

        # Save config
        with open(os.path.join(output_dir, 'data_config.yaml'), 'w') as f:
            yaml.dump(self.config, f)

        print(f"Processed data saved to {output_dir}")


def create_synthetic_multimodal_dataset(config_path=None, output_dir='data/synthetic_multimodal'):
    """
    Create a complete synthetic multimodal dataset.
    """
    processor = MultimodalDataProcessor(config_path)

    # Generate synthetic trajectory data
    num_samples = 10000
    sequence_length = processor.config['sequence_length']
    num_satellites = processor.config['num_satellites']

    print("Generating synthetic trajectory data...")
    trajectories = []
    positions = []
    velocities = []

    for _ in tqdm(range(num_samples)):
        # Generate orbital trajectory
        t = np.linspace(0, 4*np.pi, sequence_length)
        altitude = 6371 + np.random.uniform(400, 800)  # LEO altitude

        # Orbital motion
        x = altitude * np.cos(t) + np.random.normal(0, 50, sequence_length)
        y = altitude * np.sin(t) + np.random.normal(0, 50, sequence_length)
        z = np.random.normal(0, 100, sequence_length)

        # Velocities (orbital speed ~7.8 km/s for LEO)
        orbital_speed = 7.8
        vx = -orbital_speed * np.sin(t) + np.random.normal(0, 0.5, sequence_length)
        vy = orbital_speed * np.cos(t) + np.random.normal(0, 0.5, sequence_length)
        vz = np.random.normal(0, 0.5, sequence_length)

        traj = np.stack([x, y, z, vx, vy, vz], axis=1)
        trajectories.append(traj)

        # Current positions and velocities (end of sequence)
        pos = np.array([x[-1], y[-1], z[-1]])
        vel = np.array([vx[-1], vy[-1], vz[-1]])

        # Generate multiple satellites
        sat_positions = []
        sat_velocities = []
        for _ in range(num_satellites):
            # Add some satellites at different positions
            offset = np.random.uniform(-200, 200, 3)
            sat_pos = pos + offset
            sat_vel = vel + np.random.uniform(-1, 1, 3)
            sat_positions.append(sat_pos)
            sat_velocities.append(sat_vel)

        positions.append(np.array(sat_positions))
        velocities.append(np.array(sat_velocities))

    trajectories = np.array(trajectories)
    positions = np.array(positions)
    velocities = np.array(velocities)

    # Create sequences
    trajectory_sequences = processor.create_trajectory_sequences(trajectories)

    # Adjust positions and velocities to match sequence count
    num_sequences = len(trajectory_sequences)
    positions = positions[:num_sequences]
    velocities = velocities[:num_sequences]

    # Fit scalers
    processor.fit_scalers(trajectory_sequences, positions, velocities)

    # Transform features
    traj_scaled, pos_scaled, vel_scaled = processor.transform_features(
        trajectory_sequences, positions, velocities
    )

    # Create graph data
    print("Creating graph structures...")
    graph_data = processor.create_graph_data(pos_scaled, vel_scaled)

    # Create labels
    labels = processor.create_collision_labels(traj_scaled, pos_scaled)

    # Split data
    train_size = int(0.7 * num_sequences)
    val_size = int(0.15 * num_sequences)

    train_data = {
        'trajectory': traj_scaled[:train_size],
        'positions': pos_scaled[:train_size],
        'velocities': vel_scaled[:train_size],
        'labels': labels[:train_size],
        'graphs': graph_data[:train_size]
    }

    val_data = {
        'trajectory': traj_scaled[train_size:train_size+val_size],
        'positions': pos_scaled[train_size:train_size+val_size],
        'velocities': vel_scaled[train_size:train_size+val_size],
        'labels': labels[train_size:train_size+val_size],
        'graphs': graph_data[train_size:train_size+val_size]
    }

    test_data = {
        'trajectory': traj_scaled[train_size+val_size:],
        'positions': pos_scaled[train_size+val_size:],
        'velocities': vel_scaled[train_size+val_size:],
        'labels': labels[train_size+val_size:],
        'graphs': graph_data[train_size+val_size:]
    }

    # Save processed data
    processor.save_processed_data(output_dir, train_data, val_data, test_data)

    print(f"Created multimodal dataset with {num_sequences} samples")
    print(f"Class distribution: {np.bincount(labels.astype(int))}")

    return output_dir


if __name__ == "__main__":
    # Create synthetic multimodal dataset
    output_dir = create_synthetic_multimodal_dataset()
    print(f"Dataset created in: {output_dir}")