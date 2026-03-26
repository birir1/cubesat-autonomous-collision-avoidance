import os

BASE_DIR = "."

structure = {
    "configs": [
        "dataset_config.yaml",
        "model_config.yaml",
        "training_config.yaml",
        "simulation_config.yaml",
        "rl_config.yaml"
    ],

    "evaluation": [
        "benchmark_models.py"
    ],

    "phases/phase2_vision_object_detection/models": [
        "yolov8_detector.py",
        "efficientdet_detector.py"
    ],

    "phases/phase2_vision_object_detection/training": [
        "train_detector.py"
    ],

    "phases/phase2_vision_object_detection/evaluation": [
        "evaluate_detector.py"
    ],

    "phases/phase2_vision_object_detection/dataset": [
        "space_object_dataset.py"
    ],

    "phases/phase2_vision_object_detection/outputs": [],

    "phases/phase3_object_tracking/models": [
        "kalman_tracker.py",
        "deep_sort_tracker.py"
    ],

    "phases/phase3_object_tracking/scripts": [
        "run_tracking.py"
    ],

    "phases/phase3_object_tracking/outputs": [],

    "phases/phase5_maneuver_planning_rl": [
        "reward_function.py"
    ],

    "utils": [
        "orbital_mechanics.py",
        "coordinate_transforms.py",
        "collision_geometry.py",
        "experiment_tracker.py"
    ],

    "visualization": [
        "plot_orbit_3d.py",
        "plot_collision_probability.py",
        "plot_rl_policy_behavior.py"
    ],

    "experiments": [
        "run_experiment.py"
    ],

    "docs/paper_drafts/figures": [],

    ".": [
        "main_pipeline.py",
        "requirements.txt",
        "run.sh",
        "README.md"
    ]
}


def create_structure():
    for folder, files in structure.items():

        folder_path = os.path.join(BASE_DIR, folder)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            print(f"Created folder: {folder_path}")

        for file in files:
            file_path = os.path.join(folder_path, file)

            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    f.write(f"# {file}\n")
                print(f"Created file: {file_path}")
            else:
                print(f"Already exists: {file_path}")


if __name__ == "__main__":
    print("\nCreating missing project structure...\n")
    create_structure()
    print("\nDone! ")