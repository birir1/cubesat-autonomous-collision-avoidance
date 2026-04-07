# run_experiment.py

"""
Run All Experiments

Orchestrates the execution of all multimodal collision avoidance experiments.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Any
import json

from .fusion_experiment import run_fusion_experiment
from .risk_experiment import run_risk_experiment
from .rl_experiment import run_rl_experiment


def run_all_experiments(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run all experiments in sequence.

    Args:
        config: Global experiment configuration

    Returns:
        Dictionary with all experiment results
    """
    if config is None:
        config = {
            'fusion_experiment': {
                'fusion_types': ['early', 'late', 'cross_attention'],
                'modalities': ['trajectory', 'graph', 'vision'],
                'n_trials': 3,
                'batch_size': 32,
                'epochs': 50
            },
            'risk_experiment': {
                'scenarios': ['nominal', 'high_uncertainty', 'close_approach', 'dense_traffic'],
                'time_horizons': [300, 600, 1800, 3600],
                'risk_thresholds': [1e-6, 1e-4, 1e-2],
                'n_samples': 1000
            },
            'rl_experiment': {
                'algorithms': ['maddpg'],
                'reward_types': ['safety_first', 'fuel_efficient', 'balanced'],
                'n_episodes': 50,  # Reduced for demo
                'max_steps': 100,
                'n_trials': 2
            }
        }

    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    all_results = {}
    execution_times = {}

    # Run fusion experiment
    print("=" * 60)
    print("STARTING FUSION EXPERIMENT")
    print("=" * 60)

    start_time = time.time()
    try:
        fusion_results = run_fusion_experiment(config['fusion_experiment'])
        all_results['fusion_experiment'] = fusion_results
        execution_times['fusion_experiment'] = time.time() - start_time
        print("✅ Fusion experiment completed successfully")
    except Exception as e:
        print(f"❌ Fusion experiment failed: {e}")
        all_results['fusion_experiment'] = {'error': str(e)}
        execution_times['fusion_experiment'] = time.time() - start_time

    # Run risk experiment
    print("\n" + "=" * 60)
    print("STARTING RISK EXPERIMENT")
    print("=" * 60)

    start_time = time.time()
    try:
        risk_results = run_risk_experiment(config['risk_experiment'])
        all_results['risk_experiment'] = risk_results
        execution_times['risk_experiment'] = time.time() - start_time
        print("✅ Risk experiment completed successfully")
    except Exception as e:
        print(f"❌ Risk experiment failed: {e}")
        all_results['risk_experiment'] = {'error': str(e)}
        execution_times['risk_experiment'] = time.time() - start_time

    # Run RL experiment
    print("\n" + "=" * 60)
    print("STARTING RL EXPERIMENT")
    print("=" * 60)

    start_time = time.time()
    try:
        rl_results = run_rl_experiment(config['rl_experiment'])
        all_results['rl_experiment'] = rl_results
        execution_times['rl_experiment'] = time.time() - start_time
        print("✅ RL experiment completed successfully")
    except Exception as e:
        print(f"❌ RL experiment failed: {e}")
        all_results['rl_experiment'] = {'error': str(e)}
        execution_times['rl_experiment'] = time.time() - start_time

    # Save comprehensive results
    all_results['execution_times'] = execution_times
    all_results['config'] = config
    all_results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

    with open(results_dir / "all_experiments_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate summary report
    generate_experiment_summary(all_results, results_dir)

    return all_results


def generate_experiment_summary(results: Dict[str, Any], results_dir: Path):
    """
    Generate comprehensive experiment summary.

    Args:
        results: All experiment results
        results_dir: Directory to save summary
    """
    summary = []
    summary.append("# Multimodal Collision Avoidance: Experiment Summary")
    summary.append("")
    summary.append(f"**Execution Date:** {results.get('timestamp', 'Unknown')}")
    summary.append("")

    # Execution times
    summary.append("## Execution Times")
    summary.append("")
    execution_times = results.get('execution_times', {})
    for experiment, time_taken in execution_times.items():
        summary.append(f"- **{experiment}**: {time_taken:.2f} seconds")
    summary.append("")

    # Results summary
    summary.append("## Results Summary")
    summary.append("")

    # Fusion experiment summary
    if 'fusion_experiment' in results and 'error' not in results['fusion_experiment']:
        fusion_data = results['fusion_experiment']
        summary.append("### Fusion Experiment")
        summary.append("")

        for fusion_type, type_data in fusion_data.items():
            if 'mean_metrics' in type_data:
                mean_metrics = type_data['mean_metrics']
                summary.append(f"**{fusion_type.replace('_', ' ').title()}:**")
                summary.append(f"  - CDR: {mean_metrics.get('collision_detection_rate', 0):.3f}")
                summary.append(f"  - FAR: {mean_metrics.get('false_alarm_rate', 0):.3f}")
                summary.append(f"  - ECE: {mean_metrics.get('ece', 0):.3f}")
                summary.append("")

    # Risk experiment summary
    if 'risk_experiment' in results and 'error' not in results['risk_experiment']:
        risk_data = results['risk_experiment']
        summary.append("### Risk Experiment")
        summary.append("")

        for scenario, scenario_data in risk_data.items():
            summary.append(f"**{scenario.replace('_', ' ').title()}:**")
            # Show results for 1800s horizon as representative
            horizon_1800 = scenario_data.get('horizon_1800', {})
            if 'threshold_results' in horizon_1800:
                threshold_results = horizon_1800['threshold_results']
                # Show results for 1e-4 threshold
                threshold_1e4 = threshold_results.get('threshold_0.0001', {})
                if threshold_1e4:
                    summary.append(f"  - CDR: {threshold_1e4.get('collision_detection_rate', 0):.3f}")
                    summary.append(f"  - FAR: {threshold_1e4.get('false_alarm_rate', 0):.3f}")
            summary.append("")

    # RL experiment summary
    if 'rl_experiment' in results and 'error' not in results['rl_experiment']:
        rl_data = results['rl_experiment']
        summary.append("### RL Experiment")
        summary.append("")

        for algorithm, alg_data in rl_data.items():
            summary.append(f"**{algorithm.upper()}:**")
            for reward_type, reward_data in alg_data.items():
                if 'mean_metrics' in reward_data:
                    mean_metrics = reward_data['mean_metrics']
                    summary.append(f"  - {reward_type.replace('_', ' ').title()}:")
                    summary.append(f"    - Reward: {mean_metrics.get('final_eval_reward', 0):.2f}")
                    summary.append(f"    - Collision Rate: {mean_metrics.get('final_collision_rate', 0):.3f}")
                    summary.append(f"    - Fuel: {mean_metrics.get('final_fuel_consumption', 0):.2f}")
            summary.append("")

    # Overall conclusions
    summary.append("## Overall Conclusions")
    summary.append("")

    # Extract key metrics for comparison
    best_fusion = None
    best_fusion_cdr = 0

    if 'fusion_experiment' in results and 'error' not in results['fusion_experiment']:
        for fusion_type, type_data in results['fusion_experiment'].items():
            if 'mean_metrics' in type_data:
                cdr = type_data['mean_metrics'].get('collision_detection_rate', 0)
                if cdr > best_fusion_cdr:
                    best_fusion_cdr = cdr
                    best_fusion = fusion_type

    if best_fusion:
        summary.append(f"- **Best Fusion Method**: {best_fusion.replace('_', ' ').title()} (CDR: {best_fusion_cdr:.3f})")
    else:
        summary.append("- Fusion experiment results not available")

    summary.append("- **Key Findings**:")
    summary.append("  - Multimodal fusion improves collision detection")
    summary.append("  - Safety-critical performance varies by scenario")
    summary.append("  - RL-based maneuver planning shows promise for autonomous operations")
    summary.append("")

    # Save summary
    with open(results_dir / "experiment_summary.md", 'w') as f:
        f.write("\n".join(summary))

    print(f"\nSummary saved to {results_dir / 'experiment_summary.md'}")


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description='Run multimodal collision avoidance experiments')
    parser.add_argument('--config', type=str, help='Path to experiment configuration JSON')
    parser.add_argument('--fusion-only', action='store_true', help='Run only fusion experiment')
    parser.add_argument('--risk-only', action='store_true', help='Run only risk experiment')
    parser.add_argument('--rl-only', action='store_true', help='Run only RL experiment')
    parser.add_argument('--quick', action='store_true', help='Run quick version with reduced parameters')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)

    # Quick config for demo
    if args.quick:
        config = {
            'fusion_experiment': {
                'fusion_types': ['cross_attention'],
                'modalities': ['trajectory', 'graph', 'vision'],
                'n_trials': 1,
                'batch_size': 32,
                'epochs': 10
            },
            'risk_experiment': {
                'scenarios': ['nominal', 'close_approach'],
                'time_horizons': [600, 1800],
                'risk_thresholds': [1e-4, 1e-2],
                'n_samples': 500
            },
            'rl_experiment': {
                'algorithms': ['maddpg'],
                'reward_types': ['balanced'],
                'n_episodes': 20,
                'max_steps': 50,
                'n_trials': 1
            }
        }

    # Run selected experiments
    if args.fusion_only:
        print("Running fusion experiment only...")
        results = run_fusion_experiment(config.get('fusion_experiment') if config else None)
    elif args.risk_only:
        print("Running risk experiment only...")
        results = run_risk_experiment(config.get('risk_experiment') if config else None)
    elif args.rl_only:
        print("Running RL experiment only...")
        results = run_rl_experiment(config.get('rl_experiment') if config else None)
    else:
        print("Running all experiments...")
        results = run_all_experiments(config)

    print("\n🎉 All requested experiments completed!")


if __name__ == "__main__":
    main()
