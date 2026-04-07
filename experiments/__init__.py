from .fusion_experiment import run_fusion_experiment
from .risk_experiment import run_risk_experiment


# Lazy import to avoid hard dependency
def run_rl_experiment(*args, **kwargs):
    from .rl_experiment import run_rl_experiment as _run
    return _run(*args, **kwargs)


from .trajectory_experiment import run_trajectory_experiment
from .run_experiment import run_all_experiments


__all__ = [
    'run_fusion_experiment',
    'run_risk_experiment',
    'run_trajectory_experiment',
    'run_all_experiments'
]

def run_all_experiments(*args, **kwargs):
    from .run_experiment import run_all_experiments as _run
    return _run(*args, **kwargs)