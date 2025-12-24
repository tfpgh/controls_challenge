from pathlib import Path

import numpy as np
import optuna

from offline.config import PGTOConfig
from offline.pgto.optimizer import PGTOOptimizer
from offline.segment import load_segment

HORIZONS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


def objective(trial: optuna.Trial) -> float:
    """Test PGTO parameters on representative files and return mean cost."""
    SEGMENTS = [
        "00432",
        "03061",
        "03935",
        "02472",
        "04699",
        "02896",
        "04185",
        "02591",
        "00692",
        "00219",
    ]

    params = {
        "horizon": trial.suggest_categorical("horizon", HORIZONS),
    }
    print(f"Testing: {params}")

    try:
        config = PGTOConfig(
            horizon=params["horizon"],
        )
        optimizer = PGTOOptimizer(config)

        total = 0.0
        for i, seg_id in enumerate(SEGMENTS):
            segment = load_segment(Path(f"data/{seg_id}.csv"), config)
            result = optimizer.optimize(segment, verbose=False)
            total += float(np.mean([r.cost for r in result.restarts]))

            # Report running average
            avg_so_far = total / (i + 1)
            trial.report(avg_so_far, step=i)
            print(f"  Segment {seg_id}: avg so far = {avg_so_far:.2f}")

            if trial.should_prune():
                raise optuna.TrialPruned()

        return total / len(SEGMENTS)

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial failed: {e}")
        raise optuna.TrialPruned()


def main():
    sampler = optuna.samplers.GridSampler(
        search_space={
            "horizon": HORIZONS,
        }
    )

    study = optuna.create_study(
        sampler=sampler,
        storage="sqlite:///optuna_pgto_study.db",
        study_name="pgto_hyperparameter_search_horizons",
        direction="minimize",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=16)


if __name__ == "__main__":
    main()
