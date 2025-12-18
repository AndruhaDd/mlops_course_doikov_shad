# scripts/run_experiments.py
import os
from src.train import train_one_run


def main() -> None:
    data_path = os.getenv("DATA_PATH", "data/raw/city_day.csv")

    runs = []

    # Эксперимент 1: Dataset v1 + Ridge (alpha=1.0)
    run_id_1 = train_one_run(
        data_path=data_path,
        dataset_version="v1",
        model_type="ridge",
        random_state=42,
        alpha=1.0,
    )
    runs.append(("exp1_v1_ridge_a1", run_id_1))

    # Эксперимент 2: Dataset v2 + Ridge (alpha=10.0)
    run_id_2 = train_one_run(
        data_path=data_path,
        dataset_version="v2",
        model_type="ridge",
        random_state=42,
        alpha=10.0,
    )
    runs.append(("exp2_v2_ridge_a10", run_id_2))

    # Эксперимент 3: Dataset v2 + RandomForest
    run_id_3 = train_one_run(
        data_path=data_path,
        dataset_version="v2",
        model_type="rf",
        random_state=42,
        n_estimators=500,
        max_depth=12,
    )
    runs.append(("exp3_v2_rf_500_d12", run_id_3))

    print("\n=== DONE. Run IDs ===")
    for name, rid in runs:
        print(f"{name}: {rid}")

    print("\nSet for inference (example):")
    print(f"MODEL_RUN_ID={run_id_3}")
    print(f"TARGET_COL={os.getenv('TARGET_COL', 'AQI')}")


if __name__ == "__main__":
    main()
