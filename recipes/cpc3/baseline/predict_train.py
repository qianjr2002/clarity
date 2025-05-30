"""Make intelligibility predictions from HASPI scores."""

# (clarity)qclarity/recipes/cpc3/baseline$ python predict_train.py dataset=clarity_demo_data split=train
# [2025-05-30 15:39:39,196][__main__][INFO] - Loading dataset...
# [2025-05-30 15:39:39,199][__main__][INFO] - Processing signal components...
# [2025-05-30 15:39:39,203][__main__][INFO] - Starting predictions...
# 100%|██████| 30/30 [00:00<00:00, 55.15it/s]
# [2025-05-30 15:39:39,751][__main__][INFO] - Predictions saved to clarity_demo_data.train.predict.csv

# (clarity)qclarity/recipes/cpc3/baseline$ python predict_train.py dataset=clarity_data split=train
# [2025-05-30 16:13:50,973][__main__][INFO] - Loading dataset...
# [2025-05-30 16:13:51,108][__main__][INFO] - Processing signal components...
# [2025-05-30 16:13:51,122][__main__][INFO] - Starting predictions...
# 100%|██████| 15520/15520 [04:54<00:00, 52.71it/s]
# [2025-05-30 16:18:45,581][__main__][INFO] - Predictions saved to clarity_data.train.predict.csv

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from shared_predict_utils import ( # recipes.cpc3.baseline.
    LogisticModel,
    load_dataset_with_haspi,
    make_disjoint_train_set,
)

log = logging.getLogger(__name__)


# Define constants for column names
SIGNAL_COL = "signal"
PREDICTED_COL = "predicted_correctness"
HEADER = ["signal_ID", "intelligibility_score"]


# pylint: disable = no-value-for-parameter
@hydra.main(config_path=".", config_name="config", version_base=None)
def predict_train(cfg: DictConfig):
    """Predict intelligibility from HASPI scores."""

    # Load the intelligibility dataset records with HASPI scores
    log.info("Loading dataset...")
    records_df = load_dataset_with_haspi(cfg, "train")

    # Split signal into components
    log.info("Processing signal components...")
    signal_parts = records_df[SIGNAL_COL].str.split("_", expand=True)
    records_df["system"] = signal_parts[0] + "_" + signal_parts[1]
    records_df["scene"] = signal_parts[2]
    records_df["listener"] = signal_parts[3]

    output_file = f"{cfg.dataset}.train.predict.csv"

    # Make predictions for each item in the data
    log.info("Starting predictions...")
    for i, _ in tqdm(records_df.iterrows(), total=len(records_df)):
        test_df = records_df.iloc[[i]].copy()
        # Create a disjoint training set
        train_df = make_disjoint_train_set(records_df, test_df)

        model = LogisticModel()
        model.fit(train_df.haspi_score, train_df.correctness)

        # Predict correctness from HASPI scores using the model
        test_df[PREDICTED_COL] = model.predict(test_df.haspi_score)

        # Save results to CSV file
        mode = "w" if i == 0 else "a"
        test_df[[SIGNAL_COL, PREDICTED_COL]].to_csv(
            output_file,
            index=False,
            header=HEADER if i == 0 else False,
            mode=mode,
        )

    log.info(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    predict_train()
