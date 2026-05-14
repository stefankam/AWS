from __future__ import annotations
from pathlib import Path
import random
import numpy as np

import config
from data_loader import load_financial_dataset, normalize_finance_df
from client_personas import create_clients
from temporal_drift import add_temporal_rounds
from availability import create_availability_matrix
from export_data import create_available_training_stream, export_client_round_data
from diagnostics import diagnostics, plot_availability_heatmap, plot_available_clients_per_round, plot_persona_distribution


def main():
    random.seed(config.SEED); np.random.seed(config.SEED)
    out = Path(config.OUTPUT_DIR); md = out / "metadata"; pl = out / "plots"
    md.mkdir(parents=True, exist_ok=True); pl.mkdir(parents=True, exist_ok=True)

    df = load_financial_dataset(config.DATASET_NAME)
    df = normalize_finance_df(df)
    df_clients, client_metadata = create_clients(df, num_clients=config.NUM_CLIENTS, seed=config.SEED)
    df_clients = add_temporal_rounds(df_clients, num_rounds=config.NUM_ROUNDS)
    availability_df = create_availability_matrix(client_metadata, num_rounds=config.NUM_ROUNDS, seed=config.SEED)
    available_df, unavailable_df = create_available_training_stream(df_clients, availability_df)

    export_client_round_data(available_df, out)

    full = df_clients.merge(availability_df[["client_id", "round_id", "available", "availability_probability"]], on=["client_id", "round_id"], how="left")
    client_metadata.to_csv(md / "client_metadata.csv", index=False)
    availability_df.to_csv(md / "availability_matrix.csv", index=False)
    full.to_csv(md / "full_client_dataset.csv", index=False)
    available_df.to_csv(md / "available_train_dataset.csv", index=False)
    unavailable_df.to_csv(md / "unavailable_train_dataset.csv", index=False)

    diagnostics(full, full, availability_df, client_metadata)
    plot_availability_heatmap(availability_df, pl / "availability_heatmap.png")
    plot_available_clients_per_round(availability_df, pl / "available_clients_per_round.png")
    plot_persona_distribution(full, full, pl / "persona_distribution.png")

if __name__ == "__main__":
    main()
