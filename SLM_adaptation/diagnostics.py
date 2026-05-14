"""Diagnostics and visualization utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def diagnostics(
    df_clients: pd.DataFrame,
    available_train_df: pd.DataFrame,
    availability_df: pd.DataFrame,
    client_metadata: pd.DataFrame,
) -> None:
    """Print basic diagnostics for data and availability composition."""
    total_rows = len(df_clients)
    avail_rows = int((available_train_df["available"] == 1).sum())
    unavail_rows = total_rows - avail_rows

    print("\n=== Diagnostics ===")
    print(f"Total samples: {total_rows:,}")
    print(f"Available samples: {avail_rows:,} ({avail_rows / max(total_rows, 1):.2%})")
    print(f"Unavailable samples: {unavail_rows:,} ({unavail_rows / max(total_rows, 1):.2%})")

    per_round = availability_df.groupby("round_id")["available"].sum()
    print("Available clients per round:")
    print(per_round.to_string())

    print("\nPersona counts:")
    print(df_clients["persona"].value_counts().to_string())

    print("\nClient timezone distribution:")
    print(client_metadata["timezone_group"].value_counts().to_string())


def plot_availability_heatmap(availability_df: pd.DataFrame, output_path: Path) -> None:
    pivot = availability_df.pivot(index="client_id", columns="round_id", values="available")

    plt.figure(figsize=(12, 6))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Available (1) / Unavailable (0)")
    plt.xlabel("Round")
    plt.ylabel("Client ID")
    plt.title("Client Availability Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_available_clients_per_round(availability_df: pd.DataFrame, output_path: Path) -> None:
    counts = availability_df.groupby("round_id")["available"].sum()

    plt.figure(figsize=(10, 4))
    counts.plot(kind="line", marker="o")
    plt.xlabel("Round")
    plt.ylabel("Number of Available Clients")
    plt.title("Available Clients Per Round")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_persona_distribution(
    df_clients: pd.DataFrame,
    available_train_df: pd.DataFrame,
    output_path: Path,
) -> None:
    total_counts = df_clients["persona"].value_counts().sort_index()
    avail_counts = available_train_df[available_train_df["available"] == 1]["persona"].value_counts().sort_index()

    index = sorted(set(total_counts.index).union(set(avail_counts.index)))
    total_vals = [total_counts.get(i, 0) for i in index]
    avail_vals = [avail_counts.get(i, 0) for i in index]

    x = range(len(index))
    plt.figure(figsize=(12, 5))
    plt.bar(x, total_vals, width=0.45, label="All samples")
    plt.bar([i + 0.45 for i in x], avail_vals, width=0.45, label="Available samples")
    plt.xticks([i + 0.225 for i in x], index, rotation=25, ha="right")
    plt.ylabel("Samples")
    plt.title("Persona Distribution: Full vs Available Training Stream")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
