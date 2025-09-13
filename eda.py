"""Ferramentas de análise exploratória para a base Spotify."""

import csv
from collections import Counter
from math import sqrt
from typing import Dict
import kagglehub
import os
import pandas as pd


NUMERIC_COLUMNS = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]


def _update_stats(stats: Dict[str, Dict[str, float]], row: Dict[str, str]) -> None:
    """Atualiza métricas estatísticas para cada coluna numérica.

    Os campos analisados incluem contagem de valores, soma, soma dos quadrados,
    mínimos e máximos. Valores ausentes são contabilizados separadamente.
    """
    for col in NUMERIC_COLUMNS:
        value = row.get(col)
        if value in (None, ""):
            stats[col]["missing"] += 1
            continue
        num = float(value)
        col_stats = stats[col]
        col_stats["count"] += 1
        col_stats["sum"] += num
        col_stats["sum_sq"] += num * num
        col_stats["min"] = min(col_stats["min"], num)
        col_stats["max"] = max(col_stats["max"], num)


def explore_dataset(csv_path: str) -> None:
    """Executa análise exploratória básica na base de músicas do Spotify.

    A função lê o arquivo CSV linha a linha e imprime estatísticas resumidas
    das colunas numéricas, distribuição de gêneros e proporção de faixas
    explícitas.
    """
    stats = {
        col: {
            "count": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": float("inf"),
            "max": float("-inf"),
            "missing": 0,
        }
        for col in NUMERIC_COLUMNS
    }
    genre_counter: Counter[str] = Counter()
    explicit_counter: Counter[str] = Counter()

    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            genre_counter[row["track_genre"]] += 1
            explicit_counter[row["explicit"]] += 1
            _update_stats(stats, row)

    for col, col_stats in stats.items():
        count = col_stats["count"]
        if count:
            mean = col_stats["sum"] / count
            variance = col_stats["sum_sq"] / count - mean * mean
            std = sqrt(variance)
        else:
            mean = std = 0.0
        print(
            f"{col}: count={count} missing={col_stats['missing']} "
            f"mean={mean:.2f} std={std:.2f} min={col_stats['min']:.2f} "
            f"max={col_stats['max']:.2f}"
        )

    print("\nTop 10 gêneros:")
    for genre, qtd in genre_counter.most_common(10):
        print(f"{genre}: {qtd}")

    print("\nFaixas explícitas:")
    for label, qtd in explicit_counter.items():
        print(f"{label}: {qtd}")


if __name__ == "__main__":
    # Download latest version
    path = kagglehub.dataset_download("amitanshjoshi/spotify-1million-tracks")

    df = pd.read_csv(os.path.join(path, "spotify_data.csv"))
    explore_dataset = pd.read_csv(os.path.join(path, "spotify_data.csv"))
