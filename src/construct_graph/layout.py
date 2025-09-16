from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


def compute_layouts(constructs: List[dict], relationships: List[dict]) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Compute two deterministic layouts: embedding-like and centrality-based.

    The goal is to keep a simple, dependency-light layout that is stable across runs.
    """
    names = [c.get('name') for c in constructs if c.get('name')]
    unique_names = list(dict.fromkeys(names))

    # Degree centrality
    degree: Dict[str, int] = {n: 0 for n in unique_names}
    for r in relationships:
        s = r.get('source_construct')
        t = r.get('target_construct')
        if s in degree:
            degree[s] += 1
        if t in degree:
            degree[t] += 1

    # Embedding-like layout: golden-angle spiral for readability
    rng = np.random.RandomState(42)
    golden_angle = np.pi * (3 - np.sqrt(5.0))
    r_min, r_max = 150.0, 450.0
    embedding_positions: Dict[str, dict] = {}
    for i, n in enumerate(unique_names):
        radius = r_min + (r_max - r_min) * (i / max(1, len(unique_names) - 1))
        angle = (i * golden_angle) % (2 * np.pi)
        # small jitter to avoid perfect overlaps
        jitter = rng.uniform(-8.0, 8.0, size=2)
        embedding_positions[n] = {
            'x': float(radius * np.cos(angle) + jitter[0]),
            'y': float(radius * np.sin(angle) + jitter[1]),
        }

    # Centrality-based radial layout (higher degree toward center)
    max_degree = max(degree.values()) if degree else 1
    centrality_positions: Dict[str, dict] = {}
    ordered = sorted(unique_names, key=lambda n: (-degree.get(n, 0), n))
    for i, n in enumerate(ordered):
        d = degree.get(n, 0)
        norm = 1.0 - (d / max_degree) if max_degree > 0 else 1.0
        eased = norm ** 0.85
        radius = r_min + eased * (r_max - r_min)
        angle = (i * golden_angle) % (2 * np.pi)
        centrality_positions[n] = {
            'x': float(radius * np.cos(angle)),
            'y': float(radius * np.sin(angle)),
        }

    # Ensure all nodes present
    for n in unique_names:
        centrality_positions.setdefault(n, {'x': 0.0, 'y': 0.0})
        embedding_positions.setdefault(n, {'x': 0.0, 'y': 0.0})

    return embedding_positions, centrality_positions



