"""Тесты для модуля model."""

import numpy as np
import pytest

from src.model import (
    CellState,
    NeighborhoodType,
    VEGETATION_TYPES,
    create_ca,
    init_state,
    get_neighbors,
    simulate,
)


def test_create_ca_shape():
    ca = create_ca(10, 20)
    assert ca.shape == (20, 10)
    assert ca.dtype == int


def test_create_ca_empty():
    ca = create_ca(5, 5)
    assert np.all(ca == CellState.EMPTY.value)


def test_get_neighbors_cross_center():
    n = get_neighbors((2, 2), (5, 5), NeighborhoodType.CROSS)
    assert len(n) == 4


def test_get_neighbors_cross_corner():
    n = get_neighbors((0, 0), (5, 5), NeighborhoodType.CROSS)
    assert len(n) == 2


def test_get_neighbors_neuman_center():
    n = get_neighbors((2, 2), (5, 5), NeighborhoodType.NEUMAN)
    assert len(n) == 8


def test_get_neighbors_neuman_corner():
    n = get_neighbors((0, 0), (5, 5), NeighborhoodType.NEUMAN)
    assert len(n) == 3


def test_init_state_creates_ignition():
    ca = create_ca(10, 10)
    dist = {CellState.CONIFEROUS: 0.5}
    points = init_state(ca, dist, 2)
    assert len(points) == 2
    for i, j in points:
        assert ca[i, j] == CellState.FIRING.value


def test_simulate_returns_statistics():
    ca = create_ca(10, 10)
    dist = {CellState.CONIFEROUS: 0.5}
    init_state(ca, dist, 1)
    st = simulate(ca, NeighborhoodType.CROSS, 5)
    assert len(st.t) == 6  # 0 + 5 шагов
    assert len(st.a_f) == 6
    assert len(st.states) == 6