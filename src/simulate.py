"""Модуль для запуска симуляций лесного пожара."""

import numpy as np

from src.model import (
    CellState,
    NeighborhoodType,
    Statistics,
    create_ca,
    init_state,
    simulate,
    rs,  # чтобы сбросить seed
)


def run_scenario(width, height, time_steps, vegetation_distribution,
                 neighborhood_type, n_ignition, ignition_points=None):
    """
    Запускает один сценарий симуляции.
    
    Параметры:
    - width, height: размеры поля
    - time_steps: число шагов
    - vegetation_distribution: dict {CellState: доля}
    - neighborhood_type: NeighborhoodType
    - n_ignition: число очагов
    - ignition_points: список координат или None
    
    Возвращает:
    - Statistics
    """
    ca = create_ca(width, height)
    points = init_state(ca, vegetation_distribution, n_ignition, ignition_points)
    st = simulate(ca, neighborhood_type, time_steps)
    st.ignition_points = points  # сохраним для отчёта
    return st


def run_all_scenarios(width=100, height=100, time_steps=200,
                      vegetation_distribution=None, ignition_counts=(1, 3, 9)):
    """
    Запускает все 6 сценариев (2 окрестности × 3 очага).
    
    Возвращает:
    - dict {(NeighborhoodType, f): Statistics}
    """
    if vegetation_distribution is None:
        vegetation_distribution = {
            CellState.CONIFEROUS: 0.3,
            CellState.DECIDUOUS: 0.4,
            CellState.SHRUB: 0.1,
        }
    
    results = {}
    
    for nt in (NeighborhoodType.CROSS, NeighborhoodType.NEUMAN):
        for f in ignition_counts:
            # Сбросить seed для воспроизводимости
            rs.seed(1097)
            
            st = run_scenario(width, height, time_steps,
                              vegetation_distribution, nt, f)
            results[(nt, f)] = st
    
    return results