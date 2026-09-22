"""Модуль клеточного автомата для модели лесного пожара."""

from enum import Enum
import numpy as np

# Глобальный генератор случайных чисел
rs = np.random.RandomState(seed=1097)

# Глобальные параметры модели
P_GROWTH = 0.02        # вероятность роста новой растительности
P_LIGHTNING = 2e-5     # вероятность удара молнии


class CellState(Enum):
    """Возможные состояния клетки."""
    EMPTY = 0        # пустая / сгоревшая
    FIRING = 1       # горит
    CONIFEROUS = 2   # хвойное дерево
    DECIDUOUS = 3    # лиственное дерево
    SHRUB = 4        # кустарник


class NeighborhoodType(Enum):
    """Типы окрестностей."""
    CROSS = "+"      # окрестность фон Неймана (4 соседа)
    NEUMAN = "x"     # окрестность Мура (8 соседей)


class VegetationType:
    """Свойства типа растительности."""
    
    def __init__(self, ignition_bonus, burn_time, spread_chance, color, name):
        """..."""
        self.ignition_bonus = ignition_bonus
        self.burn_time = burn_time
        self.spread_chance = spread_chance
        self.color = color
        self.name = name
    
    def __repr__(self):
        return f"VegetationType({self.name})"


# Параметры для каждого типа растительности
VEGETATION_TYPES = {
    CellState.CONIFEROUS: VegetationType(0.5, 2, 0.8, "#228B22", "Хвойные"),
    CellState.DECIDUOUS:  VegetationType(0.0, 3, 0.6, "#32CD32", "Лиственные"),
    CellState.SHRUB:      VegetationType(1.0, 1, 0.4, "#90EE90", "Кустарник"),
}


def create_ca(width, height):
    """Создаёт пустое поле width × height."""
    return np.zeros((height, width), dtype=int)


def init_state(ca, vegetation_distribution, n_ignition, ignition_points=None):
    """..."""
    h, w = ca.shape
    
    all_cells = [(i, j) for i in range(h) for j in range(w)]
    rs.shuffle(all_cells)
    
    idx = 0
    for veg_state, proportion in vegetation_distribution.items():
        n_cells = int(h * w * proportion)
        for _ in range(n_cells):
            i, j = all_cells[idx]
            ca[i, j] = veg_state.value
            idx += 1
    
    vegetation_cells = []
    for i in range(h):
        for j in range(w):
            if ca[i, j] in (
                CellState.CONIFEROUS.value,
                CellState.DECIDUOUS.value,
                CellState.SHRUB.value,
            ):
                vegetation_cells.append((i, j))
    
    if ignition_points is None:
        n_ignition = min(n_ignition, len(vegetation_cells))
        indices = rs.choice(len(vegetation_cells), size=n_ignition, replace=False)
        ignition_points = [vegetation_cells[i] for i in indices]
    
    for i, j in ignition_points:
        ca[i, j] = CellState.FIRING.value
    
    return ignition_points


def get_neighbors(cell, shape, neighborhood_type):
    """
    Возвращает список соседей клетки.
    
    Параметры:
    - cell: координаты (i, j)
    - shape: размеры поля (h, w)
    - neighborhood_type: NeighborhoodType.CROSS или NEUMAN
    
    Возвращает:
    - список координат [(ni, nj), ...]
    """
    i, j = cell
    h, w = shape
    
    if neighborhood_type == NeighborhoodType.CROSS:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:  # NEUMAN
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
    
    neighbors = []
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < h and 0 <= nj < w:
            neighbors.append((ni, nj))
    
    return neighbors


class Statistics:
    """Сбор статистики по ходу симуляции."""
    
    def __init__(self):
        """Инициализация пустых списков."""
        self.t = []
        self.a_f = []
        self.a_c = []
        self.a_d = []
        self.a_s = []
        self.a_e = []
        self.states = []
        self.burning_time = {}
    
    def append(self, t, ca):
        """
        Добавить статистику на шаге t.
        
        Параметры:
        - t: номер шага
        - ca: текущее поле
        """
        self.t.append(t)
        self.a_f.append(int(np.sum(ca == CellState.FIRING.value)))
        self.a_c.append(int(np.sum(ca == CellState.CONIFEROUS.value)))
        self.a_d.append(int(np.sum(ca == CellState.DECIDUOUS.value)))
        self.a_s.append(int(np.sum(ca == CellState.SHRUB.value)))
        self.a_e.append(int(np.sum(ca == CellState.EMPTY.value)))
        self.states.append(np.copy(ca))

def update_cell(ca, cell, neighbors, burning_time):
    """
    Определяет новое состояние клетки на основе текущего состояния и соседей.
    
    Параметры:
    - ca: текущее поле
    - cell: координаты (i, j)
    - neighbors: список соседей
    - burning_time: словарь {(i,j): шагов_горит}
    
    Возвращает:
    - новое состояние клетки (CellState.value)
    """
    i, j = cell
    current = ca[i, j]
    if current == CellState.FIRING.value:
        # Увеличить счётчик горения
        burning_time[cell] = burning_time.get(cell, 0) + 1
        
        # Определить тип растительности (для burn_time)
        # Упрощение: считаем, что горит хвойное
        veg_type = VEGETATION_TYPES[CellState.CONIFEROUS]
        
        if burning_time[cell] >= veg_type.burn_time:
            return CellState.EMPTY.value
        return CellState.FIRING.value
    elif current in (
        CellState.CONIFEROUS.value,
        CellState.DECIDUOUS.value,
        CellState.SHRUB.value,
    ):
        veg_state = CellState(current)
        veg_type = VEGETATION_TYPES[veg_state]
        
        # Посчитать горящих соседей
        firing_neighbors = 0
        for ni, nj in neighbors:
            if ca[ni, nj] == CellState.FIRING.value:
                firing_neighbors += 1
        
        # Заражение от соседей
        if firing_neighbors > 0:
            if rs.random() < veg_type.spread_chance:
                return CellState.FIRING.value
        
        # Удар молнии
        if rs.random() < P_LIGHTNING * (1 + veg_type.ignition_bonus):
            return CellState.FIRING.value
        
        return current
    elif current == CellState.EMPTY.value:
        if rs.random() < P_GROWTH:
            veg_types = [
                CellState.CONIFEROUS,
                CellState.DECIDUOUS,
                CellState.SHRUB,
            ]
            return rs.choice(veg_types, p=[0.4, 0.4, 0.2]).value
        return CellState.EMPTY.value