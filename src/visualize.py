"""Модуль визуализации результатов симуляции."""

import io
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.model import CellState, NeighborhoodType, VEGETATION_TYPES


# Цвета для поля: EMPTY, FIRING, CONIFEROUS, DECIDUOUS, SHRUB
COLORS = ["#49423D", "orange", "#228B22", "#32CD32", "#90EE90"]
CMAP_FOREST = ListedColormap(COLORS)


def plot_fire_dynamics(results, title="Динамика лесного пожара"):
    """Строит графики числа горящих деревьев во времени."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, nt in zip(axes, (NeighborhoodType.CROSS, NeighborhoodType.NEUMAN)):
        for f in (1, 3, 9):
            st = results[(nt, f)]
            ax.plot(st.t, st.a_f, linewidth=2, label=f"f = {f}")

        ax.set_xlabel("Время (шаги)")
        ax.set_ylabel("Горящих деревьев")
        ax.set_title(f"Окрестность {nt.value}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_neighborhood_comparison(results):
    """Сравнивает окрестности CROSS и NEUMAN для f=1."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for nt, color in ((NeighborhoodType.CROSS, "#FF6B6B"),
                      (NeighborhoodType.NEUMAN, "#4ECDC4")):
        st = results[(nt, 1)]
        ax.plot(st.t, st.a_f, color=color, linewidth=2,
                label=f"{nt.value} окрестность")

    ax.set_xlabel("Время (шаги)")
    ax.set_ylabel("Горящих деревьев")
    ax.set_title("Сравнение окрестностей (f = 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_vegetation_dynamics(results, nt=NeighborhoodType.CROSS, f=1):
    """Строит графики по типам растительности для одного сценария."""
    fig, ax = plt.subplots(figsize=(10, 5))

    st = results[(nt, f)]
    ax.plot(st.t, st.a_c, color=VEGETATION_TYPES[CellState.CONIFEROUS].color,
            linewidth=2, label="Хвойные")
    ax.plot(st.t, st.a_d, color=VEGETATION_TYPES[CellState.DECIDUOUS].color,
            linewidth=2, label="Лиственные")
    ax.plot(st.t, st.a_s, color=VEGETATION_TYPES[CellState.SHRUB].color,
            linewidth=2, label="Кустарник")
    ax.plot(st.t, st.a_f, color="red", linewidth=2, linestyle="--",
            label="Горящие")

    ax.set_xlabel("Время (шаги)")
    ax.set_ylabel("Количество клеток")
    ax.set_title(f"Динамика растительности ({nt.value}, f = {f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_burn_rates(results):
    """Столбчатая диаграмма: % сгоревших по сценариям."""
    scenarios = []
    burn_c, burn_d, burn_s = [], [], []

    for nt in (NeighborhoodType.CROSS, NeighborhoodType.NEUMAN):
        for f in (1, 3, 9):
            st = results[(nt, f)]
            scenarios.append(f"{nt.value} f={f}")

            for arr, a0, a1 in ((burn_c, st.a_c[0], st.a_c[-1]),
                                (burn_d, st.a_d[0], st.a_d[-1]),
                                (burn_s, st.a_s[0], st.a_s[-1])):
                if a0 > 0:
                    arr.append((a0 - a1) / a0 * 100)
                else:
                    arr.append(0)

    x = np.arange(len(scenarios))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, burn_c, width, label="Хвойные",
           color=VEGETATION_TYPES[CellState.CONIFEROUS].color)
    ax.bar(x, burn_d, width, label="Лиственные",
           color=VEGETATION_TYPES[CellState.DECIDUOUS].color)
    ax.bar(x + width, burn_s, width, label="Кустарник",
           color=VEGETATION_TYPES[CellState.SHRUB].color)

    ax.set_xlabel("Сценарий")
    ax.set_ylabel("Сгорело (%)")
    ax.set_title("Процент сгоревшей растительности по сценариям")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_fire_gif_and_frame(results, nt, f=1,
                              gif_path="results/fire_animation.gif",
                              frame_path="results/fire_frame.png",
                              frame_step=100, fps=10):
    """Создаёт GIF-анимацию пожара + один кадр как PNG.

    Параметры:
    - results: dict {(NeighborhoodType, f): Statistics}
    - nt: тип окрестности
    - f: число очагов
    - gif_path: путь для GIF
    - frame_path: путь для PNG-кадра
    - frame_step: номер шага, который сохранится как PNG
    - fps: кадров в секунду
    """
    import imageio.v2 as imageio

    st = results[(nt, f)]
    frames = []

    print(f"[visualize] Создание GIF ({len(st.states)} кадров)...")

    for t, ca in zip(st.t, st.states):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(ca, cmap=CMAP_FOREST, vmin=0, vmax=4,
                  interpolation="nearest")
        ax.set_title(f"Шаг {t}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=70, bbox_inches="tight")
        buf.seek(0)
        frame = imageio.imread(buf)
        buf.close()
        plt.close(fig)
        frames.append(frame)

    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print(f"[visualize] GIF сохранён: {gif_path}")

    # Сохраняем один кадр как PNG
    if frame_step in st.t:
        idx = st.t.index(frame_step)
        ca = st.states[idx]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(ca, cmap=CMAP_FOREST, vmin=0, vmax=4,
                  interpolation="nearest")
        ax.set_title(f"Окрестность {nt.value}, f = {f}, шаг {frame_step}",
                     fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[visualize] Кадр сохранён: {frame_path}")
    else:
        print(f"[visualize] Шаг {frame_step} не найден среди {st.t[:5]}...")