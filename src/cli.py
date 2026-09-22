"""Командный интерфейс для запуска симуляций."""

import os

import click
import matplotlib
matplotlib.use("Agg")  # без окна
import matplotlib.pyplot as plt

from src.model import rs
from src.simulate import run_all_scenarios
from src.visualize import (
    plot_fire_dynamics,
    plot_neighborhood_comparison,
    plot_vegetation_dynamics,
    plot_burn_rates,
)


@click.group()
def cli():
    """CLI для модели лесного пожара."""
    pass


@cli.command()
@click.option("--width", default=100, help="Ширина поля")
@click.option("--height", default=100, help="Высота поля")
@click.option("--steps", default=200, help="Число шагов")
@click.option("--output", default="results", help="Папка для результатов")
@click.option("--seed", default=1097, help="Seed генератора")
def run(width, height, steps, output, seed):
    """Запускает все сценарии и сохраняет графики."""
    rs.seed(seed)
    
    click.echo(f"Запуск симуляции: {width}×{height}, {steps} шагов")
    
    os.makedirs(output, exist_ok=True)
    
    results = run_all_scenarios(
        width=width,
        height=height,
        time_steps=steps,
    )
    
    click.echo(f"Сценариев: {len(results)}")
    
    # Графики
    plots = {
        "dynamics.png": plot_fire_dynamics(results),
        "comparison.png": plot_neighborhood_comparison(results),
        "vegetation.png": plot_vegetation_dynamics(results),
        "burn_rates.png": plot_burn_rates(results),
    }
    
    for name, fig in plots.items():
        path = os.path.join(output, name)
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        click.echo(f"  Сохранён: {path}")
    
    # Сводка
    click.echo("\n=== Сводка ===")
    for (nt, f), st in results.items():
        click.echo(
            f"  {nt.value} f={f}: "
            f"max_hot={max(st.a_f)}, "
            f"hot_end={st.a_f[-1]}"
        )


if __name__ == "__main__":
    cli()