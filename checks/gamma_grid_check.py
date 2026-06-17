from cfm_mppi.visualization.gamma_sweep_data import build_gamma_grid


def test_build_gamma_grid_includes_bounds_and_sorts():
    grid = build_gamma_grid([0.7, -0.2, 1.2, 0.3], count=5)
    assert grid[0] == 0.0
    assert grid[-1] == 1.0
    assert grid == sorted(grid)
    assert all(0.0 <= g <= 1.0 for g in grid)
