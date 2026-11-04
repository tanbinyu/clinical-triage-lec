"""Public reporting API that avoids external mutation of plotting globals."""
from pipeline.reporting.plot_metrics import main as _render_metrics_figures


def render_metrics_figures(metrics_json, figure_dir):
    _render_metrics_figures(metrics_json=metrics_json, fig_dir=figure_dir)
