"""Compatibility bridge for the established beta=0.01 LaTeX table generator."""
import runpy
from pathlib import Path


def generate_beta001_tables(project_root):
    """Run the existing table algorithm without duplicating its 700-line layout code."""
    script = Path(project_root) / "result" / "_tools" / "gen_tables_beta001.py"
    runpy.run_path(str(script), run_name="__main__")
