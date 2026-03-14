"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based.

Pipeline order:
1. config, 2. pull_CRSP, 3. clean_CRSP, 4. pull_ravenpack, 5. clean_ravenpack,
6. pull_TAQ, 7. clean_TAQ, 8. label_headlines, 9. data_exploration,
10-11. Table1/Figure5 replication (2021.10.1-2024.5.31),
12-13. Table1/Figure5 (2024.5.31-2024.12.31), 14-15. Table1/Figure5 full (2021-2024.12.31),
16. latex_pdf
"""

#######################################
## Configuration and Helpers for PyDoit
#######################################
## Make sure the src folder is in the path
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(1, "./src/")

from os import environ
from settings import config


def _run_python(script, env_extra=None):
    """Run a Python script with extra env vars (cross-platform)."""

    def _action():
        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)
        subprocess.run(
            [sys.executable, f"src/{script}"],
            env=env,
            cwd=Path(__file__).resolve().parent,
            check=True,
        )

    return _action

DOIT_CONFIG = {"backend": "sqlite3", "dep_file": "./.doit-db.sqlite"}


BASE_DIR = config("BASE_DIR")
DATA_DIR = config("DATA_DIR")
MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")
OS_TYPE = config("OS_TYPE")
USER = config("USER")

## Helpers for handling Jupyter Notebook tasks
environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
def jupyter_execute_notebook(notebook_path):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
def jupyter_to_html(notebook_path, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir={output_dir} {notebook_path}"
def jupyter_to_md(notebook_path, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --output-dir={output_dir} {notebook_path}"
def jupyter_clear_output(notebook_path):
    """Clear the output of a notebook"""
    return f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
# fmt: on


def mv(from_path, to_path):
    """Move a file to a folder"""
    from_path = Path(from_path)
    to_path = Path(to_path)
    to_path.mkdir(parents=True, exist_ok=True)
    if OS_TYPE == "nix":
        command = f"mv {from_path} {to_path}"
    else:
        command = f"move {from_path} {to_path}"
    return command


def copy_file(origin_path, destination_path, mkdir=True):
    """Create a Python action for copying a file."""

    def _copy_file():
        origin = Path(origin_path)
        dest = Path(destination_path)
        if mkdir:
            dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, dest)

    return _copy_file


##################################
## Begin rest of PyDoit tasks here
##################################


def task_config():
    """Create empty directories for data and output if they don't exist"""
    return {
        "actions": ["ipython ./src/settings.py"],
        "targets": [DATA_DIR, OUTPUT_DIR],
        "file_dep": ["./src/settings.py"],
        "clean": [],
    }


def task_pull_CRSP_stock():
    """Pull CRSP data from WRDS and save to disk"""

    return {
        "actions": [
            "python src/settings.py",
            "python src/pull_CRSP_stock.py",
        ],
        "targets": [
            Path(DATA_DIR) / "CRSP_daily_stock.parquet",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/pull_CRSP_stock.py",
        ],
        "verbosity": 2,  # Print everything immediately. This is important in
        # case WRDS asks for credentials.
    }


def task_clean_crsp_daily():
    """
    Clean CRSP daily stock data:
    _data/CRSP_daily_stock.parquet -> _data/clean/crsp_daily.parquet
    """
    raw_path = DATA_DIR / "CRSP_daily_stock.parquet"
    out_path = DATA_DIR / "clean" / "crsp_daily.parquet"

    return {
        "actions": [
            "python src/clean_crsp_daily.py",
        ],
        "file_dep": [
            "src/clean_crsp_daily.py",
            "src/settings.py",
            str(raw_path),
        ],
        "targets": [
            str(out_path),
        ],
        "verbosity": 2,
    }


def task_pull_ravenpack_dj():
    """Pull Ravenpack data from WRDS and save to disk"""

    return {
        "actions": [
            "python src/settings.py",
            "python src/pull_ravenpack_dj.py",
        ],
        "targets": [
            Path(DATA_DIR) / "ravenpack_dj_equities.parquet",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/pull_ravenpack_dj.py",
        ],
        "verbosity": 2,  # Print everything immediately. This is important in
        # case WRDS asks for credentials.
    }



def task_clean_ravenpack():
    """
    Clean RavenPack news data.

    _data/ravenpack_dj_equities.parquet
        -> _data/clean/news_firmday.parquet
        -> _data/clean/ravenpack_intraday_story.parquet
    """

    raw_path = DATA_DIR / "ravenpack_dj_equities.parquet"
    out_firmday = DATA_DIR / "clean" / "news_firmday.parquet"
    out_story = DATA_DIR / "clean" / "ravenpack_intraday_story.parquet"

    return {
        "actions": [
            "python src/clean_ravenpack_firmday.py",
        ],
        "file_dep": [
            "src/clean_ravenpack_firmday.py",
            "src/settings.py",
            str(raw_path),
        ],
        "targets": [
            str(out_firmday),
            str(out_story),
        ],
        "verbosity": 2,
    }


def task_pull_TAQ_intraday():
    """Pull TAQ NBBO intraday data from WRDS (2021-2025). Needs WRDS + run on login node."""
    return {
        "actions": [
            "python src/settings.py",
            "python src/pull_TAQ_intraday.py --intraday --workers 1",
        ],
        "task_dep": ["clean_ravenpack", "clean_crsp_daily"],
        "targets": [str(DATA_DIR / "taqm_nbbo")],
        "file_dep": [
            "src/settings.py",
            "src/pull_TAQ_intraday.py",
        ],
        "verbosity": 2,
    }


def task_clean_taq_nbbo_minute():
    """Combine per-date TAQ NBBO parquets -> taq_nbbo_minute.parquet"""
    out_path = DATA_DIR / "clean" / "taq_nbbo_minute.parquet"
    return {
        "actions": ["python src/clean_taq_nbbo_minute.py"],
        "task_dep": ["pull_TAQ_intraday"],
        "file_dep": [
            "src/clean_taq_nbbo_minute.py",
            "src/settings.py",
        ],
        "targets": [str(out_path)],
        "verbosity": 2,
    }


def task_label_headlines_gpt_batch():
    """Label headlines via OpenAI Batch API. Run: doit label_headlines_gpt_batch; then python src/label_headlines_gpt_batch.py --fetch when batches complete."""
    return {
        "actions": ["python src/label_headlines_gpt_batch.py"],
        "task_dep": ["clean_ravenpack"],
        "file_dep": [
            "src/label_headlines_gpt_batch.py",
            str(DATA_DIR / "ravenpack_dj_equities.parquet"),
            str(DATA_DIR / "clean" / "ravenpack_intraday_story.parquet"),
        ],
        "verbosity": 2,
    }


def task_data_exploration():
    """Generate data exploration figures (CRSP, TAQ, RavenPack, GPT)."""
    return {
        "actions": ["python src/data_exploration.py"],
        "file_dep": [
            "src/data_exploration.py",
            str(DATA_DIR / "clean" / "crsp_daily.parquet"),
            str(DATA_DIR / "clean" / "taq_nbbo_minute.parquet"),
            str(DATA_DIR / "clean" / "ravenpack_intraday_story.parquet"),
            str(DATA_DIR / "interim" / "gpt_labels.parquet"),
        ],
        "targets": [
            str(OUTPUT_DIR / "crsp_summary_stats.tex"),
            str(OUTPUT_DIR / "taq_intraday_midprice_example.png"),
            str(OUTPUT_DIR / "ravenpack_news_per_day.png"),
            str(OUTPUT_DIR / "gpt_label_distribution.png"),
        ],
        "verbosity": 2,
    }


# Date ranges for replication tasks
_ENV_REPLICATION = {"SAMPLE_START": "2021-10-01", "SAMPLE_END": "2024-05-31", "OUTPUT_SUFFIX": "_replication"}
_ENV_2025 = {"SAMPLE_START": "2024-05-31", "SAMPLE_END": "2024-12-31", "OUTPUT_SUFFIX": "_2025"}
_ENV_FULL = {"SAMPLE_START": "2021-10-01", "SAMPLE_END": "2024-12-31", "OUTPUT_SUFFIX": "_full"}


def task_table1_replication():
    """Table 1: replication (2021.10.1 - 2024.5.31)"""
    return {
        "actions": [_run_python("compute_portfolio_performance.py", _ENV_REPLICATION)],
        "file_dep": [
            "src/compute_portfolio_performance.py",
            str(DATA_DIR / "clean" / "taq_nbbo_minute.parquet"),
            str(DATA_DIR / "interim" / "gpt_labels.parquet"),
            str(DATA_DIR / "clean" / "ravenpack_intraday_story.parquet"),
            str(DATA_DIR / "clean" / "crsp_daily.parquet"),
        ],
        "targets": [
            str(OUTPUT_DIR / "performance_table_replication.html"),
            str(OUTPUT_DIR / "performance_table_replication.tex"),
        ],
        "verbosity": 2,
    }


def task_figure5_replication():
    """Figure 5: replication (2021.10.1 - 2024.5.31)"""
    return {
        "actions": [_run_python("graph_trading_strategy.py", _ENV_REPLICATION)],
        "file_dep": [
            "src/graph_trading_strategy.py",
            str(DATA_DIR / "interim" / "gpt_labels.parquet"),
            str(DATA_DIR / "clean" / "ravenpack_intraday_story.parquet"),
            str(DATA_DIR / "clean" / "crsp_daily.parquet"),
        ],
        "targets": [str(OUTPUT_DIR / "cumulative_returns_paper_style_replication.png")],
        "verbosity": 2,
    }


def task_table1_2025():
    """Table 1: 2024.5.31 - 2024.12.31"""
    return {
        "actions": [_run_python("compute_portfolio_performance.py", _ENV_2025)],
        "file_dep": [
            "src/compute_portfolio_performance.py",
            str(DATA_DIR / "clean" / "taq_nbbo_minute.parquet"),
            str(DATA_DIR / "interim" / "gpt_labels.parquet"),
            str(DATA_DIR / "clean" / "ravenpack_intraday_story.parquet"),
            str(DATA_DIR / "clean" / "crsp_daily.parquet"),
        ],
        "targets": [
            str(OUTPUT_DIR / "performance_table_2025.html"),
            str(OUTPUT_DIR / "performance_table_2025.tex"),
        ],
        "verbosity": 2,
    }


def task_figure5_2025():
    """Figure 5: 2024.5.31 - 2024.12.31"""
    return {
        "actions": [_run_python("graph_trading_strategy.py", _ENV_2025)],
        "file_dep": [
            "src/graph_trading_strategy.py",
            str(DATA_DIR / "interim" / "gpt_labels.parquet"),
            str(DATA_DIR / "clean" / "ravenpack_intraday_story.parquet"),
            str(DATA_DIR / "clean" / "crsp_daily.parquet"),
        ],
        "targets": [str(OUTPUT_DIR / "cumulative_returns_paper_style_2025.png")],
        "verbosity": 2,
    }


def task_table1_full():
    """Table 1: full sample (2021 - 2024.12.31)"""
    return {
        "actions": [_run_python("compute_portfolio_performance.py", _ENV_FULL)],
        "file_dep": [
            "src/compute_portfolio_performance.py",
            str(DATA_DIR / "clean" / "taq_nbbo_minute.parquet"),
            str(DATA_DIR / "interim" / "gpt_labels.parquet"),
            str(DATA_DIR / "clean" / "ravenpack_intraday_story.parquet"),
            str(DATA_DIR / "clean" / "crsp_daily.parquet"),
        ],
        "targets": [
            str(OUTPUT_DIR / "performance_table_full.html"),
            str(OUTPUT_DIR / "performance_table_full.tex"),
        ],
        "verbosity": 2,
    }


def task_figure5_full():
    """Figure 5: full sample (2021 - 2024.12.31)"""
    return {
        "actions": [_run_python("graph_trading_strategy.py", _ENV_FULL)],
        "file_dep": [
            "src/graph_trading_strategy.py",
            str(DATA_DIR / "interim" / "gpt_labels.parquet"),
            str(DATA_DIR / "clean" / "ravenpack_intraday_story.parquet"),
            str(DATA_DIR / "clean" / "crsp_daily.parquet"),
        ],
        "targets": [str(OUTPUT_DIR / "cumulative_returns_paper_style_full.png")],
        "verbosity": 2,
    }


def _run_latex():
    def _action():
        import subprocess
        cwd = Path(__file__).resolve().parent / "reports"
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "replication_report.tex"],
            cwd=cwd,
            check=True,
        )

    return _action


def task_latex_pdf():
    """Compile replication_report.tex to PDF."""
    tex_path = BASE_DIR / "reports" / "replication_report.tex"
    pdf_path = BASE_DIR / "reports" / "replication_report.pdf"
    return {
        "actions": [_run_latex()],
        "file_dep": [str(tex_path)],
        "targets": [str(pdf_path)],
        "verbosity": 2,
    }
