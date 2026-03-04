"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based

"""

#######################################
## Configuration and Helpers for PyDoit
#######################################
## Make sure the src folder is in the path
import sys

sys.path.insert(1, "./src/")

import shutil
from os import environ
from pathlib import Path

from settings import config

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


def task_generate_exploratory_charts():
    return {
        "actions": ["python src/generate_chart.py"],
        "file_dep": [
            "_data/CRSP_daily_stock.parquet",
            "_data/ravenpack_dj_equities.parquet",
        ],
        "targets": [
            "_output/crsp_price_timeseries.html",
            "_output/ravenpack_sentiment_timeseries.html",
        ],
        "clean": True,
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

def task_clean_ravenpack_firmday():
    """
    Build firm-day news table:
    _data/ravenpack_dj_equities.parquet -> _data/clean/news_firmday.parquet
    """
    raw_path = DATA_DIR / "ravenpack_dj_equities.parquet"
    out_path = DATA_DIR / "clean" / "news_firmday.parquet"

    return {
        "actions": [
            "python src/clean_ravenpack.py",  # runs both in __main__, see note below
        ],
        "file_dep": [
            "src/clean_ravenpack.py",
            "src/settings.py",
            str(raw_path),
        ],
        "targets": [
            str(out_path),
        ],
        "verbosity": 2,
    }


def task_build_ravenpack_intraday_story():
    """
    Build intraday story-level table:
    _data/ravenpack_dj_equities.parquet -> _data/clean/ravenpack_intraday_story.parquet
    """
    raw_path = DATA_DIR / "ravenpack_dj_equities.parquet"
    out_path = DATA_DIR / "clean" / "ravenpack_intraday_story.parquet"

    return {
        "actions": [
            "python src/clean_ravenpack.py",  # runs both in __main__, see note below
        ],
        "file_dep": [
            "src/clean_ravenpack.py",
            "src/settings.py",
            str(raw_path),
        ],
        "targets": [
            str(out_path),
        ],
        "verbosity": 2,
    }
