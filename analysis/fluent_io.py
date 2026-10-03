"""
fluent_io.py - Readers for the raw solver exports.

Purpose
-------
Turns the two kinds of text file produced by ANSYS into pandas DataFrames:

1. Fluent report files (.out), written by a "report file" definition.
   Layout:
       "inlet-pressure-rfile"                  <- header line 1: report name
       "Iteration" "inlet-pressure"            <- header line 2: column names
       ("Iteration" "inlet-pressure")          <- header line 3: same, in ()
       1 0.6156371570340399                    <- data: iteration/time-step, value, ...
       2 1.544097661688011
   We skip the 3 header lines, take the column names from line 2, and read
   the whitespace-separated numbers.

2. ANSYS Workbench design-point table (.csv).
   Lines starting with '#' are comments describing the parameters; the first
   non-comment line is the header (Name,P1,P2,P3). The file may also start
   with a UTF-8 byte-order mark and use Windows line endings, both handled.
"""

import shlex
from pathlib import Path

import pandas as pd


def read_fluent_out(path: Path) -> pd.DataFrame:
    """Read a Fluent .out report file into a DataFrame.

    The first column (iteration or time-step) is used as the index so that
    windows can be selected with ``df.loc[300:559]``.
    """
    path = Path(path)
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()

    # Line 2 holds the quoted column names, e.g. "Iteration" "inlet-pressure".
    # shlex.split understands the quotes, so names containing spaces survive.
    columns = shlex.split(lines[1])

    rows = []
    for line in lines[3:]:            # data starts after the 3 header lines
        if line.strip():              # ignore blank lines at the end
            rows.append([float(x) for x in line.split()])

    df = pd.DataFrame(rows, columns=columns)
    index_col = columns[0]
    df[index_col] = df[index_col].astype(int)
    return df.set_index(index_col)


def read_workbench_csv(path: Path, rename: dict | None = None,
                       exclude: list[str] | None = None) -> pd.DataFrame:
    """Read an ANSYS Workbench design-point export.

    Parameters
    ----------
    rename  : mapping from Workbench parameter IDs (P1, P2...) to readable names.
    exclude : design-point names (e.g. ["DP 0"]) to drop from the table.
    """
    df = pd.read_csv(path, comment="#", encoding="utf-8-sig",
                     skipinitialspace=True)
    df["Name"] = df["Name"].str.strip()
    if exclude:
        df = df[~df["Name"].isin(exclude)]
    if rename:
        df = df.rename(columns=rename)
    return df.reset_index(drop=True)
