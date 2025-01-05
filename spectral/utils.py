"""
Includes utility functions for the project.
"""

import sys
import os
import io
import functools
import inspect
from contextlib import contextmanager, redirect_stdout
from importlib import resources
from pathlib import Path
import pandas as pd
import scipy.io as sio


def load_conditions_coding() -> pd.DataFrame:
    """
    Load coding telling us what conditions was representing specific recording (Sham or real TDCS)
    and extract session information
    """
    keys_path = resources.files("spectral.data").joinpath("keys.mat")
    keys = sio.loadmat(keys_path)
    row_names = keys["keys"].tolist()[0]

    df = pd.DataFrame()
    df["subject_session"] = [el[0].flatten()[0] for el in row_names]
    df["subject_session"] = df["subject_session"].astype("category").str.strip()
    df["T"] = [el[1].flatten()[0] for el in row_names]

    # Extract session (A or B) from subject_session
    df["session"] = df["subject_session"].str[-1]
    df["subject"] = df["subject_session"].str.extract(r"(Subject\d+)")
    df["session_rs"] = df["session"].map({"A": "Session1", "B": "Session2"})
    subject_id_list = [f"S{str(i).zfill(3)}" for i in range(1, 16)]
    # Extract subject number and session
    df[["subject_num"]] = df["subject_session"].str.extract(r"Subject(\d+)")
    # Map subject numbers to the predefined list
    subject_map = {str(i + 1).zfill(2): sid for i, sid in enumerate(subject_id_list)}
    df["subject_rs"] = df["subject_num"].map(subject_map)

    return df.drop(columns=["subject_num"])


def assign_run_status(df: pd.DataFrame) -> pd.DataFrame:
    """Assign pre/post status based on the 'run' column"""
    return df.assign(P=lambda x: x["run"].astype("str").str.lstrip("0")).replace(
        {
            "P": {
                "1": "pre",
                "2": "post",
                "3": "pre",
                "4": "post",
            }
        }
    )


def assign_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Assign metadata to the dataframe"""
    assr_keys = load_conditions_coding()
    return df.assign(
        subject_cond=lambda x: "Subject"
        + x["sub"].astype(str).str.zfill(2)
        + "_"
        + x["ses"]
    ).join(assr_keys.set_index("subject_cond"), on="subject_cond")


@contextmanager
def suppress_stdout():
    """Suppress stdout."""
    with io.open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull):
            try:
                yield
            except Exception:
                print(sys.exc_info()[0])
                raise


@contextmanager
def empty_folder_check(folder):
    """
    A context manager that checks if a folder is empty before allowing processing.

    Parameters:
    - folder (str): Path to the folder to check

    Yields:
    - bool: True if the folder is empty, False otherwise

    Usage:
    with empty_folder_check(folder_path) as is_empty:
        if is_empty:
            # Perform processing
        else:
            # Skip processing
    """
    path = Path(folder) if isinstance(folder, str) else folder
    path.mkdir(parents=True, exist_ok=True)
    is_empty = not any(path.iterdir())

    yield is_empty

    print(
        f"Processing in {folder} {'completed' if is_empty else 'was skipped as it was not empty'}."
    )


def debug_print(*variables_to_print):
    """A decorator that prints the specified variables from the function's local scope."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Call the original function
            result = func(*args, **kwargs)

            # Get the function's signature
            sig = inspect.signature(func)

            # Bind the arguments to the signature
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Create a dictionary of parameter names and values
            local_vars = dict(bound_args.arguments)

            # Update with any other local variables
            local_vars.update(inspect.currentframe().f_locals)

            # Print the specified variables
            print(f"Debug info for {func.__name__}:")
            for var in variables_to_print:
                if var in local_vars:
                    print(f"  {var} = {local_vars[var]}")
                else:
                    print(f"  {var} not found in local scope")

            return result

        return wrapper

    return decorator
