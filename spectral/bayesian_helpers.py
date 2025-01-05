"""Script to run RBA analysis for a given variable."""

import os
import subprocess
from pathlib import Path
from typing import Optional


def run_rba_analysis(
    sample: str,
    command: str,
    data_dir: str = "rba/data",
    output_dir: str = "rba/output",
    rlib: Optional[str] = None,
) -> Optional[str]:
    """
    Run the RBA (ridge-based analysis) for a given sample.

    Args:
        sample (str): The sample identifier, used to name input, output, and log files.
        command (str): The command template to run the RBA analysis. Must include placeholders for `rlib`, `prefix`, and `input_table`.
        data_dir (str, optional): Path to the directory where input data files are stored. Default is "rba/data".
        output_dir (str, optional): Path to the directory where output (results and logs) will be stored. Default is "rba/output".
        rlib (Optional[str], optional): Path to the R library for the analysis. If not provided, a default path is used.

    Returns:
        Optional[str]: If an error occurs during the analysis, returns the error message. Otherwise, returns None.

    Side Effects:
        - Creates necessary directories if they don't exist.
        - Runs the RBA analysis command via subprocess.
        - Saves log and result files in the specified output directory.
        - Moves the generated ridge plot to the results folder.

    Example:
        command = "Rscript --vanilla my_script.R --rlib={rlib} --input={input_table} --output={prefix}"
        run_rba_analysis(sample="sample1", command=command)
    """

    # Convert paths to Path objects
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Define log and results directories as subfolders of the output directory
    log_path = output_path / "log"
    results_path = output_path / "results"

    # Ensure directories exist
    for dir_path in [log_path, results_path]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Define paths using Path object
    input_table = data_path / f"table_{sample}.txt"
    log_diary = log_path / f"diary_{sample}.txt"
    plot = results_path / f"ridge_{sample}.pdf"
    prefix = results_path / f"results_{sample}"
    results_file = results_path / f"results_{sample}.txt"

    # Print or log paths for debugging (optional)
    print(f"Input Table: {input_table}")
    print(f"Log Diary: {log_diary}")
    print(f"Plot: {plot}")
    print(f"Results File: {results_file}")

    # Default R library path if not provided
    if rlib is None:
        rlib = "/Users/daniel/PhD/Projects/meg-assr-2023/renv/library/R-4.4/aarch64-apple-darwin20"

    # Construct the RBA command
    cmd = command.format(rlib=rlib, prefix=prefix, input_table=input_table)

    # Run the command
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )

        # Write output and error streams to log file
        with open(log_diary, "w", encoding="utf-8") as f:
            f.write(result.stdout)
            f.write(result.stderr)

        # Move the generated ridge plot
        os.rename("Intercept_ridge.pdf", plot)

        print(f"Analysis completed successfully for {sample}")
        print(f"Results saved to {results_file}")
        print(f"Log saved to {log_diary}")
        print(f"Ridge plot saved to {plot}")

    except subprocess.CalledProcessError as e:
        # Handle any errors that occur during the subprocess call
        error_message = f"""
        Error occurred while running analysis for {sample}
        Error message: {e}
        Command output:
        {e.stdout}
        {e.stderr}
        """
        print(error_message)

        # Write error message to log file
        with open(log_diary, "w", encoding="utf-8") as f:
            f.write(error_message)

        return error_message  # Return the error message in case of failure

    return None  # Return None if the analysis is successful
