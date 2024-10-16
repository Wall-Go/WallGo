import subprocess
import pathlib
import logging

## Put common matheamtica and DRalgo related functions here. Common physics/math functions should go into helpers.py


def generateMatrixElementsViaSubprocess(filePath: pathlib.Path) -> None:
    # Command to execute with the given file path
    command = ["wolframscript", "-file", filePath]

    try:
        # run wolframscript
        result = subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    except subprocess.CalledProcessError as e:
        # Handle errors in case the command fails
        logging.error(
            "Fatal: Error when generating matrix elements from mathematica via DRalgo"
        )
        logging.error(e.stderr.decode("utf-8"))
