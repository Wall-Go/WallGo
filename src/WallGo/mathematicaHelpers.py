import pathlib
import subprocess
import sys

# Put common matheamtica and DRalgo related functions here.
# Common physics/math functions should go into helpers.py

def generateMatrixElementsViaSubprocess(filePath: pathlib.Path) -> None:
    # Ensure filePath is string representation of the path
    filePathStr = str(filePath)

    # Command to execute with the given file path, adjusting for platform
    if sys.platform == "win32":
        command = ["wolframscript", "-file", filePathStr]
    else:  # For Linux and macOS
        command = ["wolframscript", "-file", filePathStr]

    try:
        # run wolframscript
        result = subprocess.run(
            command, check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(result.stdout.decode('utf-8'))  # If you want to print the output

    except subprocess.CalledProcessError as e:
        # Handle errors in case the command fails
        print("Fatal: Error when generating matrix elements from mathematica via DRalgo")
        print(e.stderr.decode("utf-8"))