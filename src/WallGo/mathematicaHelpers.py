import pathlib
import subprocess
import sys

# Put common Wolfram Mathematica and WallGoMatrix related functions here.
# Common physics/math functions should go into helpers.py

def generateMatrixElementsViaSubprocess(inFilePath: pathlib.Path, outFilePath: pathlib.Path) -> None:
    """
    Generates matrix elements by executing a Mathematica script via a subprocess.

    This function takes the input and output file paths, converts them to string representations,
    and constructs a command to run a Mathematica script using `wolframscript`. The command is
    executed using the `subprocess.run` method, and the output is printed to the console. If the
    command fails, an error message is printed.

    This requires an active WolframKernel.

    Args:
        inFilePath (pathlib.Path): The path to the input file containing the Mathematica script.
        outFilePath (pathlib.Path): The path to the output file where the results will be saved.

    Raises:
        subprocess.CalledProcessError: If the subprocess command fails.
    """
    # Ensure filePath is string representation of the path
    filePathStr = str(inFilePath)
    outFilePathStr = str(outFilePath)

    # Command to execute with the given file path, adjusting for platform
    if sys.platform == "win32":
        command = ["wolframscript", "-script", filePathStr, outFilePathStr]
    else:  # For Linux and macOS
        command = ["wolframscript", "-script", filePathStr, outFilePathStr]

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
        print("Fatal: Error when generating matrix elements from mathematica via WallGoMatrix")
        print(e.stderr.decode("utf-8"))