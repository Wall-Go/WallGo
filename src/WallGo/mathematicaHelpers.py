import subprocess
import pathlib

## Put common matheamtica and DRalgo related functions here. Common physics/math functions should go into helpers.py

def calculateMatrixElements(filePath: pathlib.Path) -> None:
    # Command to execute with the given file path
    command = ['wolframscript', '-file', filePath]
    
    try:
        # run wolframscript 
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
    except subprocess.CalledProcessError as e:
        # Handle errors in case the command fails
        print("Fatal: Error when generating matrix elements from mathematica via DRalgo")
        print(e.stderr.decode('utf-8'))

