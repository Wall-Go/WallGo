## Will load the module from a fixed directory to facilitate testing and avoid accidentally loading a global installation.
# Normally the user would just do 'import WallGoCollision' assuming they have pip installed the module

import importlib.util
import sys

spec = importlib.util.spec_from_file_location("WallGoCollision", "../module/Debug/WallGoCollision.cp312-win_amd64.pyd")
WallGoCollision = importlib.util.module_from_spec(spec)
sys.modules["WallGoCollision"] = WallGoCollision
spec.loader.exec_module(WallGoCollision)

WallGoCollision.setSeed(1)

integrationOptions = WallGoCollision.IntegrationOptions()

input("done")