from lib import CollisionModule

## Example: create mock data in format that the C++ program can understand.
# This can be any additional input that the integral may need, eg. particle masses.
data = CollisionModule.InputData()
data.values["mq"] = 0.5013256549
data.values["mg"] = 1.7366430109
data.values["gs"] = 1.2279920495357861

## Call C++ function to compute the collision integral
CollisionModule.pybindTestFunction(data)
CollisionModule.calculateAllCollisions()
