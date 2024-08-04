import sys

try:
    # NB: package in working dir gets priority over whatever global installation
    import WallGoCollision
    
except Exception as e:
    print(f"Error: {e}")
    input("Press any key to exit...")
    sys.exit(1)

#WallGoCollision.setSeed() ## crash??

#help(WallGoCollision.ModelParameters)

integrationOptions = WallGoCollision.IntegrationOptions()
integrationOptions.maxTries = 10
integrationOptions.maxIntegrationMomentum = 20
integrationOptions.relativeErrorGoal = 0.1
integrationOptions.absoluteErrorGoal = 1

verbosity = WallGoCollision.CollisionTensorVerbosity()
verbosity.progressReportPercentage = 0.25

topQuark = WallGoCollision.ParticleDescription()
topQuark.type = WallGoCollision.EParticleType.eFermion
topQuark.name = "top"
topQuark.index = 0
topQuark.bUltrarelativistic = True
topQuark.bInEquilibrium = False
# gives verbose error:
"""
topQuark.massSqFunction = lambda params: 2*params
msq = topQuark.massSqFunction(3)
"""
# works:
def topMsq(p: WallGoCollision.ModelParameters) -> float:
    return p["gs"]**2 * 2

topQuark.massSqFunction = topMsq

params = WallGoCollision.ModelParameters()
params.addOrModifyParameter("gs", 2.42)

msq = topQuark.massSqFunction(params)

print(msq)

input("done")