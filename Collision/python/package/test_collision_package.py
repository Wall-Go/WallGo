import sys

try:
    # NB: package in working dir gets priority over whatever global installation
    import WallGoCollision
    
except Exception as e:
    print(f"Error: {e}")
    input("Press any key to exit...")
    sys.exit(1)

#WallGoCollision.setSeed() ## crash??

integrationOptions = WallGoCollision.IntegrationOptions()
integrationOptions.maxTries = 10
integrationOptions.maxIntegrationMomentum = 20

input("done")