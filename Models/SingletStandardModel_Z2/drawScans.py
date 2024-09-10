import numpy as np
import matplotlib.pyplot as plt

scanResults = np.load('scanResultsTruncated.npy', allow_pickle=True).tolist()
models = np.load('modelsTruncated.npy', allow_pickle=True).tolist()

def drawDeflHyb():
    ms = []
    msRun = []
    lHS = []
    lHSRun = []
    vw = []
    for i,model in enumerate(models):
        scan = scanResults[i]
        if scan['error'] == 'success':
            if 0 < scan['vwOut'] < 1:
                ms.append(model['ms'])
                lHS.append(model['lambdaHS'])
                vw.append(scan['vwOut'])
            elif scan['vwOut'] == 1:
                msRun.append(model['ms'])
                lHSRun.append(model['lambdaHS'])
    
    fig,ax = plt.subplots(1)
    c=ax.scatter(ms, lHS, s=4, c=vw)
    cbar = fig.colorbar(c)
    ax.scatter(msRun, lHSRun, s=4, c='r')
    ax.set_xlim((60,160))
    plt.grid(True)
    plt.show()
    
def drawDeton():
    ms = []
    msRun = []
    msDef = []
    lHS = []
    lHSRun = []
    lHSDef = []
    vw = []
    for i,model in enumerate(models):
        scan = scanResults[i]
        if scan['errorDeton'] == 'success':
            if 0 < scan['vwDeton'] < 1:
                ms.append(model['ms'])
                lHS.append(model['lambdaHS'])
                vw.append(scan['vwDeton'])
            elif scan['vwDeton'] == 1:
                msRun.append(model['ms'])
                lHSRun.append(model['lambdaHS'])
            elif scan['vwDeton'] == 0:
                msDef.append(model['ms'])
                lHSDef.append(model['lambdaHS'])
    
    fig,ax = plt.subplots(1)
    c=ax.scatter(ms, lHS, s=4, c=vw)
    cbar = fig.colorbar(c)
    ax.scatter(msRun, lHSRun, s=4, c='r')
    ax.scatter(msDef, lHSDef, s=4, c='grey')
    ax.set_xlim((60,160))
    plt.grid(True)
    plt.show()