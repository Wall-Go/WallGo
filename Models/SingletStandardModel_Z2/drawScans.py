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
                if 79 < model['ms'] < 81 and 0.7 < model['lambdaHS'] < 0.72:
                    print(i, scan['vwOut'], model['ms'], model['lambdaHS'])
            elif scan['vwOut'] == 1 and model['lambdaHS'] > (0.2/20)*model['ms']-0.35:
                msRun.append(model['ms'])
                lHSRun.append(model['lambdaHS'])
    
    fig,ax = plt.subplots(1)
    c=ax.scatter(ms, lHS, s=4, c=vw)
    cbar = fig.colorbar(c)
    ax.scatter(msRun, lHSRun, s=4, c='r', label=r'$v_w>v_J$')
    cbar.ax.tick_params(labelsize=12)
    ax.tick_params(labelsize=12)
    cbar.set_label(r'$v_w$', fontsize=15)
    ax.set_xlabel(r"$m_s$ (GeV)", fontsize=15)
    ax.set_ylabel(r'$\lambda_{HS}$', fontsize=15)
    ax.set_xlim((125/2,160))
    ax.legend(fontsize=15, markerscale=2)
    # ax.set_ylim((0.7,0.75))
    plt.grid(True)
    plt.savefig('plots/deflagrationScan.pdf',bbox_inches='tight')
    plt.show()
    
def drawDeton():
    ms = []
    msRun = []
    msDef = []
    msUnknown = []
    lHS = []
    lHSRun = []
    lHSDef = []
    lHSUnknown = []
    vw = []
    for i,model in enumerate(models):
        scan = scanResults[i]
        if scan['errorDeton'] == 'success':
            if 0 < scan['vwDeton'] < 1:
                ms.append(model['ms'])
                lHS.append(model['lambdaHS'])
                vw.append(scan['vwDeton'])
                # if 0.975 < model['lambdaHS'] < 1 and 137.5 > model['ms'] > 135.5:
                #     print(i, scan['vwDeton'], model['ms'], model['lambdaHS'], model['Tn'])
            elif scan['vwDeton'] == 1:
                msRun.append(model['ms'])
                lHSRun.append(model['lambdaHS'])
                # if model['lambdaHS'] < 1 and model['ms'] > 132:
                #     print(i, scan['vwDeton'], model['ms'], model['lambdaHS'], model['Tn'])
            elif scan['vwDeton'] == 0:
                msDef.append(model['ms'])
                lHSDef.append(model['lambdaHS'])
                if 0.86 < model['lambdaHS'] < 0.9 and 115 < model['ms'] < 120:
                    print(i, scan['vwDeton'], model['ms'], model['lambdaHS'], model['Tn'])
            elif scan['vwDeton'] == -2:
                msUnknown.append(model['ms'])
                lHSUnknown.append(model['lambdaHS'])
    
    fig,ax = plt.subplots(1)
    c=ax.scatter(ms, lHS, s=4, c=vw)
    cbar = fig.colorbar(c)
    ax.scatter(msRun, lHSRun, s=4, c='r', label=r'$\gamma_w\gg 1$')
    ax.scatter(msDef, lHSDef, s=4, c='grey', label=r'$v_w<v_J$')
    ax.scatter(msUnknown, lHSUnknown, s=4, c='black', label=r'$\gamma_w\gg 1$ or $v_w < v_J$')
    cbar.ax.tick_params(labelsize=12)
    ax.tick_params(labelsize=12)
    cbar.set_label(r'$v_w$', fontsize=15)
    ax.set_xlabel(r"$m_s$ (GeV)", fontsize=15)
    ax.set_ylabel(r'$\lambda_{HS}$', fontsize=15)
    ax.legend(fontsize=13, markerscale=2)
    # x = np.linspace(130,160,10)
    # ax.plot(x, 0.008333*x-0.08333, c='black')
    ax.set_xlim((125/2,160))
    plt.grid(True)
    plt.savefig('plots/detonationScan.pdf',bbox_inches='tight')
    plt.show()
    n = len(ms)
    nRun = len(msRun)
    nDef = len(msDef)
    nUnknown = len(msUnknown)
    ntot = n+nRun+nDef
    print(n,nRun,nDef, nUnknown,ntot)
    print(n/ntot, nRun/ntot, nDef/ntot, nUnknown/ntot)
    
def drawTn(msRange, lHSRange):
    ms = []
    lHS = []
    Tn = []
    for i,model in enumerate(models):
        if msRange[0] < model['ms'] < msRange[1] and lHSRange[0] < model['lambdaHS'] < lHSRange[1]:
            ms.append(model['ms'])
            lHS.append(model['lambdaHS'])
            Tn.append(model['Tn'])
    fig,ax = plt.subplots(1)
    c=ax.scatter(ms, lHS, s=4, c=Tn)
    cbar = fig.colorbar(c)
    # ax.set_xlim(msRange)
    # ax.set_ylim(lHSRange)
    plt.grid(True)
    plt.show()