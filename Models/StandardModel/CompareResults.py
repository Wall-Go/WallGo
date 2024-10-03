import numpy as np
import matplotlib.pyplot as plt

MPResults = np.array([0.351, 0.356, 0.365,0.377, 0.390])

KNRmHs = np.array([0.120968, 1.63306, 3.54839, 5.16129, 7.17742, 10.504, 13.9315, \
17.1573, 20.6855, 25.7258, 34.1935, 39.2339, 46.3911, 50.3226, \
54.4556, 58.9919, 63.8306, 67.2581, 68.7702, 69.6774])
KNRResults = np.array([0.266239, 0.266251, 0.266251, 0.266239, 0.266251, 0.266251, 0.266263, \
0.266286, 0.26632, 0.266402, 0.266656, 0.266876, 0.267247, 0.267502, \
0.267838, 0.26829, 0.268915, 0.269471, 0.269761, 0.269923])

WallGoOutputMPN5M20 = np.loadtxt("./ResultsMPPotential/N5M20.txt", float)
WallGoOutputMPN11M20 = np.loadtxt("./ResultsMPPotential/N11M20.txt", float)
WallGoOutputMPN15M30 = np.loadtxt("./ResultsMPPotential/N15M30.txt", float)

WallGoOutputKNRN5M20 = np.loadtxt("./ResultsKNRPotential/N5M20.txt", float)
WallGoOutputKNRN11M20 = np.loadtxt("./ResultsKNRPotential/N11M20.txt", float)
WallGoOutputKNRN15M30 = np.loadtxt("./ResultsKNRPotential/N15M30.txt", float)

#Values of mH
mHs = WallGoOutputMPN5M20[:,0]

# Show the pressure curve
plt.plot(mHs, MPResults, label = 'MP', marker = '.', linestyle = 'dashed')
plt.plot(KNRmHs, KNRResults, label = 'KNR', linestyle = 'dotted')
#plt.errorbar(mHs, WallGoOutputMPN11M20[:,2],yerr = WallGoOutputMPN11M20[:,3], capsize =5 )
plt.errorbar(mHs, WallGoOutputMPN15M30[:,2],yerr = WallGoOutputMPN15M30[:,3], capsize =5 , label = 'WallGo, MP', marker = '.', linestyle = '-.')
plt.errorbar(mHs, WallGoOutputKNRN11M20[:,2],yerr = WallGoOutputMPN15M30[:,3], capsize =5, label = 'WallGo, KNR', marker = '.')
plt.errorbar(mHs, WallGoOutputMPN5M20[:,2],yerr = WallGoOutputMPN5M20[:,3], capsize = 5, label = 'WallGo, MP, N = 5', marker = '.', linestyle = ':')
plt.plot(mHs, WallGoOutputMPN15M30[:,1], color = 'grey')
plt.xlabel(r"$m_H$", fontsize=15)
plt.ylabel(r"$v_w$", fontsize=15)
plt.grid()
#plt.legend(loc='center right')
#plt.savefig('Compare.eps', format='eps')
plt.show()


# Check how the results depend on temperatureScale and fieldScale
L01phi50 = np.loadtxt("./ResultsMPPotential/N11M20tempScale0.1fieldScale50.0.txt", float)
L1phi50 = np.loadtxt("./ResultsMPPotential/N11M20tempScale1.0fieldScale50.0.txt", float)
L05phi50 = np.loadtxt("./ResultsMPPotential/N11M20tempScale0.5fieldScale50.0.txt", float)
L2phi50 = np.loadtxt("./ResultsMPPotential/N11M20tempScale2.0fieldScale50.0.txt", float)
L5phi50 = np.loadtxt("./ResultsMPPotential/N11M20tempScale5.0fieldScale50.0.txt", float)

plt.errorbar(mHs, L01phi50[:,2], yerr = L01phi50[:,3], capsize = 5, label = 'Tscale = 0.1 GeV, Fieldscale = 50 GeV')
plt.errorbar(mHs, L05phi50[:,2], yerr = L05phi50[:,3], capsize = 5, label = 'Tscale = 0.5 GeV, Fieldscale = 50 GeV', linestyle = ':')
plt.errorbar(mHs, L1phi50[:,2], yerr = L1phi50[:,3], capsize = 5, label = 'Tscale = 1.0 GeV, Fieldscale = 50 GeV', linestyle = '-.')
plt.errorbar(mHs, L2phi50[:,2], yerr = L2phi50[:,3], capsize = 5, label = 'Tscale = 2.0 GeV, Fieldscale = 50 GeV', linestyle = '--')
plt.errorbar(mHs, L5phi50[:,2], yerr = L2phi50[:,3], capsize = 5, label = 'Tscale = 5.0 GeV, Fieldscale = 50 GeV', linestyle = ':')

plt.legend(loc='lower left')

plt.xlabel(r"$m_H$", fontsize=15)
plt.ylabel(r"$v_w$", fontsize=15)

plt.savefig('CompareTempScale.eps', format='eps')

plt.show()

L1phi10 = np.loadtxt("./ResultsMPPotential/N11M20tempScale1.0fieldScale10.0.txt", float)
L1phi100 = np.loadtxt("./ResultsMPPotential/N11M20tempScale1.0fieldScale100.0.txt", float)

plt.errorbar(mHs, L1phi10[:,2], yerr = L1phi10[:,3], capsize = 5, label = 'Tscale = 1.0 GeV, Fieldscale = 10 GeV')
plt.errorbar(mHs, L1phi50[:,2], yerr = L1phi50[:,3], capsize = 5, label = 'Tscale = 1.0 GeV, Fieldscale = 50 GeV', linestyle = ':')
plt.errorbar(mHs, L1phi100[:,2], yerr = L1phi100[:,3], capsize = 5, label = 'Tscale = 1.0 GeV, Fieldscale = 100 GeV', linestyle = '-.')

plt.legend(loc='upper left')

plt.xlabel(r"$m_H$", fontsize=15)
plt.ylabel(r"$v_w$", fontsize=15)

plt.show()