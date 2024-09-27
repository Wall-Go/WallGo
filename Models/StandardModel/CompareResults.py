import numpy as np
import matplotlib.pyplot as plt

MPResults = np.array([0.351, 0.356, 0.365,0.377, 0.390])

WallGoOutputMPN5M20 = np.loadtxt("./ResultsMPPotential/N5M20.txt", float)
WallGoOutputMPN11M20 = np.loadtxt("./ResultsMPPotential/N11M20.txt", float)

#Values of mH
mHs = WallGoOutputMPN5M20[:,0]

# Show the pressure curve
plt.plot(mHs, MPResults)
plt.errorbar(mHs, WallGoOutputMPN5M20[:,2],yerr = WallGoOutputMPN5M20[:,3])
plt.errorbar(mHs, WallGoOutputMPN11M20[:,2],yerr = WallGoOutputMPN11M20[:,3])
plt.xlabel(r"$m_H$", fontsize=15)
plt.ylabel(r"$v_w$", fontsize=15)
plt.grid()
plt.show()