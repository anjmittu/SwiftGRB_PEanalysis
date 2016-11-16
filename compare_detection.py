import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

rf_graff = np.loadtxt('support_data/splines_detection_fraction_z_RF.txt')
rf_anj = np.loadtxt('support_data/splines_detection_fraction_z_RF_500tree_zpt10001.txt')

fig, ax = plt.subplots(1)
fig2, ax2 = plt.subplots(1)

ax.plot(rf_graff[0::10,0],rf_graff[0::10,1],'-r',lw=2,label="Graff RF")
ax.plot(rf_anj[0::10,0],rf_anj[0::10,1],'-b',lw=2,label="RF")
ax.set_yscale('log')
fig.savefig('ML_Plots/DF_500tree_zpt10001log.png')

#ax2.plot(rf_anj[:,0],abs(rf_anj[:,1]-rf_graff[0::10,1]) ,'-b',lw=2,label="Diff")
ax2.plot(rf_anj[:,0],abs(rf_anj[:,1]-rf_graff[:,1]) ,'-b',lw=2,label="Diff")
fig2.savefig('ML_Plots/DF_Diff_500tree_zpt10001.png')

plt.show()


