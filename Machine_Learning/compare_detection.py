import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d

rf_graff = np.loadtxt('../support_data/splines_detection_fraction_z_RF.txt')
#rf_anj_1 = np.loadtxt('support_data/splines_detection_fraction_z_RF_1_LargeBin.txt')
#rf_anj_1a = np.loadtxt('support_data/splines_detection_fraction_z_RF_1a_LargeBin.txt')
#rf_anj_1ab = np.loadtxt('support_data/splines_detection_fraction_z_RF_1_combined_LargeBin.txt')
#rf_anj_2 = np.loadtxt('support_data/splines_detection_fraction_z_RF_2_LargeBin.txt')
#rf_anj_12 = np.loadtxt('support_data/splines_detection_fraction_z_RF_500tree_zpt10001_LargeBin.txt')
#rf_anj_new = np.loadtxt('support_data/splines_detection_fraction_z_RF_500tree_NewData_Quick.txt')
rf_anj_v2 = np.loadtxt('../support_data/splines_detection_fraction_RF_v2_z10000.txt')

fig, ax = plt.subplots(1)
#fig2, ax2 = plt.subplots(1)

ax.plot(rf_graff[0::10,0],rf_graff[0::10,1],'-r',lw=2,label="Graff RF")
ax.plot(rf_anj_v2[0::10,0],rf_anj_v2[0::10,1],'-b',lw=2,label="RF v2")
#f = interp1d(rf_anj_12[0::10,0],rf_anj_12[0::10,1], kind='cubic')
#x = np.linspace(0, 10, num=100, endpoint=False)
#ax.plot(x,f(x),'-b',lw=2,label="RF")
#ax.plot(rf_anj_new[0::10,0],rf_anj_new[0::10,1],'-g',lw=2,label="Flatter Luminosity")
ax.set_yscale('log')
fig.savefig('ML_Plots/DF_v2_z10000compare.png')

#ax2.plot(rf_anj[:,0],abs(rf_anj[:,1]-rf_graff[0::10,1]) ,'-b',lw=2,label="Diff")
#ax2.plot(rf_anj[:,0],abs(rf_anj[:,1]-rf_graff[:,1]) ,'-b',lw=2,label="Diff")
#fig2.savefig('ML_Plots/DF_Diff_500tree_NewData_4_5_Quick.png')

#ax.plot(rf_graff[0::10,0],rf_graff[0::10,1],'-r',lw=2,label="Graff")
#ax.plot(rf_anj_1[0::10,0],rf_anj_1[0::10,1],'-b',lw=2,label="Set 1a")
#ax.plot(rf_anj_1a[0::10,0],rf_anj_1a[0::10,1],'-y',lw=2,label="Set 1b")
#ax.plot(rf_anj_1ab[0::10,0],rf_anj_1ab[0::10,1],'-g',lw=2,label="Set 1 Combined")
#ax.plot(rf_anj_2[0::10,0],rf_anj_2[0::10,1],'-c',lw=2,label="Set 2")
#ax.plot(rf_anj_12[0::10,0],rf_anj_12[0::10,1],'-m',lw=2,label="Set 1 and 2 Combined")
#f = interp1d(rf_anj_12[0::10,0],rf_anj_12[0::10,1], kind='cubic')
#x = np.linspace(0, 10, num=100, endpoint=False)
#ax.plot(x,f(x),'-m',lw=2,label="Set 1 and 2 Combined")
#ax.set_yscale('log')
#plt.xlabel('')
#plt.ylabel('')
#plt.legend(prop={'size':10.5})
#fig.savefig('ML_Plots/DF_Compare.png')

plt.show()
