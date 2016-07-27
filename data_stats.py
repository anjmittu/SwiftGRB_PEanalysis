import numpy as np

datafileRF1 = 'chains/RD_RF_Onebreak_post_equal_weights.dat'
datafileAB1 = 'chains/RD_AB_Onebreak_post_equal_weights.dat'
datafileNN1 = 'chains/RD_NN_Onebreak_post_equal_weights.dat'
datafileRF2 = 'chains/RD_RF_Twobreak_post_equal_weights.dat'
datafileAB2 = 'chains/RD_AB_Twobreak_post_equal_weights.dat'
datafileNN2 = 'chains/RD_NN_Twobreak_post_equal_weights.dat'


dataRF1n0 = np.loadtxt(datafileRF1, usecols=[0])
dataRF1n1 = np.loadtxt(datafileRF1, usecols=[1])
dataRF1n2 = np.loadtxt(datafileRF1, usecols=[2])
dataRF1z1 = np.loadtxt(datafileRF1, usecols=[3])
dataRF1N = np.loadtxt(datafileRF1, usecols=[5])

dataAB1n0 = np.loadtxt(datafileAB1, usecols=[0])
dataAB1n1 = np.loadtxt(datafileAB1, usecols=[1])
dataAB1n2 = np.loadtxt(datafileAB1, usecols=[2])
dataAB1z1 = np.loadtxt(datafileAB1, usecols=[3])
dataAB1N = np.loadtxt(datafileAB1, usecols=[5])

dataNN1n0 = np.loadtxt(datafileNN1, usecols=[0])
dataNN1n1 = np.loadtxt(datafileNN1, usecols=[1])
dataNN1n2 = np.loadtxt(datafileNN1, usecols=[2])
dataNN1z1 = np.loadtxt(datafileNN1, usecols=[3])
dataNN1N = np.loadtxt(datafileNN1, usecols=[5])

dataRF2n0 = np.loadtxt(datafileRF2, usecols=[0])
dataRF2n1 = np.loadtxt(datafileRF2, usecols=[1])
dataRF2n2 = np.loadtxt(datafileRF2, usecols=[2])
dataRF2n3 = np.loadtxt(datafileRF2, usecols=[3])
dataRF2z1 = np.loadtxt(datafileRF2, usecols=[4])
dataRF2z2 = np.loadtxt(datafileRF2, usecols=[5])
dataRF2N = np.loadtxt(datafileRF2, usecols=[7])

dataAB2n0 = np.loadtxt(datafileAB2, usecols=[0])
dataAB2n1 = np.loadtxt(datafileAB2, usecols=[1])
dataAB2n2 = np.loadtxt(datafileAB2, usecols=[2])
dataAB2n3 = np.loadtxt(datafileAB2, usecols=[3])
dataAB2z1 = np.loadtxt(datafileAB2, usecols=[4])
dataAB2z2 = np.loadtxt(datafileAB2, usecols=[5])
dataAB2N = np.loadtxt(datafileAB2, usecols=[7])

dataNN2n0 = np.loadtxt(datafileNN2, usecols=[0])
dataNN2n1 = np.loadtxt(datafileNN2, usecols=[1])
dataNN2n2 = np.loadtxt(datafileNN2, usecols=[2])
dataNN2n3 = np.loadtxt(datafileNN2, usecols=[3])
dataNN2z1 = np.loadtxt(datafileNN2, usecols=[4])
dataNN2z2 = np.loadtxt(datafileNN2, usecols=[5])
dataNN2N = np.loadtxt(datafileNN2, usecols=[7])

print "One Break RF"
print "n0: %.4f to %.4f" % (np.sort(dataRF1n0)[(dataRF1n0.size*.05)], np.sort(dataRF1n0)[(dataRF1n0.size*.95)])
print "n1: %.4f to %.4f" % (np.sort(dataRF1n1)[(dataRF1n1.size*.05)], np.sort(dataRF1n1)[(dataRF1n1.size*.95)])
print "n2: %.4f to %.4f" % (np.sort(dataRF1n2)[(dataRF1n2.size*.05)], np.sort(dataRF1n2)[(dataRF1n2.size*.95)])
print "z1: %.4f to %.4f" % (np.sort(dataRF1z1)[(dataRF1z1.size*.05)], np.sort(dataRF1z1)[(dataRF1z1.size*.95)])
print "N: %.4f to %.4f" % (np.sort(dataRF1N)[(dataRF1N.size*.05)], np.sort(dataRF1N)[(dataRF1N.size*.95)])
print ""

print "One Break AB"
print "n0: %.4f to %.4f" % (np.sort(dataAB1n0)[(dataAB1n0.size*.05)], np.sort(dataAB1n0)[(dataAB1n0.size*.95)])
print "n1: %.4f to %.4f" % (np.sort(dataAB1n1)[(dataAB1n1.size*.05)], np.sort(dataAB1n1)[(dataAB1n1.size*.95)])
print "n2: %.4f to %.4f" % (np.sort(dataAB1n2)[(dataAB1n2.size*.05)], np.sort(dataAB1n2)[(dataAB1n2.size*.95)])
print "z1: %.4f to %.4f" % (np.sort(dataAB1z1)[(dataAB1z1.size*.05)], np.sort(dataAB1z1)[(dataAB1z1.size*.95)])
print "N: %.4f to %.4f" % (np.sort(dataAB1N)[(dataAB1N.size*.05)], np.sort(dataAB1N)[(dataAB1N.size*.95)])
print ""

print "One Break NN"
print "n0: %.4f to %.4f" % (np.sort(dataNN1n0)[(dataNN1n0.size*.05)], np.sort(dataNN1n0)[(dataNN1n0.size*.95)])
print "n1: %.4f to %.4f" % (np.sort(dataNN1n1)[(dataNN1n1.size*.05)], np.sort(dataNN1n1)[(dataNN1n1.size*.95)])
print "n2: %.4f to %.4f" % (np.sort(dataNN1n2)[(dataNN1n2.size*.05)], np.sort(dataNN1n2)[(dataNN1n2.size*.95)])
print "z1: %.4f to %.4f" % (np.sort(dataNN1z1)[(dataNN1z1.size*.05)], np.sort(dataNN1z1)[(dataNN1z1.size*.95)])
print "N: %.4f to %.4f" % (np.sort(dataNN1N)[(dataNN1N.size*.05)], np.sort(dataNN1N)[(dataNN1N.size*.95)])
print ""

print "Two Break RF"
print "n0: %.4f to %.4f" % (np.sort(dataRF2n0)[(dataRF2n0.size*.05)], np.sort(dataRF2n0)[(dataRF2n0.size*.95)])
print "n1: %.4f to %.4f" % (np.sort(dataRF2n1)[(dataRF2n1.size*.05)], np.sort(dataRF2n1)[(dataRF2n1.size*.95)])
print "n2: %.4f to %.4f" % (np.sort(dataRF2n2)[(dataRF2n2.size*.05)], np.sort(dataRF2n2)[(dataRF2n2.size*.95)])
print "n3: %.4f to %.4f" % (np.sort(dataRF2n3)[(dataRF2n3.size*.05)], np.sort(dataRF2n3)[(dataRF2n3.size*.95)])
print "z1: %.4f to %.4f" % (np.sort(dataRF2z1)[(dataRF2z1.size*.05)], np.sort(dataRF2z1)[(dataRF2z1.size*.95)])
print "z2: %.4f to %.4f" % (np.sort(dataRF2z2)[(dataRF2z2.size*.05)], np.sort(dataRF2z2)[(dataRF2z2.size*.95)])
print "N: %.4f to %.4f" % (np.sort(dataRF2N)[(dataRF2N.size*.05)], np.sort(dataRF2N)[(dataRF2N.size*.95)])
print ""

print "Two Break AB"
print "n0: %.4f to %.4f" % (np.sort(dataAB2n0)[(dataAB2n0.size*.05)], np.sort(dataAB2n0)[(dataAB2n0.size*.95)])
print "n1: %.4f to %.4f" % (np.sort(dataAB2n1)[(dataAB2n1.size*.05)], np.sort(dataAB2n1)[(dataAB2n1.size*.95)])
print "n2: %.4f to %.4f" % (np.sort(dataAB2n2)[(dataAB2n2.size*.05)], np.sort(dataAB2n2)[(dataAB2n2.size*.95)])
print "n3: %.4f to %.4f" % (np.sort(dataAB2n3)[(dataAB2n3.size*.05)], np.sort(dataAB2n3)[(dataAB2n3.size*.95)])
print "z1: %.4f to %.4f" % (np.sort(dataAB2z1)[(dataAB2z1.size*.05)], np.sort(dataAB2z1)[(dataAB2z1.size*.95)])
print "z2: %.4f to %.4f" % (np.sort(dataAB2z2)[(dataAB2z2.size*.05)], np.sort(dataAB2z2)[(dataAB2z2.size*.95)])
print "N: %.4f to %.4f" % (np.sort(dataAB2N)[(dataAB2N.size*.05)], np.sort(dataAB2N)[(dataAB2N.size*.95)])
print ""

print "Two Break NN"
print "n0: %.4f to %.4f" % (np.sort(dataNN2n0)[(dataNN2n0.size*.05)], np.sort(dataNN2n0)[(dataNN2n0.size*.95)])
print "n1: %.4f to %.4f" % (np.sort(dataNN2n1)[(dataNN2n1.size*.05)], np.sort(dataNN2n1)[(dataNN2n1.size*.95)])
print "n2: %.4f to %.4f" % (np.sort(dataNN2n2)[(dataNN2n2.size*.05)], np.sort(dataNN2n2)[(dataNN2n2.size*.95)])
print "n3: %.4f to %.4f" % (np.sort(dataNN2n3)[(dataNN2n3.size*.05)], np.sort(dataNN2n3)[(dataNN2n3.size*.95)])
print "z1: %.4f to %.4f" % (np.sort(dataNN2z1)[(dataNN2z1.size*.05)], np.sort(dataNN2z1)[(dataNN2z1.size*.95)])
print "z2: %.4f to %.4f" % (np.sort(dataNN2z2)[(dataNN2z2.size*.05)], np.sort(dataNN2z2)[(dataNN2z2.size*.95)])
print "N: %.4f to %.4f" % (np.sort(dataNN2N)[(dataNN2N.size*.05)], np.sort(dataNN2N)[(dataNN2N.size*.95)])
print ""


