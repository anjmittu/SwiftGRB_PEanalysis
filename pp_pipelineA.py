import numpy as np
import os
import triangle
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib.ticker import MultipleLocator

def RunAnalysis(outfile, methd):
	cmd1 = './Analysis --file=support_data/FynboGRB_lum_z_Zonly.txt --method='+str(methd)+' --nlive=2000 --varyz1 '
        cmd1 += '--outfile=chains/RD_'+str(outfile)+'Onebreak_ --silent'
        cmd2 = './Analysis --file=support_data/FynboGRB_lum_z_Zonly.txt --method='+str(methd)+' --nlive=2000 --varyz1 '
        cmd2 += '--logz --outfile=chains/RD_'+str(outfile)+'Onebreak_logz_ --silent'
        cmd3 = './Analysis --file=support_data/FynboGRB_lum_z_Zonly.txt --method='+str(methd)+' --nlive=2000 --twobreak --varyz1 '
        cmd3 += '--varyz2 --outfile=chains/RD_'+str(outfile)+'Twobreak_ --silent'
        cmd4 = './Analysis --file=support_data/FynboGRB_lum_z_Zonly.txt --method='+str(methd)+' --nlive=2000 --twobreak --varyz1 '
        cmd4 += '--varyz2 --logz --outfile=chains/RD_'+str(outfile)+'Twobreak_logz_ --silent'
	# run the analysis
	os.system(cmd1)
        os.system(cmd2)
        os.system(cmd3)
        os.system(cmd4)
	
	


from optparse import OptionParser
parser=OptionParser()
parser.add_option("-n","--number",action="store",type="int",default=100,help="Number of analyses to run")
parser.add_option("-o","--outdir",action="store",type="string",default="chains",help="Directory for analyses' output")
parser.add_option("-l","--nlive",action="store",type="int",default=1000,help="Number of live points to use")
parser.add_option("-r","--resume",action="store_true",default=False,help="Resume previous run")
(opts,args)=parser.parse_args()


# loop to perform all analyses
RunAnalysis("NN_",0)
RunAnalysis("RF_",1)
RunAnalysis("AB_",2)


# plot posterior results
cmd5 = 'python posterior_plotsOneBreak.py'
cmd6 = 'python posterior_plotsOneBreak.py --outdir="Onebreak_logz_"'
cmd7 = 'python posterior_plotsTwoBreak.py'
cmd8 = 'python posterior_plotsTwoBreak.py --outdir="Twobreak_logz_"'

os.system(cmd5)
os.system(cmd6)
os.system(cmd7)
os.system(cmd8)

