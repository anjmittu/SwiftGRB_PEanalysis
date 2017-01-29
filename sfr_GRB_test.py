import sys
import math
from matplotlib import pyplot as plt

def Rz(z,n):
    n0 = n[0]
    n1 = n[1]
    n2 = n[2]
    z1 = n[3]
    if(z <= z1):
	log_rho = n1*math.log10(1.0+z)+math.log10(n0);
    if(z > z1):
	log_rho = (n1-n2)*math.log10(1.0+z1)+n2*math.log10(1.0+z)+math.log10(n0);
        
    val = pow(10.0,log_rho);
    return val;

def RzTwoBreak(z,n):
    n0 = n[0]
    n1 = n[1]
    n2 = n[2]
    n3 = n[3]
    z1 = n[4]
    z2 = n[5]
    if(z <= z1):
	log_rho = n1*math.log10(1.0+z)+math.log10(n0);
    if(z > z1 and z <= z2):
	log_rho = (n1-n2)*math.log10(1.0+z1)+n2*math.log10(1.0+z)+math.log10(n0);
    if(z > z2):
	log_rho = (n2-n3)*math.log10(1.0+z2)+(n1-n2)*math.log10(1.0+z1)+n3*math.log10(1.0+z)+math.log10(n0);
        
    val = pow(10.0,log_rho);
    return val;

def sfr_Hopkins06(z):

	##sfr from Hopkins and Beacom 06, the Mod SalA fit
	a=0.0170; b=0.13; c=3.3; d=5.3;
        val=(a+b*z)/(1.0+pow((z/c),d));
	return val;

def sfr_Hopkins06_piecewise(z):
	
	##sfr from Hopkins and Beacom 06, piecewise fit
	z1 = 1.04;
        z2 = 4.48;
	n1 = 3.28;
	n2 = -0.26;
	n3 = -8.0;
        if(z <= z1):
        	#log_rho = 3.28*math.log10(1.0+z)-1.82;
		log_rho = n1*math.log10(1.0+z)-1.82;
        if(z > z1 and z <= z2):
	        #log_rho = -0.26*math.log10(1.0+z)-0.724;
		log_rho = (n1-n2)*math.log10(1.0+z1)+n2*math.log10(1.0+z)-1.82;
        if(z > z2):
	        #log_rho = -8.0*math.log10(1.0+z)+4.99;
		log_rho = (n2-n3)*math.log10(1.0+z2)+(n1-n2)*math.log10(1.0+z1)+n3*math.log10(1.0+z)-1.82;
        val = pow(10.0,log_rho);
	return val;

def snr(z): ## number/yr/Mpc^3
	
	## sfr from Horiuchi, Beacom, Dwek 09
	rho_0 = 0.0178; alpha = 3.4; beta = -0.3; gamma = -3.5;
        z1 = 1.0; z2 = 4.0;
        eta = -10.0;
        b = pow((1.0+z1),(1.0-alpha/beta));
        c = pow((1.0+z1),((beta-alpha)/gamma))*pow((1.0+z2),(1.0-beta/gamma));
        y1=pow((1.0+z),alpha*eta);
        y2=pow(((1.0+z)/b),beta*eta);
        y3=pow(((1.0+z)/c),gamma*eta);
        val = rho_0*pow((y1+y2+y3),1.0/eta);

	## change to snr
        val=0.007*val;
        return val;

def snr_upper(z): ## number/yr/Mpc^3

        ## sfr from Horiuchi, Beacom, Dwek 09
        rho_0 = 0.0213; alpha = 3.6; beta = -0.1; gamma = -2.5;
        z1 = 1.0; z2 = 4.0;
        eta = -10.0;
        b = pow((1.0+z1),(1.0-alpha/beta));
        c = pow((1.0+z1),((beta-alpha)/gamma))*pow((1.0+z2),(1.0-beta/gamma));
        y1=pow((1.0+z),alpha*eta);
        y2=pow(((1.0+z)/b),beta*eta);
        y3=pow(((1.0+z)/c),gamma*eta);
        val = rho_0*pow((y1+y2+y3),1.0/eta);

        ## change to snr
        val=0.007*val;
        return val;

def snr_lower(z): ## number/yr/Mpc^3

        ## sfr from Horiuchi, Beacom, Dwek 09
        rho_0 = 0.0142; alpha = 3.2; beta = -0.5; gamma = -4.5;
        z1 = 1.0; z2 = 4.0;
        eta = -10.0;
        b = pow((1.0+z1),(1.0-alpha/beta));
        c = pow((1.0+z1),((beta-alpha)/gamma))*pow((1.0+z2),(1.0-beta/gamma));
        y1=pow((1.0+z),alpha*eta);
        y2=pow(((1.0+z)/b),beta*eta);
        y3=pow(((1.0+z)/c),gamma*eta);
        val = rho_0*pow((y1+y2+y3),1.0/eta);

        ## change to snr
        val=0.007*val;
        return val;

def R_GRB(z):
        rate_GRB_0_global = 0.84/2; ## Gpc^{-3} yr^{-1}      

        z1_global = 3.60;
        n1_global = 2.07;
        n2_global = -0.70;

        if(z<=z1_global):
                val=pow((1.0+z),n1_global);
        else:
                val=pow((1+z1_global),(n1_global-n2_global))*pow((1+z),n2_global);

        val=val*rate_GRB_0_global;

        return val;

def R_GRB_test(z):
        rate_GRB_0_global = 1.01; ## Gpc^{-3} yr^{-1}      

        z1_global = 3.60;
        n1_global = 2.00;
        n2_global = -0.00;

        if(z<=z1_global):
                val=pow((1.0+z),n1_global);
        else:
                val=pow((1+z1_global),(n1_global-n2_global))*pow((1+z),n2_global);

        val=val*rate_GRB_0_global;

        return val;

def R_GRB_bestlow(z):
        #rate_GRB_0_global = 0.72; ## Gpc^{-3} yr^{-1}      
	#rate_GRB_0_global = 0.78; ## Gpc^{-3} yr^{-1} 
	rate_GRB_0_global = 0.75/2; ## Gpc^{-3} yr^{-1} 

        #z1_global = 3.60;
        #n1_global = 2.20;
        #n2_global = -3.50;
	#z1_global = 3.60;
        #n1_global = 2.10;
        #n2_global = -3.50;
	z1_global = 3.60;
        n1_global = 2.10;
        n2_global = -3.50;

        if(z<=z1_global):
                val=pow((1.0+z),n1_global);
        else:
                val=pow((1+z1_global),(n1_global-n2_global))*pow((1+z),n2_global);

        val=val*rate_GRB_0_global;

        return val;

def R_GRB_besthi(z):
        rate_GRB_0_global = 1.01/2; ## Gpc^{-3} yr^{-1}      

        z1_global = 3.60;
        n1_global = 1.95;
        n2_global = -0.00;

        if(z<=z1_global):
                val=pow((1.0+z),n1_global);
        else:
                val=pow((1+z1_global),(n1_global-n2_global))*pow((1+z),n2_global);

        val=val*rate_GRB_0_global;

        return val;


print 'Yuksel z = 0', snr(0.0)	
print 'Hopkins z = 0', sfr_Hopkins06_piecewise(0.0)

zi = 0.0
zf = 10.0
dz = 0.01
i_max = int((zf-zi)/dz)
z_array = []
sfr_hopkins06_z_array = []
snr_z_array = []
snr_z_low_array = []
snr_z_hi_array = []
R_GRB_z_array = []
R_GRB_z_test_array = []
R_GRB_z_bestlow_array = []
R_GRB_z_besthi_array = []
One_R_GRB_z_array = []
Two_R_GRB_z_array = []
Perley_GRB_z_array = []
Zero_R_GRB_z_array = []
ZeroHigh_R_GRB_z_array = []
ZeroLow_R_GRB_z_array = []
for i in range(0,i_max+1):
	z = i*dz
	snr_z = (1.11/2)/snr(0.0)*snr(z)
	snr_z_low = (1.11/2)/snr(0.0)*snr_lower(z)
	snr_z_hi = (1.11/2)/snr(0.0)*snr_upper(z)
	#sfr_hopkins06_z = R_GRB(0.0)/sfr_Hopkins06(0.0)*sfr_Hopkins06(z)
	sfr_hopkins06_z = (1.07/2)/sfr_Hopkins06_piecewise(0.0)*sfr_Hopkins06_piecewise(z)
	R_GRB_z = R_GRB(z)
	R_GRB_test_z = R_GRB_test(z)
	R_GRB_bestlow_z = R_GRB_bestlow(z)
	R_GRB_besthi_z = R_GRB_besthi(z)
        #One_R_GRB_z = Rz(z,[.513, 1.656, -5.997, 6.70, 4434])
        One_R_GRB_z = Rz(z,[.74, 1.68, -2.73, 6.82, 4434])
        #Two_R_GRB_z = RzTwoBreak(z, [.411, 1.878, .978, -8.804, 3.403, 6.600, 3919])
        Two_R_GRB_z = RzTwoBreak(z, [.72, 1.69, .42, -4.89, 5.46, 7.96, 4460])
        #Perley_GRB_z = Rz(z, [1.15, 1.39, -5.88, 5.44, 5294])
        Perley_GRB_z = Rz(z, [1.05, 1.37, -2.95, 6.01, 6190])
        Zero_R_GRB_z = RzTwoBreak(z, [.331, 3.28, -.26, -8, 1.04, 4.48, 1762])
        ZeroHigh_R_GRB_z = RzTwoBreak(z, [.401, 3.28, -.26, -8, 1.04, 4.48, 1762])
        ZeroLow_R_GRB_z = RzTwoBreak(z, [.271, 3.28, -.26, -8, 1.04, 4.48, 1762])
	#print R_GRB(0.0)

	z_array.append(z)
	snr_z_array.append(snr_z)
	snr_z_low_array.append(snr_z_low)
	snr_z_hi_array.append(snr_z_hi)
	R_GRB_z_array.append(R_GRB_z)
	R_GRB_z_test_array.append(R_GRB_test_z)
	R_GRB_z_bestlow_array.append(R_GRB_bestlow_z)
	R_GRB_z_besthi_array.append(R_GRB_besthi_z)
	sfr_hopkins06_z_array.append(sfr_hopkins06_z)
	One_R_GRB_z_array.append(One_R_GRB_z)
        Two_R_GRB_z_array.append(Two_R_GRB_z)
        Perley_GRB_z_array.append(Perley_GRB_z)
        Zero_R_GRB_z_array.append(Zero_R_GRB_z)
        ZeroHigh_R_GRB_z_array.append(ZeroHigh_R_GRB_z)
        ZeroLow_R_GRB_z_array.append(ZeroLow_R_GRB_z)

	##print z, '%e' % snr_z, '%e' % R_GRB_z

print R_GRB_besthi(3.6)
print R_GRB(3.6)

z_marker = []
sfr_hopkins06_z_marker = []
### this is just to make the plot prettier.....
for i in range(0,100+1):
        z = i*10.0/100.0
        sfr_hopkins06_z = (1.07/2)/sfr_Hopkins06_piecewise(0.0)*sfr_Hopkins06_piecewise(z)

        z_marker.append(z)
        sfr_hopkins06_z_marker.append(sfr_hopkins06_z)


fig = plt.figure()
fig.set_rasterized(True)

plt.plot(z_array,R_GRB_z_besthi_array,'r:')
plt.plot(z_array,R_GRB_z_bestlow_array,'r:')
plt.fill_between(z_array,R_GRB_z_besthi_array,R_GRB_z_bestlow_array,color='r',alpha=0.25)
#plt.plot(z_array,R_GRB_z_test_array,'k',label='GRB Rate test')
plt.plot(z_array,One_R_GRB_z_array,'black',linewidth=1.5,label='GRB Rate from this research (One-Break)')
plt.plot(z_array,Perley_GRB_z_array,'orange',linewidth=1.5,label='GRB Rate from this perley (One-Break)')
plt.plot(z_array,Two_R_GRB_z_array,'yellow',linewidth=1.5,label='GRB Rate from this research (Two-Break)')
plt.plot(z_array,Zero_R_GRB_z_array,'cyan',linewidth=1.5,label='GRB Rate from this research (using parameters from Hopkins & Beacom 2006)')
plt.plot(z_array,ZeroHigh_R_GRB_z_array,'c:')
plt.plot(z_array,ZeroLow_R_GRB_z_array,'c:')
plt.fill_between(z_array,ZeroHigh_R_GRB_z_array,ZeroLow_R_GRB_z_array,color='c',alpha=0.25)
plt.plot(z_array,R_GRB_z_array,'r',linewidth=1.5,label='GRB Rate from Lien et al. (2014) (without Luminosity Evolution)')
plt.plot(z_array,snr_z_hi_array,'b:')
plt.plot(z_array,snr_z_low_array,'b:')
plt.fill_between(z_array,snr_z_hi_array,snr_z_low_array,color='b',alpha=0.25)
plt.plot(z_array,snr_z_array,'b--',linewidth=1.5,label='GRB Rate $\propto$ SFR from Yuksel et al. 2008 (with Luminosity Evolution)')
#plt.plot(z_array,sfr_hopkins06_z_array,'yellow',linewidth='3.5',label='GRB Rate $\propto$ SFR from Hopkins & Beacom 2006 (with Luminosity Evolution)')
plt.plot(z_marker,sfr_hopkins06_z_marker,'green',linewidth=1.5,marker='o',markersize=2.0,label='GRB Rate $\propto$ SFR from Hopkins & Beacom 2006 (with Luminosity Evolution)')
plt.yscale('log')
plt.ylim(1.0e-2,1.0e+3)
plt.xlabel('Redshift z')
plt.ylabel('GRB Rate [Gpc$^{-3}$ yr$^{-1}$]')
plt.legend(prop={'size':10.5})

#plt.savefig('snr_GRB_paper.eps')
plt.show()
