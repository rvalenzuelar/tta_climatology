import numpy as np
import parse_data
import matplotlib.pyplot as plt


fig1,ax1=plt.subplots(7,2,figsize=(10,11))
fig2,ax2=plt.subplots(7,2,figsize=(10,11))
fig3,ax3=plt.subplots(7,2,figsize=(10,11))
ax1=ax1.flatten()
ax2=ax2.flatten()
ax3=ax3.flatten()
ax=np.vstack((ax1,ax2,ax3))

var=['precip','wspd','wdir']

for n,y in enumerate([0,1998]+range(2001,2013)):
	if n==0:
		ax[0,n].axis('off')
		ax[1,n].axis('off')
		ax[2,n].axis('off')
	else:
		print y
		bby=parse_data.bby(y)
		for i,v in enumerate(var):
			bby[v].plot(ax=ax[i,n])
			ax[i,n].set_xticklabels('')
			ax[i,n].text(0.8,1.0,str(y),transform=ax[i,n].transAxes)

fig1.suptitle('BBY Precipitation [mm]')
fig2.suptitle('BBY Surface wind speed [m s^-1]')
fig3.suptitle('BBY Surface wind direction [deg]')

fig1.subplots_adjust(bottom=0.05,top=0.95, left=0.05,right=0.95)
fig2.subplots_adjust(bottom=0.05,top=0.95, left=0.05,right=0.95)
fig3.subplots_adjust(bottom=0.05,top=0.95, left=0.05,right=0.95)

plt.show(block=False)


