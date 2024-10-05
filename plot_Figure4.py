import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import pickle

case_name = ['RH54', 'RH34', 'RH74']

# Read data
fname = 'fort_RH54.98'
th, P, qv = np.loadtxt(fname, skiprows = 262, max_rows = 71, usecols = (2, 3, 5), unpack = True); P = P[:-1]/1e2
xc = np.arange(1024)/10

with open('data_CldStructure_snapshot_RH54.pkl', 'rb') as f:
	dic = pickle.load(f)

t   = dic['time']
cld = dic['cld']
u   = dic['u']
w   = dic['w']
uu  = dic['u_interp']
ww  = dic['w_interp']
prec= dic['prec']
zt  = dic['zt']/1e3
z2  = dic['z_interp']

ws = (uu**2+ww**2)**(1/2); print(np.nanmax(ws))

# Plot setting
x0 = 560
x1 = 720

def plot_(peak, ax):
    ax[1].set_xlabel('[km]',       fontsize = 18)
    ax[0].tick_params(labelsize = 15)
    ax[1].tick_params(axis = 'x', labelsize = 15)
    ax[1].tick_params(axis = 'y', color = 'navy', labelcolor = 'navy', labelsize = 15)
    ax[0].set_ylim([0, 15])
    ax[0].set_yticks(np.linspace(0, 15, 4))
    ax[1].set_ylim([0, 149])
    ax[0].set_xlim([xc[x0], xc[x1]])
    ax[1].set_xlim([xc[x0], xc[x1]])
    ax[0].set_xticks([])
    ax[1].set_xticks([x0/10, 64.0, x1/10])
    ax[1].grid(lw = 0.5, ls = ':', c = 'grey')
    secax = ax[0].secondary_yaxis('right', functions=(forward, inverse))
    secax.set_ticks(np.arange(1000, 0, -200))
    secax.tick_params(labelsize = 15)
    if peak == 0:
        ax[0].set_ylabel('Height [km]', fontsize = 18)
        ax[1].set_ylabel('[mm h$^{-1}$]', color = 'navy', fontsize = 18)
    else:
        secax.set_ylabel('Pressure [hPa]', fontsize = 18)

def forward(x):
    return np.interp(x, zt, P)
def inverse(x):
    return np.interp(x, P[::-1], zt[::-1])

# Plot

k = 0

fig  = plt.figure(figsize = (14, 9))
for peak in range(2):
	ax0  = fig.add_axes([0.06+0.47*peak, 0.37 , 0.39, 0.569])
	ax0.set_facecolor('0.4')
	CS1 = ax0.contourf(xc, zt, u[peak], cmap = cm.PiYG_r, levels = np.linspace(-4, 4, 41), extend = 'both')
	qq_ = np.copy(cld[peak]); qq_[qq_ < 1e-5] = np.nan
	CS2 = ax0.pcolormesh(xc, zt, qq_*1e3, cmap = cm.jet, vmax = 5.8, vmin = 0, alpha = 0.8)
	ax0.contour(xc, zt, cld[peak] >= 1e-5, colors = '0.2', levels = [0.5])
	ws[peak,0,x0] = 25
	CS3 = ax0.streamplot(xc[x0:x1+1], z2, uu[peak][:,x0:x1+1], ww[peak][:,x0:x1+1], density = (4, 1.8), linewidth = 0.8, color = ww[peak][:,x0:x1+1], cmap = cm.Greys, norm = colors.Normalize(vmin = 0, vmax = 13.5))
	ax0.contour(xc, zt, w[peak] > 1, colors = 'maroon', levels = [0.5], linewidths = 2)
	ax1 = fig.add_axes([0.06+0.47*peak, 0.195, 0.39, 0.175])
	ax1.plot(xc, prec[peak], 'navy', lw =  2)
	ax0.set_title(' Peak%i'%(peak+1), loc = 'left', fontsize = 20, weight = 'heavy')
	ax0.set_title('%02d:%02d / %s'%(t[peak]//30+6, (t[peak]%30)*2, case_name[k]), loc = 'right', fontsize = 15)
	plot_(peak, [ax0, ax1])

ax_cbar = fig.add_axes([0.015, 0.08, 0.30, 0.02])
cbar = plt.colorbar(CS2, orientation = 'horizontal', cax = ax_cbar)
cbar.ax.set_xlabel('Cloud [g kg$^{-1}$]', fontsize = 18)
cbar.ax.tick_params(labelsize = 15)

ax_cbar = fig.add_axes([0.35 , 0.08, 0.30, 0.02])
cbar = plt.colorbar(CS1, orientation = 'horizontal', cax = ax_cbar)
cbar.ax.set_xlabel('Inflow [m s$^{-1}$]', fontsize = 18)
cbar.set_ticks(np.arange(-4, 5, 1))
cbar.ax.tick_params(labelsize = 15)

ax_cbar = fig.add_axes([0.685, 0.08, 0.30, 0.02])
cbar = plt.colorbar(CS3.lines, orientation = 'horizontal', cax = ax_cbar)
cbar.ax.set_xlabel('Updraft [m s$^{-1}$]', fontsize = 18)
cbar.set_ticks(np.arange(0, 13, 4))
cbar.ax.tick_params(labelsize = 15)

plt.savefig('Figure4.png', dpi = 300)
plt.close()
