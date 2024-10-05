import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from scipy import integrate

case_name = ['RH54', 'RH34', 'RH74']

#### Figure 3a ==========
# Read data 
fname = 'fort_RH54.98'
th, P, qv = np.loadtxt(fname, skiprows = 262, max_rows = 71, usecols = (2, 3, 5), unpack = True); P = P[:-1]/1e2
zt = np.loadtxt(fname, skiprows = 188, max_rows = 71, usecols = (2,), unpack = True)[:-1]/1e3
xc = np.arange(1024)/10
xx, zz = np.meshgrid(xc, zt)

with open('data_BLstructure_RH54.pkl', 'rb') as f:
    dic = pickle.load(f)

hm   = np.roll(dic['hm'], -256, axis = -1)
qv   = np.roll(dic['qv'], -256, axis = -1)
cld  = np.roll(dic['cld'], -256, axis = -1)
u    = np.roll(dic['u'], -256, axis = -1)
w    = np.roll(dic['w'], -256, axis = -1)
prec = np.roll(dic['prec'], -256, axis = -1)

# Plot setting
def plot_(t, ax):
    ax[0].tick_params(labelsize = 15)
    ax[1].tick_params(axis = 'x', labelsize = 15)
    ax[1].tick_params(axis = 'y', color = 'navy', labelcolor = 'navy', labelsize = 15)
    ax[0].set_ylim([0, 5])
    ax[1].set_ylim([0, 39])
    ax[0].set_xlim([xc[0], xc[-1]])
    ax[1].set_xlim([xc[0], xc[-1]])
    ax[1].set_xticks(np.linspace(0, 102.4, 5))
    ax[0].set_xticks([])
    secax = ax[0].secondary_yaxis('right', functions=(forward, inverse))
    secax.tick_params(labelsize = 15)
    if t//2 == 2:
        ax[1].set_xlabel('[km]', fontsize = 18)
    if t%2 == 0:
        ax[0].set_ylabel('Height [km]', fontsize = 18)
        ax[1].set_ylabel('[mm h$^{-1}$]', color = 'navy', fontsize = 18)
    else:
        secax.set_ylabel('Pressure [hPa]', fontsize = 18)

def forward(x):
    return np.interp(x, zt, P)
def inverse(x):
    return np.interp(x, P[::-1], zt[::-1])

# Plot
fig = plt.figure(figsize = (14, 21))

# (a)
i = 0
ii = 32; scale = 50
ii = 64; scale = 30
time_list = [90, 120, 150, 180, 195, 210]
for t in range(6):
    time = time_list[t]
    H = time//30 + 6
    M = 2*(time%30)
    ax0  = fig.add_axes([0.06+0.42*(t%2), 0.811-0.234*(t//2), 0.35, 0.15])
    ax0.set_facecolor('0.92')
    CS = ax0.contourf(xc, zt, hm[t], cmap = cm.jet, levels = np.linspace(330, 350, 21), extend = 'both')
    ax0.contour(xc, zt, qv[t]*1e3, colors = 'w', levels = np.linspace(0, 20, 11))
    ax0.contour(xc, zt, cld[t], colors = 'silver', levels = [-1, 1e-5], linewidths = 2)
    q  = ax0.quiver(xc[::ii], zt, u[t][:, ::ii], w[t][:, ::ii], color = 'k', scale = scale, zorder = 10, angles = 'xy')
    ax0.quiverkey(q, 0.05, 1.04, 1, '1 m/s', labelpos = 'E', fontproperties = {'size': 15})
    ax1 = fig.add_axes([0.06+0.42*(t%2), 0.811-0.045-0.234*(t//2), 0.35, 0.045])
    ax1.plot(xc, prec[t], 'navy')
    ax0.set_title('%02d:%02d'%(H, M), fontsize = 20, weight = 'heavy')
    ax0.set_title(case_name[i], fontsize = 15, loc = 'right')
    ax1.grid(lw = 0.5, ls = ':', c = 'grey')
    plot_(t, [ax0, ax1])
    if t == 0: ax0.text(-14.8, 5.75, '(a) Time evolution', fontsize = 25.2, weight = 'heavy')

ax_cbar = fig.add_axes([0.94, 0.343, 0.02, 0.618])
cbar = plt.colorbar(CS, orientation = 'vertical', cax = ax_cbar)
cbar.ax.set_title('MSE [K]', fontsize = 18)
cbar.ax.tick_params(labelsize = 15)

### Figure 3b ==========

# Read data
with open('data_BLstructure_iniPeaks_RH54.pkl', 'rb') as f:
        dic = pickle.load(f)
hm = np.roll(dic['hm'], -256, axis = -1)
qv = np.roll(dic['qv'], -256, axis = -1)
u  = np.roll(dic['u'], -256, axis = -1)
w  = np.roll(dic['w'], -256, axis = -1)

# Calc stream function
u_ = np.copy(u); u_[np.isnan(u_)] = 0
w_ = np.copy(w); w_[np.isnan(w_)] = 0
u_ = u_[:,:,::-1]
w_ = w_[:,:,::-1]
zz_= zz[::-1]

psi = np.zeros(hm.shape)
for k in range(len(case_name)):
  for peak in range(2):
    intx = integrate.cumtrapz(w_[k,peak], xx,  axis = 1, initial = 0)
    intz = integrate.cumtrapz(u_[k,peak], zz_, axis = 0, initial = 0)
    psi[k,peak] = -intx + intz

psi = psi[:,:,::-1]
psi[np.isnan(qv)] = np.nan

# Plot setting
def plot_(k, ax):
    ax[0].tick_params(labelsize = 15)
    ax[0].set_xlim([xc[0], xc[-1]])
    ax[0].set_xticks(np.linspace(0, 102.4, 5))
    ax[0].set_ylim([0, 6.5])
    secax = ax[0].secondary_yaxis('right', functions=(inverse, forward))
    secax.tick_params(labelsize = 15)
    ax[0].set_xlabel('[km]', fontsize = 18)
    if k%2 == 0:
        ax[0].set_ylabel('Height [km]', fontsize = 18)
    else:
        secax.set_ylabel('Pressure [hPa]', fontsize = 18)

def forward(x):
    return np.interp(x, P[::-1], zt[::-1])
def inverse(x):
    return np.interp(x, zt, P)

# Plot (b)

ax1 = fig.add_axes([0.06, 0.031, 0.35, 0.195])
ax2 = fig.add_axes([0.48, 0.031, 0.35, 0.195])
ax  = [ax1, ax2]

peak_time = [90, 160]
i  = 64
i0 = [0, 16]
cmap = cm.jet

for peak in range(2):
  for k in range(1):
    ax[peak].set_facecolor('0.82')
    CS   = ax[peak].contourf(xc, zt, hm[k,peak], cmap = cmap, levels = np.linspace(330, 350, 21), extend = 'both')
    ax[peak].contour(xc, zt, psi[k,peak], colors = 'k', linewidths = 1.8, levels = [-2.4, -1.6, -0.8, 0.8, 1.6, 2.4], alpha = 0.8)
    q = ax[peak].quiver(xc[i0[peak]::i], zt, u[k,peak][:, i0[peak]::i], w[k,peak][:, i0[peak]::i], color = '0.44', scale = 20, zorder = 10, angles = 'xy', width = 0.0035)
    ax[peak].quiverkey(q, 0.85, 1.025, 1, '1 m/s', labelpos = 'E', fontproperties = {'size': 15})
    plot_(peak, [ax[peak]])
    t1 = peak_time[peak]
    t2 = peak_time[peak]+10
    ax[peak].set_title('%s: %02d:%02d - %02d:%02d'%(case_name[k], t1//30+6, 2*(t1%30), t2//30+6, 2*(t2%30)), fontsize = 15, loc = 'left')
    ax[peak].text(4, 5.76, 'Peak%i'%(peak+1), color = 'w', fontsize = 20, weight = 'heavy')

ax[0].text(-14.8, 7.2, '(b) Structures of local circulation', weight = 'heavy', fontsize = 25.2)
ax_cbar = fig.add_axes([0.94, 0.031, 0.02, 0.195])
cbar = plt.colorbar(CS, orientation = 'vertical', cax = ax_cbar)
cbar.ax.set_title('MSE [K]', fontsize = 18)
cbar.ax.tick_params(labelsize = 15)
cbar.set_ticks(np.arange(330, 350, 3))

plt.savefig('Figure3.png', dpi = 300)
plt.close()
