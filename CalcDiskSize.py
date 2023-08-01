from __future__               import division
from   scipy.special          import spence
from   numpy                  import *
import matplotlib.pyplot      as plt
import matplotlib.patheffects as pe
import sys

# ------------------------------------------------------------------------------------------
# ---  Simple script to compute the effect spurious heating on a stellar disk embedded
# ---  within an spherically symmetric NFW halo with isotropic DM particle velocities.
# ---  The number of DM particles within r200, N200, is determined by the halo mass and
# ---  the assumed DM particle mass. The empirical model predicts how scattering of
# ---  stellar particles off DM particles increases their velocity dispersion in the
# ---  (phi,z,R) directions over a time interval t. The model can be applied at any
# ---  galacto-centric radius (below we choose r50, the typical half-stellar mass radius,
# ---  and R=8 [kpc].
# ---  
# ---  For model details, see Ludlow et al (2021); Wilkinson et al (2023) 
# ---  For an application to cosmological simulations, see Ludlow et al (2023)
# ------------------------------------------------------------------------------------------

# ---  Set some preliminary numbers

# ---  [M] = [10^10 Msun/h]
# ---  [L] = [Mpc/h]
# ---  [V] = [km/s]

sec_per_Gyr= 3.15576e16
km_per_Mpc = 3.0857e19
Gc         = 43.0091 
hubble     = 0.6777
z          = 0.             # redshift (model applies at z>0 as well)
Omega      = [0.307,0.693]  # Density parameters (OmegaM,OmegaLambda)

# --- Some relevant functions ---
def rho_NFW(r,rhos,rs):
    return rhos / (r/rs*(1.+r/rs)**2)

def fc(x):
    return 1./(log(1. + x) - x / (1. + x))

def sig1D_nfw(r,rhos,rs,c):
    # 1D velocity dispersion profile for isotropic, non-rotatin, NFW halo (Lokas & Mamon, 2001)
    x   = r / (rs*c)
    V200= sqrt(Gc * (4/3*pi*(rs*c)**3 * 200 * rhocrit(z,Omega)) / (rs*c))
    f1  = 0.5 * fc(c) * x * (1 + c*x)**2
    f2  = c**2 * (pi**2 - log(c*x))
    f3  = c/x + c**2/(1 + c*x)**2 + 6*c**2/(1 + c*x)
    f4  = c**2 + 1/x**2 - 4*c/x - 2*c**2/(1 + c*x)
    L12 = spence(1+c*x)
    sig = f1 * (f2 - f3 + f4 * log(1 + c*x) + 3 * (c*log(1 + c*x))**2 + 6*c**2*L12)
    sig = V200 * sqrt(sig)
    return sig

def rhocrit(z, Om):
    opz = (1.+z)
    Ez2 = Om[0]*opz**3 + Om[1] + (1.-Om[0]-Om[1])*opz**2
    return 3. / (8.*pi*Gc) * 1e4 * Ez2

# --- Open a figure for sigma_*(M200,z) ---
plt.clf() ; plt.ion()
fig  = plt.figure(1, figsize=(1,1))
fig.set_size_inches(7, 3, forward=True)
pos  = [0.12,0.15,0.40,0.83] ; ax1  = fig.add_axes(pos)
pos  = [0.53,0.15,0.40,0.83] ; ax2  = fig.add_axes(pos)
axu  = [ax1,ax2]
xmin = -2.5 ; xmax = 5 ; ymin = -0.2 ; ymax = 0.3
for ax in axu:
    ax.set_xlim(xmin, xmax) ; ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r'$\log\,M_{200}\,[10^{10} h^{-1}{\rm M_\odot]}$',fontsize=13)
ax1.set_ylabel(r'$\log\,\sigma_{\rm tot}/V_{200}$', fontsize=13)
plt.setp(ax2.get_yticklabels(),visible=False)

# --- Read tabulated c(M,z) relation for Planck (Ludlow et al 2016)
# --- These tables can be updated to use your preferred c(M200) relation
f          = 'Planck_cmz_z0z1z2z3.dat'
redshift   = genfromtxt(f, dtype=float)[:,0]
lgM        = genfromtxt(f, dtype=float)[:,1]
cnfw       = genfromtxt(f, dtype=float)[:,3]
# --- Interpolate: arrays are sparsely sampled from the Earth Mass to cluster masses
Nm_arr     = 80
lgM200_min =-3  # min log10(M200 / [10^10 Msun/h])
lgM200_max = 5  # max log10(M200 / [10^10 Msun/h])
M200       = 10.**linspace(lgM200_min,lgM200_max,Nm_arr)
cnfw       = interp(M200, 10.**lgM[where(redshift==z)[0]],cnfw[where(redshift==z)[0]]) 

# --- Compute other halo properties (assuming NFW profile)
rhoc       = rhocrit(z,Omega)
r200       = (3 * M200 / (800 * pi * rhoc))**0.3333
V200       = sqrt(Gc * M200 / r200)
rs         = r200 / cnfw
rhos       = M200 / (4.*pi*rs**3 * (log(1. + cnfw) - cnfw / (1. + cnfw)))

# --- Appoint a haracteristic radii for galaxies
r50        = 0.22 * rs                # --- stellar half-mass radii (see Ludlow et al 2023); Modify as appropriate

# ------------------------------------------------------------------------------------------
# --- DM halo profiles can be modified by baryons, which can increase or decrease the
# --- central DM density. Various empirical models have been published claiming to
# --- approximate the outcome of these effects in cosmological simulations. We found
# --- it unnecessary to model them in our analysis, but this can be done. Once a "modified"
# --- DM halo has been modelled, the DM halo velocity dispersion must be calculated from,
# --- e.g., Jeans' equations; we use the velocity dispersion for an spherical, isotropic,
# --- NFW profile (Lokas & Mamon, 2002)
# ---
# --- Note that, on broad mass scale, the net effect of baryons on DM halo structure is
# --- very sensitive to spurious collisional heating (see Ludlow et al, 2023)
# ------------------------------------------------------------------------------------------

# --- Plot the spurious velocity dispersions from Ludlow et al (2021) & Wilkinson et al (2023)
lnk0  = [9.40,   20.19, 20.17]        # --- sigma_{phi,z,R}: best-fit parameters from Wilkinson et al (2023)
alpha = [-0.115,-0.308,-0.189]
sig0  = [0,0,0]                       # --- initial disc velocity dispersion in (phi,z,R); assume initially "cold"
mp_dm = [0.000657369, 0.000657369/7.] # --- DM particle masses for Eagle simulationes used in Ludlow et al (2023)
col   = ['salmon','dodgerblue']

# --- DM density and vel. disp. at r=r50 and r=8 [kpc]
rhor50_DM     = rho_NFW(r50,            rhos,rs)      # DM halo density    at r=r50
sigr50_DM     = sig1D_nfw(r50,          rhos,rs,cnfw) # DM halo vel. disp. at r=r50
rho8kpc_DM    = rho_NFW(8.*hubble/1e3,  rhos,rs)      # DM halo density    at r=8 kpc
sig8kpc_DM    = sig1D_nfw(8.*hubble/1e3,rhos,rs,cnfw) # DM halo vel. disp. at r=8 kpc
times         = 7.0                                   # age of galaxy or stellar population [Gyr]

# --- If non-zero, plot the initial total velocity dispersion of the stellat disc
if sum(asarray(sig0)**2)==0:
    lab = ''
else:
    lab = r'$\sigma_0$'
ax1.plot(log10(M200),log10(sqrt(sum(asarray(sig0)**2))/V200),color='lightgrey',linestyle='--',label=lab)
ax2.plot(log10(M200),log10(sqrt(sum(asarray(sig0)**2))/V200),color='lightgrey',linestyle='--')

# --- plot total 1D vel. disp. at r = r50  as a function of M200
ax1.plot(log10(M200),log10(sqrt(3)*sig1D_nfw(r50,rhos,rs,cnfw)/V200),          color='k',linewidth=1,label=r'$\sigma_{\rm DM}$')
# --- plot total 1D vel. disp. at r = 8kpc as a function of M200
ax2.plot(log10(M200),log10(sqrt(3)*sig1D_nfw(8.*hubble/1e3,rhos,rs,cnfw)/V200),color='k',linewidth=1)

# --- compute and plot the spurious stellar velocity dispersions 
lab      = [r'$m_{\rm DM}=6.6\times 10^6\,{\rm M_\odot}/h$',r'$m_{\rm DM}=9.4\times 10^5\,{\rm M_\odot}/h$']
for im, mp in enumerate(mp_dm):

    # plot spurios vel. disp. at r50
    tc        = V200**3 / (rhor50_DM * Gc**2 * mp) * hubble * km_per_Mpc / sec_per_Gyr
    tvir_phi  = tc / (sqrt(2) * pi * lnk0[0] * (rhor50_DM / (200 * rhoc))**alpha[0] * (V200/sigr50_DM)**2)
    tvir_z    = tc / (sqrt(2) * pi * lnk0[1] * (rhor50_DM / (200 * rhoc))**alpha[1] * (V200/sigr50_DM)**2)
    tvir_R    = tc / (sqrt(2) * pi * lnk0[2] * (rhor50_DM / (200 * rhoc))**alpha[2] * (V200/sigr50_DM)**2)
    sig_spur  = sqrt(3*sigr50_DM**2 -
                     (sigr50_DM**2  - sig0[0]**2)*exp(-(times/tvir_phi)) - 
                     (sigr50_DM**2  - sig0[1]**2)*exp(-(times/tvir_z))   -
                     (sigr50_DM**2  - sig0[2]**2)*exp(-(times/tvir_R))) / V200

    ax1.plot(log10(M200),log10(sig_spur),color=col[im],linewidth=1,zorder=2,label=lab[im],
             path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])

    # plot spurios vel. disp. at 8 kpc
    tc        = V200**3 / (rho8kpc_DM * Gc**2 * mp) * hubble * km_per_Mpc / sec_per_Gyr
    tvir_phi  = tc / (sqrt(2) * pi * lnk0[0] * (rho8kpc_DM / (200 * rhoc))**alpha[0] * (V200/sig8kpc_DM)**2)
    tvir_z    = tc / (sqrt(2) * pi * lnk0[1] * (rho8kpc_DM / (200 * rhoc))**alpha[1] * (V200/sig8kpc_DM)**2)
    tvir_R    = tc / (sqrt(2) * pi * lnk0[2] * (rho8kpc_DM / (200 * rhoc))**alpha[2] * (V200/sig8kpc_DM)**2)
    sig_spur  = sqrt(3*sig8kpc_DM**2 -
                     (sig8kpc_DM**2  - sig0[0]**2)*exp(-(times/tvir_phi)) - 
                     (sig8kpc_DM**2  - sig0[1]**2)*exp(-(times/tvir_z))   -
                     (sig8kpc_DM**2  - sig0[2]**2)*exp(-(times/tvir_R))) / V200 

    ax2.plot(log10(M200),log10(sig_spur),color=col[im],linewidth=1,zorder=2,
             path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])
    
# --- Add a few annotations to the plot
labels    = [r'$\sigma_{\rm tot}(r_{50})$',r'$\sigma_{\rm tot}(r=8\,{\rm kpc})$']
for i, ax in enumerate(axu):
    ax.annotate(labels[i],[0.05,0.90],xycoords='axes fraction',textcoords='axes fraction',color='k',fontsize=12)

ax1.legend(frameon=True,fancybox=True,borderpad=0.5,shadow=False,prop={'size':9},loc=(0.3,0.68),framealpha=0,handlelength=3)

plt.show()
sys.exit()            
