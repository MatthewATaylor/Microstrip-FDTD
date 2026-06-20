import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy as sp
from pysmithchart import R_DOMAIN
from tqdm import tqdm
import skrf as rf



def set_mpl_params():
    mpl.rcParams['mathtext.default'] = 'regular'
    mpl.rcParams.update({'font.size': 13})
set_mpl_params()



# User-specified parameters

ANIM_RATIO = 10
LOAD_ENABLE = False

eps0 = 8.854188e-12
epsr_pcb = 4.4
eps_pcb = epsr_pcb*eps0
sig_pcb = 0.01  # [S/m]
mu0 = 1.256637e-6
c = 1 / np.sqrt(eps0*mu0)
v_pcb = 1 / np.sqrt(eps_pcb*mu0)
f_r = 2.45e9
lambda_r = v_pcb/f_r

# Excitation
tau_src = 6.5e-11  # Gaussian first derivative parameter (2.45 GHz frequency peak)
t0_src = 2.5e-10  # Shift the pulse back to the start of the negative spike

Z0 = 50
INPUT_FEED_W_m = 3.35e-3  # Should provide 50 ohms
ZL = 50
SERIES_INDUCTANCE = 1e-9  # [H]
BOARD_H_m = 1.6e-3

PML_D_m = 5e-3
VAC_D_m = 3.5e-3

FEED_W = 4  # Number of cells used to exactly represent feedline width (use to adjust cell size)
FEED_W_m = 3.35e-3
FEED_L_m = 10e-3

STUB_L_m = 28.299671824471e-3
FEED_HI_L_m = 11.920613206434e-3

PATCH_MARGIN_m = 3e-3
PATCH_W_m = 31e-3
CORNER_TRUNCATION_m = 3.8e-3

BOARD_H = 3

dx = FEED_W_m / FEED_W
dy = dx
dz = BOARD_H_m / BOARD_H

INDUCTOR_L_m = 0.96e-3
INPUT_FEED_L_m = dy*1
INPUT_FEED_W = int(np.round(INPUT_FEED_W_m/dx))

BOARD_W_m = PATCH_W_m + 2*PATCH_MARGIN_m + STUB_L_m - PATCH_W_m/2 + FEED_W_m/2
BOARD_L_m = PATCH_MARGIN_m + PATCH_W_m + FEED_L_m + FEED_HI_L_m + INDUCTOR_L_m + INPUT_FEED_L_m + dy

NX = int(np.round((BOARD_W_m + 2*PML_D_m + 2*VAC_D_m) / dx))
NY = int(np.round((BOARD_L_m + 2*PML_D_m + 2*VAC_D_m) / dy))
NZ = int(np.round((BOARD_H_m + PML_D_m + VAC_D_m) / dz))
SS = (NX+1)*(NY+1)*(NZ+1)

dt = 1 / (c * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2))
NT = int(15e-9/dt)

print('Simulation discretization:')
print(f'\tSS = {NX+1}x{NY+1}x{NZ+1} = {SS}')
print(f'\tdt = {dt} s')
print(f'\tNT = {NT}')
print()




def get_animation(anim_frames):
    fig_anim, ax_anim = plt.subplots(1, 1, figsize=(7,6))
    def get_surf_anim(frame):
        return ax_anim.imshow(
            frame.T,
            vmin=np.min(anim_frames)*0.2, vmax=np.max(anim_frames)*0.2,
            cmap='viridis',
            origin='lower',
            extent=[0, (NX+1)*dx*1e2, 0, (NY+1)*dy*1e2],
            aspect='equal'
        )
    surf_anim = get_surf_anim(anim_frames[0])
    cbar_anim = plt.colorbar(surf_anim, ax=ax_anim, label='Ez [V/m]')
    def animate(n):
        t = n*dt*ANIM_RATIO
        ax_anim.clear()
        surf_anim = get_surf_anim(anim_frames[n])
        ax_anim.set_xticks(np.arange(0, int((NX+1)*dx*1e2), 1))
        ax_anim.set_yticks(np.arange(0, int((NY+1)*dy*1e2), 1))
        ax_anim.tick_params(which='minor', bottom=False, left=False)
        #for i in range(NX+1):
        #    ax_anim.axvline(i*dx*1e2, color='black', linewidth=0.5)
        #for i in range(NY+1):
        #    ax_anim.axhline(i*dy*1e2, color='black', linewidth=0.5)
        ax_anim.set_title(f'Time = {1e9*t:.2f} ns')
        ax_anim.set_xlabel('X [cm]')
        ax_anim.set_ylabel('Y [cm]')
        fig_anim.tight_layout()
        return surf_anim,
    return matplotlib.animation.FuncAnimation(
        fig_anim,
        animate,
        frames=int(np.ceil(NT/ANIM_RATIO)),
        interval=0.1*1000
    )



def plot_s11(freqs, s11s, labels, colors):
    fig, ax = plt.subplots(1, 1, figsize=(6,6), subplot_kw={'projection':'smith', 'Z0': Z0})
    for i, s11 in enumerate(s11s):
        ax.plot(s11, domain=R_DOMAIN, label=labels[i], color=colors[i])
    if len(freqs) > 1:
        ax.legend()
    fig.tight_layout()
    set_mpl_params()

    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    for i, freq in enumerate(freqs):
        s11 = s11s[i]
        ax[0].plot(freq*1e-9, 20*np.log10(np.abs(s11)+1e-12), color=colors[i], label=labels[i], marker='.')
        ax[0].set_xlabel('Frequency [GHz]')
        ax[0].set_ylabel('|S11| [dB]')
        ax[1].plot(freq*1e-9, np.angle(s11)*360 / (2*np.pi), color=colors[i], label=labels[i], marker='.')
        ax[1].set_xlabel('Frequency [GHz]')
        ax[1].set_ylabel('∠S11 [degrees]')
    if len(freqs) > 1:
        ax[0].legend()
    fig.tight_layout()



# Geometry setup and display

negatives = []
class StripNegative:
    def __init__(self, diag_xi, diag_xf, diag_yi, diag_yf, subtract_left):
        # (xi,yi) is left point (smaller x)
        # (xf,yf) is right point (larger x)
        # Points must lie on the grid
        self.diag_xi = diag_xi
        self.diag_xf = diag_xf
        self.diag_yi = diag_yi
        self.diag_yf = diag_yf
        self.subtract_left = subtract_left
        self.subtract_below = (subtract_left and self.diag_yi > self.diag_yf) or (not subtract_left and self.diag_yi < self.diag_yf)
        negatives.append(self)
    def apply(self, msx_3d, msy_3d, z):
        # Each element in msx_3d, msy_3d describes the fraction of an edge contained in PEC

        if self.subtract_below == self.subtract_left:
            yi = self.diag_yf
            yf = self.diag_yi
        else:
            yi = self.diag_yi
            yf = self.diag_yf

        m = (self.diag_xf-self.diag_xi) / (self.diag_yf-self.diag_yi)

        for y in range(int(yi), int(yf)+1):
            if self.subtract_left and self.subtract_below:
                xi = self.diag_xi
                xf = self.diag_xf + m * (y - yi)
            elif self.subtract_left and not self.subtract_below:
                xi = self.diag_xi
                xf = self.diag_xi + m * (y - yi)
            elif not self.subtract_left and self.subtract_below:
                xf = self.diag_xf
                xi = self.diag_xi + m * (y - yi)
            elif not self.subtract_left and not self.subtract_below:
                xf = self.diag_xf
                xi = self.diag_xf + m * (y - yi)

            for x in range(int(xi), int(xf)+1):
                if xf < x + 1:
                    # End of triangle contained within the edge
                    frac = x+1 - xf
                elif xi > x:
                    # Start of triangle contained within the edge
                    frac = xi - x
                else:
                    frac = 0
                msx_3d[z, x, y] *= frac

        xi = self.diag_xi
        xf = self.diag_xf

        ymin = yi
        ymax = yf

        for x in range(int(xi), int(xf)+1):
            if self.subtract_left and self.subtract_below:
                yf = ymax + (x - xi) / m
            elif self.subtract_left and not self.subtract_below:
                yi = ymin + (x - xi) / m
            elif not self.subtract_left and self.subtract_below:
                yf = ymin + (x - xi) / m
            elif not self.subtract_left and not self.subtract_below:
                yi = ymax + (x - xi) / m

            for y in range(int(yi), int(yf)+1):
                if yf < y + 1:
                    # End of triangle contained within the edge
                    frac = y+1 - yf
                elif yi > y:
                    # Start of triangle contained within the edge
                    frac = yi - y
                else:
                    frac = 0
                msy_3d[z, x, y] *= frac

strips = []
def coord_in_rect(x, y, rect_xi, rect_xf, rect_yi, rect_yf):
    return x >= rect_xi and x < rect_xf and y >= rect_yi and y < rect_yf
class Strip:
    def __init__(self, xi, xf, yi, yf):
        self.xi = xi
        self.xf = xf
        self.yi = yi
        self.yf = yf
        strips.append(self)
    def coord_inside(self, x, y):
        return coord_in_rect(x, y, self.xi, self.xf, self.yi, self.yf)
    # Due to location of E in cells, x and y E-fields must be set differently on microstrip boundary
    def coord_inside_x(self, x, y):
        return coord_in_rect(x, y, self.xi, self.xf, self.yi, self.yf+1)
    def coord_inside_y(self, x, y):
        return coord_in_rect(x, y, self.xi, self.xf+1, self.yi, self.yf)



# Calculated parameters

INDUCTOR_L = int(np.round(INDUCTOR_L_m/dy))

PML_Dx = int(np.round(PML_D_m/dx))
PML_Dy = int(np.round(PML_D_m/dy))
PML_Dz = int(np.round(PML_D_m/dz))

VAC_Dx = int(np.round(VAC_D_m/dx))
VAC_Dy = int(np.round(VAC_D_m/dy))
VAC_Dz = int(np.round(VAC_D_m/dz))

BOARD_W = int(np.round(BOARD_W_m/dx))
BOARD_L = int(np.round(BOARD_L_m/dy))
BOARD_Xi = PML_Dx + VAC_Dx
BOARD_Xf = BOARD_Xi + BOARD_W
BOARD_Yi = PML_Dy + VAC_Dy
BOARD_Yf = BOARD_Yi + BOARD_L

INPUT_FEED_L = int(np.round(INPUT_FEED_L_m/dy))
INPUT_FEED_Xi = int(np.round(BOARD_Xi + PATCH_MARGIN_m/dx + PATCH_W_m/dx/2 - INPUT_FEED_W/2))
INPUT_FEED_Xf = INPUT_FEED_Xi + INPUT_FEED_W
INPUT_FEED_Yi = BOARD_Yi+1
INPUT_FEED_Yf = INPUT_FEED_Yi + INPUT_FEED_L
strip_input_feed = Strip(INPUT_FEED_Xi, INPUT_FEED_Xf, INPUT_FEED_Yi, INPUT_FEED_Yf)

FEED_L = int(np.round(FEED_L_m/dy))
FEED_Xi = int(np.round(BOARD_Xi + PATCH_MARGIN_m/dx + PATCH_W_m/dx/2 - FEED_W/2))
FEED_Xf = FEED_Xi + FEED_W
FEED_Yi = BOARD_Yi + INPUT_FEED_L + INDUCTOR_L + 1
FEED_Yf = FEED_Yi + FEED_L
strip_feed = Strip(FEED_Xi, FEED_Xf, FEED_Yi, FEED_Yf)

FEED_HI_L = int(np.round(FEED_HI_L_m/dy))
FEED_HI_W = int(np.round(FEED_W_m/dx))
FEED_HI_Xi = int(np.round(FEED_Xi + FEED_W/2 - FEED_HI_W/2))
FEED_HI_Xf = FEED_HI_Xi + FEED_HI_W
FEED_HI_Yi = FEED_Yf
FEED_HI_Yf = FEED_HI_Yi + FEED_HI_L
strip_feed_hi = Strip(FEED_HI_Xi, FEED_HI_Xf, FEED_HI_Yi, FEED_HI_Yf)

PATCH_W = int(np.round(PATCH_W_m/dx))
PATCH_L = int(np.round(PATCH_W_m/dy))
PATCH_Xi = int(np.round(BOARD_Xi + PATCH_MARGIN_m/dx))
PATCH_Xf = PATCH_Xi + PATCH_W
PATCH_Yi = FEED_HI_Yf
PATCH_Yf = PATCH_Yi + PATCH_L
strip_patch = Strip(PATCH_Xi, PATCH_Xf, PATCH_Yi, PATCH_Yf)

STUB_L = int(np.round(STUB_L_m/dx))
STUB_W = FEED_W
STUB_Xi = FEED_Xi + FEED_W
STUB_Xf = STUB_Xi + STUB_L
STUB_Yf = int(np.round(PATCH_Yi - FEED_HI_L + STUB_W/2))
STUB_Yi = STUB_Yf - STUB_W
stub_patch = Strip(STUB_Xi, STUB_Xf, STUB_Yi, STUB_Yf)

strip_feed_start = Strip(FEED_Xi, FEED_Xf, BOARD_Yi, BOARD_Yi+1)

CORNER_TRUNCATION_X = int(np.round(CORNER_TRUNCATION_m / dx))
CORNER_TRUNCATION_Y = int(np.round(CORNER_TRUNCATION_m / dy))
corner_negative_10 = StripNegative(
    PATCH_Xf-CORNER_TRUNCATION_X, PATCH_Xf,
    PATCH_Yi, PATCH_Yi+CORNER_TRUNCATION_Y,
    subtract_left=False
)
corner_negative_01 = StripNegative(
    PATCH_Xi, PATCH_Xi+CORNER_TRUNCATION_X,
    PATCH_Yf-CORNER_TRUNCATION_Y, PATCH_Yf,
    subtract_left=True
)

READ_X = FEED_Xi + FEED_W//2
READ_Y = BOARD_Yi + 1 + INPUT_FEED_L//2

# Source inductor
inductor_range_y = \
    list(range(BOARD_Yi+INPUT_FEED_L+1, BOARD_Yi+INPUT_FEED_L+INDUCTOR_L+1))

print(f'Key parameter listing:')
print(f'\tPATCH_W = {PATCH_W*dx*1e3} mm')
print(f'\tPATCH_L = {PATCH_L*dy*1e3} mm')
print(f'\tFEED_HI_L = {FEED_HI_L*dy*1e3} mm')
print(f'\tFEED_HI_W = {FEED_HI_W*dx*1e3} mm')
print(f'\tSTUB_L = {STUB_L*dy*1e3} mm')
print(f'\tFEED_W = STUB_W = {FEED_W*dx*1e3} mm')
print(f'\tFEED_L = {FEED_L*dy*1e3} mm')
print(f'\tCORNER_TRUNCATION = {CORNER_TRUNCATION_X*dx*1e3} mm')
print()
print('Press enter to begin simulation...')
input()



# Geometry display

GEOM_PEC  = 0
GEOM_VAC  = 1
GEOM_PML  = 2
GEOM_PCB  = 3
GEOM_RESISTOR = 4
GEOM_INDUCTOR = 5
GEOM_READ = 6
geom_dict = {
    GEOM_PEC:      {'color': 'black',         'name': 'PEC'},
    GEOM_VAC:      {'color': 'white',         'name': 'Vacuum'},
    GEOM_PML:      {'color': 'gray',          'name': 'PML'},
    GEOM_PCB:      {'color': 'wheat',         'name': 'PCB'},
    GEOM_RESISTOR: {'color': 'red',           'name': 'Resistor'},
    GEOM_INDUCTOR: {'color': 'green',         'name': 'Inductor'},
    GEOM_READ:     {'color': 'yellow',        'name': 'Readout'}
}
geom_colors_neg = [str(abs(i) / 275.0) for i in range(-255, 0)]
geom_colors_pos = [geom_type['color'] for geom_type in geom_dict.values()]
geom_colors = geom_colors_neg + geom_colors_pos
cmap_geom = mpl.colors.ListedColormap(geom_colors)

geom = np.ones((NY+1, NX+1), dtype=int)

# Add PEC/PML boundary geometry to geom array
for x in range(NX+1):
    for y in range(NY+1):
        if x == 0 or y == 0 or x == NX or y == NY:
            geom[y,x] = GEOM_PEC
        else:
            if x < PML_Dx or y < PML_Dy or x >= NX-PML_Dx or y >= NY-PML_Dy:
                geom[y,x] = GEOM_PML

# Set up epsilon geometry (PCB)
eps = np.full((NZ+1, NX+1, NY+1), eps0)
sig = np.zeros((NZ+1, NX+1, NY+1))
for x in range(BOARD_Xi, BOARD_Xf):
    for y in range(BOARD_Yi, BOARD_Yf):
        for z in range(BOARD_H):
            eps[z,x,y] = eps_pcb
            sig[z,x,y] = sig_pcb
        geom[y,x] = GEOM_PCB

# Tangential E-field is zero on microstrip conductor surface
# 1s along matrix diagonal indicate conductor surface on bottom of that cell
msx_3d = np.zeros((NZ+1, NX+1, NY+1))
msy_3d = np.zeros((NZ+1, NX+1, NY+1))
for x in range(NX+1):
    for y in range(NY+1):
        for strip in strips:
            if strip.coord_inside_x(x, y):
                msx_3d[BOARD_H, x, y] = 1
            if strip.coord_inside_y(x, y):
                msy_3d[BOARD_H, x, y] = 1
            #if strip.coord_inside(x, y):
            #    geom[y,x] = GEOM_PEC
for strip_negative in negatives:
    strip_negative.apply(msx_3d, msy_3d, BOARD_H)
for x in range(NX+1):
    for y in range(NY+1):
        msx = msx_3d[BOARD_H, x, y]
        msy = msy_3d[BOARD_H, x, y]
        pec_frac = np.sqrt(msx**2 / 2 + msy**2 / 2)
        if pec_frac > 0:
            geom[y,x] = int(-255 * (1 - pec_frac))

msx_mat = sp.sparse.diags_array(
    np.reshape(msx_3d, SS)
)
msy_mat = sp.sparse.diags_array(
    np.reshape(msy_3d, SS)
)

# Add lumped elements to geom array
for x in range(FEED_Xi,FEED_Xf):
    geom[BOARD_Yi,x] = GEOM_RESISTOR
    if LOAD_ENABLE:
        geom[BOARD_Yf,x] = GEOM_RESISTOR
    for y in inductor_range_y:
        geom[y,x] = GEOM_INDUCTOR

# Readout position
geom[READ_Y,READ_X] = GEOM_READ

fig_geom, ax_geom = plt.subplots(1, 1, figsize=(7, 6))
im_geom = ax_geom.imshow(
    geom,
    cmap=cmap_geom,
    vmin=-255.5, vmax=len(geom_dict)-0.5,
    origin='lower',
    extent=[0, (NX+1)*dx*1e2, 0, (NY+1)*dy*1e2],
    aspect='equal'
)
ax_geom.set_xlabel('X [cm]')
ax_geom.set_ylabel('Y [cm]')
ax_geom.set_xticks(np.arange(0, int((NX+1)*dx*1e2), 1))
ax_geom.set_yticks(np.arange(0, int((NY+1)*dy*1e2), 1))
ax_geom.tick_params(which='minor', bottom=False, left=False)
for i in range(NX+1):
    ax_geom.axvline(i*dx*1e2, color='black', linewidth=0.5)
for i in range(NY+1):
    ax_geom.axhline(i*dy*1e2, color='black', linewidth=0.5)
cmap_geom_cbar = mpl.colors.ListedColormap(geom_colors_pos)
norm_geom_cbar = mpl.colors.Normalize(vmin=-0.5, vmax=len(geom_dict)-0.5)
cbar_geom_sm = mpl.cm.ScalarMappable(cmap=cmap_geom_cbar, norm=norm_geom_cbar)
cbar_geom_sm.set_array([])
cbar_geom = plt.colorbar(cbar_geom_sm, ax=ax_geom, ticks=range(len(geom_dict)))
cbar_geom.set_ticklabels([geom_type['name'] for geom_type in geom_dict.values()])
fig_geom.tight_layout()



# Initialize CPML-CFS constants

# Parameters were initialized with those used by Roden and Gedney, 2000
# Then parameters were optimized to minimize reflection in this design
PML_M = 4  # PML parameter scaling order
PML_SIG_MAX = 1.5 * (PML_M + 1) / (150 * np.pi)  # Maximum PML conductivity (divide by dx, dy, or dz)
PML_KAP_MAX = 2.0
PML_ALPHA = 0.05

# Start with 3D arrays, then reshape to one SS-length column
# Each array contains two slabs of non-zero values
pml_px = np.zeros((NZ+1, NX+1, NY+1))
pml_py = np.zeros((NZ+1, NX+1, NY+1))
pml_pz = np.zeros((NZ+1, NX+1, NY+1))
pml_qx = np.zeros((NZ+1, NX+1, NY+1))
pml_qy = np.zeros((NZ+1, NX+1, NY+1))
pml_qz = np.zeros((NZ+1, NX+1, NY+1))
pml_kx = np.ones((NZ+1, NX+1, NY+1))
pml_ky = np.ones((NZ+1, NX+1, NY+1))
pml_kz = np.ones((NZ+1, NX+1, NY+1))

for dim, PML_D in enumerate([PML_Dx, PML_Dy, PML_Dz]):
    for u in range(PML_D):
        depth_frac = ((u+1)**PML_M) / (PML_D**PML_M)
        kap_u = 1 + depth_frac * (PML_KAP_MAX - 1)
        if dim == 0:
            sig_x = depth_frac * PML_SIG_MAX / dx
            px = np.exp(-(sig_x/kap_u + PML_ALPHA) * dt/eps0)
            qx = sig_x / (kap_u * (sig_x + kap_u*PML_ALPHA)) * (px - 1)
            pml_px[:,(PML_D-1)-u,:] = px
            pml_px[:,NX-(PML_D-1)+u,:] = px
            pml_qx[:,(PML_D-1)-u,:] = qx
            pml_qx[:,NX-(PML_D-1)+u,:] = qx
            pml_kx[:,(PML_D-1)-u,:] = 1/kap_u
            pml_kx[:,NX-(PML_D-1)+u,:] = 1/kap_u
        elif dim == 1:
            sig_y = depth_frac * PML_SIG_MAX / dy
            py = np.exp(-(sig_y/kap_u + PML_ALPHA) * dt/eps0)
            qy = sig_y / (kap_u * (sig_y + kap_u*PML_ALPHA)) * (py - 1)
            pml_py[:,:,(PML_D-1)-u] = py
            pml_py[:,:,NY-(PML_D-1)+u] = py
            pml_qy[:,:,(PML_D-1)-u] = qy
            pml_qy[:,:,NY-(PML_D-1)+u] = qy
            pml_ky[:,:,(PML_D-1)-u] = 1/kap_u
            pml_ky[:,:,NY-(PML_D-1)+u] = 1/kap_u
        elif dim == 2:
            # PML at positive z boundary
            # PEC boundary conditions at z=0 (infinite ground plane)
            sig_z = depth_frac * PML_SIG_MAX / dz
            pz = np.exp(-(sig_z/kap_u + PML_ALPHA) * dt/eps0)
            qz = sig_z / (kap_u * (sig_z + kap_u*PML_ALPHA)) * (pz - 1)
            #pml_pz[(PML_D-1)-u,:,:] = pz
            pml_pz[NZ-(PML_D-1)+u,:,:] = pz
            #pml_qz[(PML_D-1)-u,:,:] = qz
            pml_qz[NZ-(PML_D-1)+u,:,:] = qz
            #pml_kz[(PML_D-1)-u,:,:] = 1/kap_u
            pml_kz[NZ-(PML_D-1)+u,:,:] = 1/kap_u

pml_px = np.reshape(pml_px, SS)
pml_py = np.reshape(pml_py, SS)
pml_pz = np.reshape(pml_pz, SS)
pml_qx = np.reshape(pml_qx, SS)
pml_qy = np.reshape(pml_qy, SS)
pml_qz = np.reshape(pml_qz, SS)
pml_kx = np.reshape(pml_kx, SS)
pml_ky = np.reshape(pml_ky, SS)
pml_kz = np.reshape(pml_kz, SS)



# PML auxiliary variables
pml_psi_exy = np.zeros(SS)
pml_psi_eyx = np.zeros(SS)
pml_psi_exz = np.zeros(SS)
pml_psi_ezx = np.zeros(SS)
pml_psi_eyz = np.zeros(SS)
pml_psi_ezy = np.zeros(SS)
pml_psi_hxy = np.zeros(SS)
pml_psi_hyx = np.zeros(SS)
pml_psi_hxz = np.zeros(SS)
pml_psi_hzx = np.zeros(SS)
pml_psi_hyz = np.zeros(SS)
pml_psi_hzy = np.zeros(SS)

Ex = np.zeros(SS)
Ey = np.zeros(SS)
Ez = np.zeros(SS)
Hx = np.zeros(SS)
Hy = np.zeros(SS)
Hz = np.zeros(SS)

# Accumulate Ey for lumped inductors
Ey_prev = np.zeros(SS)
Ey_accum = np.zeros(SS)



# Initialize matrices for computing spatial derivatives
dEdx_mat = 1/dx * sp.sparse.kron(
    sp.sparse.eye_array(NZ+1),
    sp.sparse.kron(
        sp.sparse.diags_array([-1.0, 1.0], offsets=[0, 1], shape=(NX+1,NX+1)),
        sp.sparse.eye_array(NY+1)
    )
)
dEdy_mat = 1/dy * sp.sparse.kron(
    sp.sparse.eye_array(NZ+1),
    sp.sparse.kron(
        sp.sparse.eye_array(NX+1),
        sp.sparse.diags_array([-1.0, 1.0], offsets=[0, 1], shape=(NY+1,NY+1))
    )
)
dEdz_mat = 1/dz * sp.sparse.kron(
    sp.sparse.diags_array([-1.0, 1.0], offsets=[0, 1], shape=(NZ+1,NZ+1)),
    sp.sparse.kron(
        sp.sparse.eye_array(NX+1),
        sp.sparse.eye_array(NY+1)
    )
)
dHdx_mat = 1/dx * sp.sparse.kron(
    sp.sparse.eye_array(NZ+1),
    sp.sparse.kron(
        sp.sparse.diags_array([-1.0, 1.0], offsets=[-1, 0], shape=(NX+1,NX+1)),
        sp.sparse.eye_array(NY+1)
    )
)
dHdy_mat = 1/dy * sp.sparse.kron(
    sp.sparse.eye_array(NZ+1),
    sp.sparse.kron(
        sp.sparse.eye_array(NX+1),
        sp.sparse.diags_array([-1.0, 1.0], offsets=[-1, 0], shape=(NY+1,NY+1))
    )
)
dHdz_mat = 1/dz * sp.sparse.kron(
    sp.sparse.diags_array([-1.0, 1.0], offsets=[-1, 0], shape=(NZ+1,NZ+1)),
    sp.sparse.kron(
        sp.sparse.eye_array(NX+1),
        sp.sparse.eye_array(NY+1)
    )
)



# Ey=0 and Ez=0 at x-axis boundaries
x_boundary_eye = np.eye(NX+1)
x_boundary_eye[0] = 0
x_boundary_eye[NX] = 0
x_boundary_mat = sp.sparse.kron(
    sp.sparse.eye_array(NZ+1),
    sp.sparse.kron(
        x_boundary_eye,
        sp.sparse.eye_array(NY+1)
    )
)

# Ex=0 and Ez=0 at y-axis boundaries
y_boundary_eye = np.eye(NY+1)
y_boundary_eye[0] = 0
y_boundary_eye[NY] = 0
y_boundary_mat = sp.sparse.kron(
    sp.sparse.eye_array(NZ+1),
    sp.sparse.kron(
        sp.sparse.eye_array(NX+1),
        y_boundary_eye
    )
)

# Ex=0 and Ey=0 at z-axis boundaries
z_boundary_eye = np.eye(NZ+1)
z_boundary_eye[0] = 0
z_boundary_eye[NZ] = 0
z_boundary_mat = sp.sparse.kron(
    z_boundary_eye,
    sp.sparse.kron(
        sp.sparse.eye_array(NX+1),
        sp.sparse.eye_array(NY+1)
    )
)



# Set up coefficients for FDTD update
E_C1 = (2*eps - dt*sig) / (2*eps + dt*sig)
E_C2 = 2*dt / (2*eps + dt*sig)

# Modify coefficients for Ez calculation with lumped resistors
C_lump = dz*dt / (2*eps_pcb*dx*dy)
Rl = ZL * (FEED_Xf+1-FEED_Xi) / BOARD_H  # Load resistance
Cl = C_lump / Rl
Rs = Z0 * (FEED_Xf+1-FEED_Xi) / BOARD_H  # Source resistance
Cs = C_lump / Rs
Ez_C1 = np.copy(E_C1)
Ez_C2 = np.copy(E_C2)
Ez_C3 = np.zeros((NZ+1, NX+1, NY+1))

# Modify coefficients for Ey calculation with lumped inductors
if INDUCTOR_L == 0:
    inductance = 0
else:
    inductance = SERIES_INDUCTANCE * (FEED_Xf+1-FEED_Xi) / INDUCTOR_L
Ey_L_C1 = (-dy*dt**2 + 2*dx*dz*eps_pcb*inductance) / (dy*dt**2 + 2*dx*dz*eps_pcb*inductance)
Ey_L_C2 = (2*dt*dx*dz*inductance) / (dy*dt**2 + 2*dx*dz*eps_pcb*inductance)
Ey_L_C3 = (-2*dy*dt**2) / (dy*dt**2 + 2*dx*dz*eps_pcb*inductance)
Ey_C1 = np.copy(E_C1)
Ey_C2 = np.copy(E_C2)
Ey_C3 = np.zeros((NZ+1, NX+1, NY+1))

for z in range(BOARD_H):
    for x in range(FEED_Xi, FEED_Xf+1):
        # Resistive voltage source
        Ez_C1[z,x,BOARD_Yi] = (1 - Cs) / (1 + Cs)
        Ez_C2[z,x,BOARD_Yi] = (dt/eps_pcb) / (1 + Cs)
        Ez_C3[z,x,BOARD_Yi] = (2*Cs/dz) / (1 + Cs) / BOARD_H

        # Load resistor
        if LOAD_ENABLE:
            Ez_C1[z,x,BOARD_Yf] = (1 - Cl) / (1 + Cl)
            Ez_C2[z,x,BOARD_Yf] = (dt/eps_pcb) / (1 + Cl)

for x in range(FEED_Xi, FEED_Xf+1):
    # Source series inductance
    for y in inductor_range_y:
        Ey_C1[BOARD_H,x,y] = Ey_L_C1
        Ey_C2[BOARD_H,x,y] = Ey_L_C2
        Ey_C3[BOARD_H,x,y] = Ey_L_C3

E_C1 = np.reshape(E_C1, SS)
E_C2 = np.reshape(E_C2, SS)
Ez_C1 = np.reshape(Ez_C1, SS)
Ez_C2 = np.reshape(Ez_C2, SS)
Ez_C3 = np.reshape(Ez_C3, SS)
Ey_C1 = np.reshape(Ey_C1, SS)
Ey_C2 = np.reshape(Ey_C2, SS)
Ey_C3 = np.reshape(Ey_C3, SS)



t = np.array(range(NT)) * dt

# Resistive voltage source (n+1/2 timesteps)
def Vs_func(t):
    return 0.5e-3 * (t-t0_src)/tau_src * np.exp(-0.5 * ((t-t0_src)/tau_src)**2)

# Readout voltage and current (n+1/2 timesteps)
Vt = np.zeros(NT)
It = np.zeros(NT)

Vs = Vs_func(t)
anim_frames = np.zeros((int(np.ceil(NT/ANIM_RATIO)), NX+1, NY+1))

for n in tqdm(range(NT)):
    pml_psi_exy = pml_py*pml_psi_exy + pml_qy * (dEdy_mat @ Ex)
    pml_psi_eyx = pml_px*pml_psi_eyx + pml_qx * (dEdx_mat @ Ey)
    pml_psi_exz = pml_pz*pml_psi_exz + pml_qz * (dEdz_mat @ Ex)
    pml_psi_ezx = pml_px*pml_psi_ezx + pml_qx * (dEdx_mat @ Ez)
    pml_psi_eyz = pml_pz*pml_psi_eyz + pml_qz * (dEdz_mat @ Ey)
    pml_psi_ezy = pml_py*pml_psi_ezy + pml_qy * (dEdy_mat @ Ez)

    Hx += dt/mu0 * (pml_kz*(dEdz_mat @ Ey) - pml_ky*(dEdy_mat @ Ez) + pml_psi_eyz - pml_psi_ezy)
    Hy += dt/mu0 * (pml_kx*(dEdx_mat @ Ez) - pml_kz*(dEdz_mat @ Ex) + pml_psi_ezx - pml_psi_exz)
    Hz += dt/mu0 * (pml_ky*(dEdy_mat @ Ex) - pml_kx*(dEdx_mat @ Ey) + pml_psi_exy - pml_psi_eyx)

    pml_psi_hxy = pml_py*pml_psi_hxy + pml_qy * (dHdy_mat @ Hx)
    pml_psi_hyx = pml_px*pml_psi_hyx + pml_qx * (dHdx_mat @ Hy)
    pml_psi_hxz = pml_pz*pml_psi_hxz + pml_qz * (dHdz_mat @ Hx)
    pml_psi_hzx = pml_px*pml_psi_hzx + pml_qx * (dHdx_mat @ Hz)
    pml_psi_hyz = pml_pz*pml_psi_hyz + pml_qz * (dHdz_mat @ Hy)
    pml_psi_hzy = pml_py*pml_psi_hzy + pml_qy * (dHdy_mat @ Hz)

    Ey_prev = Ey[:]

    Ex = \
        E_C1 * Ex + \
        E_C2 * (pml_ky*(dHdy_mat @ Hz) - pml_kz*(dHdz_mat @ Hy) + pml_psi_hzy - pml_psi_hyz)
    Ey = \
        Ey_C1 * Ey + \
        Ey_C2 * (pml_kz*(dHdz_mat @ Hx) - pml_kx*(dHdx_mat @ Hz) + pml_psi_hxz - pml_psi_hzx) + \
        Ey_C3 * Ey_accum
    Ez = \
        Ez_C1 * Ez + \
        Ez_C2 * (pml_kx*(dHdx_mat @ Hy) - pml_ky*(dHdy_mat @ Hx) + pml_psi_hyx - pml_psi_hxy) + \
        Ez_C3 * Vs[n]

    # Apply PEC boundary on far perimeter of simulation space
    Ex = z_boundary_mat @ (y_boundary_mat @ Ex)
    Ey = z_boundary_mat @ (x_boundary_mat @ Ey)
    Ez = y_boundary_mat @ (x_boundary_mat @ Ez)

    # Apply PEC boundary on microstrip conductor surface
    Ex -= msx_mat @ Ex
    Ey -= msy_mat @ Ey

    # Accumulate Ey for lumped inductors
    Ey_accum += 0.5 * (Ey_prev + Ey)

    Ez_3d = np.reshape(Ez, (NZ+1, NX+1, NY+1), copy=True)
    if n % ANIM_RATIO == 0:
        anim_frames[n//ANIM_RATIO] = Ez_3d[BOARD_H-1]

    # Integrate Ez along z to get readout voltage
    # E timestep: n+1
    for z in range(BOARD_H):
        index = READ_Y + (NY+1)*READ_X + (NY+1)*(NX+1)*z
        Vt[n] -= Ez[index]*dz  # Positive voltage corresponds to -Ez

    # Integrate H around top microstrip conductor to get readout current
    # H timestep: n+1/2
    # Note: H is also offset from E by 1/2 cell dimension
    for x in range(FEED_Xi, FEED_Xf+1):
        index_H_top = READ_Y + (NY+1)*x + (NY+1)*(NX+1)*BOARD_H
        index_H_bot = READ_Y + (NY+1)*x + (NY+1)*(NX+1)*(BOARD_H-1)
        # Interpolation is between neighboring y positions
        It[n] -= 0.5 * (Hx[index_H_bot] + Hx[index_H_bot-1]) * dx
        It[n] += 0.5 * (Hx[index_H_top] + Hx[index_H_top-1]) * dx
    index_H_l = READ_Y + (NY+1)*(FEED_Xi-1) + (NY+1)*(NX+1)*BOARD_H
    index_H_r = READ_Y + (NY+1)*FEED_Xf + (NY+1)*(NX+1)*BOARD_H
    It[n] += 0.5 * (Hz[index_H_l] + Hz[index_H_l-1]) * dz
    It[n] -= 0.5 * (Hz[index_H_r] + Hz[index_H_r-1]) * dz

# Shift Vt values to n+1/2 timesteps (backwards average)
Vt = 0.5 * (np.concatenate(([0], Vt[:-1])) + Vt)



anim1 = get_animation(anim_frames)

fig, ax = plt.subplots(1, 1, figsize=(7,5))
ax.set_title('Time-Domain Measurement')
ax.set_xlabel('Time [ns]')
ax.plot(t*1e9, Vt*1e3, color='blue')
ax.set_ylabel('Voltage [mV]')
ax.set_ylim([-np.max(np.abs(Vt*1e3))*1.05, np.max(np.abs(Vt*1e3))*1.05])
ax_I = ax.twinx()
ax_I.plot(t*1e9, It*1e3, color='red')
ax_I.set_ylabel('Current [mA]')
ax_I.set_ylim([-np.max(np.abs(It*1e3))*1.05, np.max(np.abs(It*1e3))*1.05])
fig.tight_layout()

freq_max = 1/dt
freq_max_disp = 3e9
freq_max_index = int(NT * freq_max_disp/freq_max)
freq_min_disp = 2e9
freq_min_index = int(NT * freq_min_disp/freq_max)
assert freq_max_disp < freq_max/2
freqs = np.array(range(freq_min_index, freq_max_index)) / NT * 1/dt
Vf = np.fft.fft(Vt)
Vf = Vf[freq_min_index:freq_max_index]
If = np.fft.fft(It)
If = If[freq_min_index:freq_max_index]
Zf = Vf/If

# Determine ZL from Zf and Z0 (assumes Z0 = Rs = 50 ohms)
D = (FEED_Yi-INDUCTOR_L-READ_Y)*dy  # Reference plane is just before source inductor
k = 2*np.pi*freqs / v_pcb
e2jkz = np.exp(2j*k*D)
ZL_measured = Z0 * ((1+e2jkz)*Zf + (-1+e2jkz)*Z0) / ((-1+e2jkz)*Zf + (1+e2jkz)*Z0)
S11 = (ZL_measured - Z0) / (ZL_measured + Z0)

network_fdtd = rf.Network(f=freqs, s=S11)
network_fdtd.write_touchstone('fdtd_truncated_corners')

plot_s11(
    [freqs],
    [S11],
    ['FDTD'],
    ['black']
)

plt.show()

