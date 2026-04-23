#%% Sweep first_step_Du (=third_step_Du) and plot photon number for each

import matplotlib.pyplot as plt
import numpy as np
from Readout_pulse_shaping import Readout_cal_amp_phase_correction

para_base = {
    'ki': 0, 'ke': 0.0004,
    'fc': 3.078602573, 'frf': 3.078602573,
    'X_eff': 0.000075,
    'pulse_shape': 'Three_step',
    'phase_correction': False,
    'P_in': -120,
    'time': 6000,
    'ti': 0,
    'Du': 4000,
    'first_step_Du': 100,
    'third_step_Du': 100,
}

# sweep_vals = [ 100, 120, 160, 200, 240, 300, 340, 400, 500, 600, 800, 1000, 1200]
sweep_vals = [ 100,  200, 400,  1000]

fig, axes = plt.subplots(nrows=3, figsize=(8, 9), dpi=200)
fig2, ax3 = plt.subplots(figsize=(8, 4), dpi=200)

# Square pulse reference
para_sq = para_base.copy()
para_sq['pulse_shape'] = 'Square'
L_sq = Readout_cal_amp_phase_correction(para_sq)
t_us_sq = L_sq['t'] * 1e9 / 1000
dt_sq = L_sq['t'][1] - L_sq['t'][0]
dn_g_dt_sq = np.gradient(L_sq['n_g'], dt_sq)
dn_e_dt_sq = np.gradient(L_sq['n_e'], dt_sq)

axes[0].plot(t_us_sq, np.real(L_sq['pulse_timeamp_g']), alpha=0.7, lw=1.5, color='k', label='Square')
line_sq, = axes[1].plot(t_us_sq, L_sq['n_g'], alpha=0.7, lw=1.5, color='k', label='Square (g)')
axes[1].plot(t_us_sq, L_sq['n_e'], alpha=0.7, lw=1.5, linestyle='--', color='k', label='Square (e)')
line_sq2, = axes[2].plot(t_us_sq, np.abs(dn_g_dt_sq / 1e9), alpha=0.7, lw=1.5, color='k', label='Square (g)')
axes[2].plot(t_us_sq, np.abs(dn_e_dt_sq / 1e9), alpha=0.7, lw=1.5, linestyle='--', color='k', label='Square (e)')
line_sq3, = ax3.plot(L_sq['n_g'], np.abs(dn_g_dt_sq / 1e9), alpha=0.7, lw=1.5, color='k', label='Square (g)')
ax3.plot(L_sq['n_e'], np.abs(dn_e_dt_sq / 1e9), alpha=0.7, lw=1.5, linestyle='--', color='k', label='Square (e)')

for du in sweep_vals:
    print(f"Simulating for step Du = {du} ns...")
    para = para_base.copy()
    para['first_step_Du'] = du
    para['third_step_Du'] = du
    L = Readout_cal_amp_phase_correction(para)
    t_us = L['t'] * 1e9 / 1000  # convert to µs
    dt = L['t'][1] - L['t'][0]
    dn_g_dt = np.gradient(L['n_g'], dt)
    dn_e_dt = np.gradient(L['n_e'], dt)

    axes[0].plot(t_us, np.real(L['pulse_timeamp_g']), alpha=0.7, lw=1.5, label=f'{du} ns')
    line_n, = axes[1].plot(t_us, L['n_g'], alpha=0.7, lw=1.5, label=f'{du} ns (g)')
    axes[1].plot(t_us, L['n_e'], alpha=0.7, lw=1.5, linestyle='--', color=line_n.get_color(), label=f'{du} ns (e)')
    line_g3, = axes[2].plot(t_us,  np.abs(dn_g_dt / 1e9), alpha=0.7, lw=1.5, label=f'{du} ns (g)')
    axes[2].plot(t_us,  np.abs(dn_e_dt / 1e9), alpha=0.7, lw=1.5, linestyle='--', color=line_g3.get_color(), label=f'{du} ns (e)')
    line_g4, = ax3.plot(L['n_g'], np.abs(dn_g_dt / 1e9), alpha=0.7, lw=1.5, label=f'{du} ns (g)')
    ax3.plot(L['n_e'], np.abs(dn_e_dt / 1e9), alpha=0.7, lw=1.5, linestyle='--', color=line_g4.get_color(), label=f'{du} ns (e)')

axes[0].set_ylabel(r"$V_g(t)$", size=12)
axes[0].set_title("Pulse shape (I)")
axes[0].set_xlim(0, max(t_us))
axes[0].legend(fontsize=9, title="step Du")

axes[1].set_ylabel(r"$\bar{n}$", size=13)
axes[1].set_title(r"Photon number")
axes[1].legend(fontsize=7, title="step Du", ncol=2)

axes[2].set_xlabel(r'$t\ (\mu$s)', size=13)
axes[2].set_ylabel(r'$\Gamma_{r}\ (n/\mathrm{ns})$', size=12)
axes[2].axhline(y=0, color='k', linestyle='dashed', lw=1)
axes[2].legend(fontsize=7, title="step Du", ncol=2)
axes[2].set_yscale('log')
axes[2].set_ylim(1e-1, 10)

ax3.set_xlabel(r'$n$', size=13)
ax3.set_ylabel(r'$|\Gamma_{r}|\ (n/\mathrm{ns})$', size=12)
ax3.set_yscale('log')
ax3.set_ylim(1e-1, 10)
ax3.legend(fontsize=7, title="step Du", ncol=2)

fig.tight_layout()
fig2.tight_layout()
plt.show()

# %%
