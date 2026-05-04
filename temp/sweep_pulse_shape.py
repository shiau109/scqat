#%% Sweep pulse_shape and plot photon number for each

import matplotlib.pyplot as plt
import numpy as np
from Readout_pulse_shaping import Readout_cal_amp_phase_correction

para_base = {
    'ki': 0, 'ke': 0.0004,
    'fc': 3.078602573, 'frf': 3.078602573,
    'X_eff': 0.000075,
    'pulse_shape': 'Three_step',
    'phase_correction': False,
    'P_in': -115,
    'time': 6000,
    'ti': 0,
    'Du': 4000,
    'first_step_Du': 100,
    'third_step_Du': 100,
}

sweep_shapes = ["Square", "Two_step", "Three_step"]
colors = ['blue', 'red', 'green']
alphas = [1, 1, 1]
linestyles = ['-', '--', ':']
fig, axes = plt.subplots(nrows=3, figsize=(6, 4), dpi=200, sharex=True)
fig2, ax3 = plt.subplots(figsize=(6, 2), dpi=200)

for i, shape in enumerate(sweep_shapes):
    print(f"Simulating pulse_shape = {shape}...")
    para = para_base.copy()
    para['pulse_shape'] = shape
    L = Readout_cal_amp_phase_correction(para)
    t_us = L['t'] * 1e9 / 1000  # convert to µs
    dt = L['t'][1] - L['t'][0]
    dn_g_dt = np.gradient(L['n_g'], dt)
    dn_e_dt = np.gradient(L['n_e'], dt)

    # Prepend (0, 0) to all data lines
    t_us_0 = np.insert(t_us, 0, 0)
    pulse_g_0 = np.insert(np.real(L['pulse_timeamp_g']), 0, 0)
    n_g_0 = np.insert(L['n_g'], 0, 0)
    n_e_0 = np.insert(L['n_e'], 0, 0)
    dn_g_0 = np.insert(np.abs(dn_g_dt / 1e9), 0, 0)
    dn_e_0 = np.insert(np.abs(dn_e_dt / 1e9), 0, 0)

    axes[0].plot(t_us_0, pulse_g_0, alpha=alphas[i], lw=2, linestyle=linestyles[i], color=colors[i], label=shape)
    line_n, = axes[1].plot(t_us_0, n_g_0, alpha=alphas[i], lw=2, linestyle=linestyles[i], color=colors[i], label=f'{shape} (g)')
    # axes[1].plot(t_us_0, n_e_0, alpha=0.7, lw=1.5, linestyle='--', color=line_n.get_color(), label=f'{shape} (e)')
    line_g3, = axes[2].plot(t_us_0, dn_g_0, alpha=alphas[i], lw=2, linestyle=linestyles[i], color=colors[i], label=f'{shape} (g)')
    # axes[2].plot(t_us_0, dn_e_0, alpha=0.7, lw=1.5, linestyle='--', color=line_g3.get_color(), label=f'{shape} (e)')
    line_g4, = ax3.plot(n_g_0, dn_g_0, alpha=alphas[i], lw=2, linestyle=linestyles[i], color=colors[i], label=f'{shape}')
    # ax3.plot(n_e_0, dn_e_0, alpha=0.7, lw=1.5, linestyle='--', color=line_g4.get_color())#, label=f'{shape} (e)')

axes[0].axhline(y=0, color='k', linestyle='dashed', lw=1)
axes[0].axvline(x=0, color='grey', linestyle='dashed', lw=0.8)
axes[0].axvline(x=4, color='grey', linestyle='dashed', lw=0.8)
axes[0].set_ylabel(r"$V_{RO}(t)$", size=13)
# axes[0].set_title("Driving Amplitude")
axes[0].set_xlim(-0.5, max(t_us))
axes[0].legend(fontsize=9, title="pulse shape", loc="upper right")

axes[1].axhline(y=0, color='k', linestyle='dashed', lw=1)
axes[1].axvline(x=0, color='grey', linestyle='dashed', lw=0.8)
axes[1].axvline(x=4, color='grey', linestyle='dashed', lw=0.8)
axes[1].set_ylabel(r"$\bar{n}_r$", size=13)
# axes[1].set_title(r"Photon number")
# axes[1].legend(fontsize=7, title="pulse_shape", ncol=2)

axes[2].set_xlabel(r'$Time \ (\mu$s)', size=13)
axes[2].set_ylabel(r'$\Gamma_{r}\ (\bar{n}_r/\mathrm{ns})$', size=13)
axes[2].axhline(y=0, color='k', linestyle='dashed', lw=1)
# axes[2].set_title("Photon Number Changing Rate")

# axes[2].legend(fontsize=7, title="pulse_shape", ncol=2)
axes[2].set_yscale('log')
axes[2].set_ylim(1e-1, 100)

ax3.set_xlabel(r'$\bar{n}_r$', size=13)
ax3.set_ylabel(r'$|\Gamma_{r}|\ (\bar{n}_r/\mathrm{ns})$', size=13)
ax3.set_yscale('log')
ax3.set_ylim(1e-1, 100)
ax3.legend(fontsize=7, title="pulse shape", ncol=2)

fig.tight_layout()
fig2.tight_layout()
fig.savefig("sweep_pulse_shape.png", dpi=200, bbox_inches="tight")
fig2.savefig("sweep_pulse_shape_n_vs_rate.png", dpi=200, bbox_inches="tight")
plt.show()

# %%
