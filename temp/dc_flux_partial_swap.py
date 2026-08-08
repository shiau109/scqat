"""
qc_n_swap_amp simulation: repeat a DC-flux swap pulse N times on a
control-target transmon pair and sweep the control's flat-top detuning
(the ctrl_amp knob), recording the target population P(|01>) from |10>.

Reproduces the qc_n_swap_amp error-amplification map (N x flux-amplitude
chevron): each pulse partially swaps |10> <-> |01> by an angle set by the
on-flat-top detuning; repeating N times amplifies the swap, so the swap
resonance and the N-period (= pi/theta) show up as a chevron.

Frame convention
----------------
All simulation is done in the frame rotating at omega_2 for BOTH transmons
("common frame").  In this frame the exchange term g(a1^dag a2 + h.c.) is
static -- no explicit time-dependent phase -- which is exactly the statement
that a DC flux pulse carries no phase reference.

    H(t) = Delta(t) n1 + (alpha1/2) n1(n1-1) + (alpha2/2) n2(n2-1)
           + g (a1^dag a2 + a2^dag a1)
    Delta(t) = omega_1(t) - omega_2          (control detuning from target)

Populations are frame-independent, so nothing below depends on this choice.

Pulse shape
-----------
Quarter-sine flat-top (the lab's FlatTopCosinePulse): rise sin(0 -> pi/2), a
flat top, and fall sin(pi/2 -> pi).  Between consecutive swaps the control
returns to its park (dispersive) frequency for T_DELAY ns.

Everything is a GIVEN parameter -- nothing is calibrated to a target angle.
"""

import numpy as np
import matplotlib.pyplot as plt

TWOPI = 2 * np.pi

# ---------------------------------------------------------------- operators
NL, DIM = 3, 9                                   # qutrit x qutrit
a  = np.diag(np.sqrt(np.arange(1, NL)), 1)
I3, Id = np.eye(NL), np.eye(DIM)
a1, a2 = np.kron(a, I3), np.kron(I3, a)
n1, n2 = a1.conj().T @ a1, a2.conj().T @ a2

ix = lambda m, n: m * NL + n
S00, S01, S10, S11 = ix(0, 0), ix(0, 1), ix(1, 0), ix(1, 1)

# ======================================================== GIVEN PARAMETERS
# --- system ---------------------------------------------------------------
F2       = 5.000      # fixed-frequency transmon (target)            [GHz]
F1_IDLE  = 5.150      # tunable transmon (control), parked           [GHz]
A1, A2   = -0.200, -0.205                       # anharmonicities    [GHz]
G        = 0.005      # effective control-target exchange (J_eff)    [GHz]

# --- swap pulse -----------------------------------------------------------
EDGE     = 1.0        # quarter-sine edge width, per edge            [ns]
T_HOLD   = 25.0       # flat-top hold duration                       [ns]

# --- sequence -------------------------------------------------------------
T_DELAY  = 0.0        # delay at park between consecutive swaps       [ns]
NMAX     = 16         # max number of swap repetitions

# --- detuning sweep (the ctrl_amp axis) -----------------------------------
DMIN, DMAX = -10.0, 10.0        # control flat-top detuning sweep    [MHz]
NDET       = 161                # sweep points

DT       = 0.01       # integration step                             [ns]
# ==========================================================================

D_IDLE = TWOPI * (F1_IDLE - F2)

# NOTE on parking choice: |Delta_idle| = 150 MHz < |alpha| = 200 MHz, so the
# flux excursion never sweeps through the |11>-|02> or |11>-|20> two-photon
# resonances.  (The map starts from |10>, so those are irrelevant here anyway.)

H0 = (TWOPI * A1 / 2) * (n1 @ (n1 - Id)) \
   + (TWOPI * A2 / 2) * (n2 @ (n2 - Id)) \
   + TWOPI * G * (a1.conj().T @ a2 + a2.conj().T @ a1)


# ------------------------------------------------------------- propagators
def expmh(H, dt):
    """exp(-i H dt) for Hermitian H, via eigendecomposition."""
    w, v = np.linalg.eigh(H)
    return (v * np.exp(-1j * w * dt)) @ v.conj().T


def env(t, t_hold):
    """Flat-top with quarter-sine edges (FlatTopCosinePulse): rise sin(0->pi/2),
    flat top at 1, fall cos over pi/2.  Support = [0, 2*EDGE + t_hold]."""
    t = np.asarray(t, float)
    rise = np.sin(0.5 * np.pi * t / EDGE)                    # [0, EDGE]          : 0 -> 1
    fall = np.cos(0.5 * np.pi * (t - EDGE - t_hold) / EDGE)  # [EDGE+th, 2EDGE+th]: 1 -> 0
    e = np.where(t < EDGE, rise, np.where(t < EDGE + t_hold, 1.0, fall))
    return np.clip(e, 0.0, 1.0)


def detuning(t, t_hold, d_gate):
    """Delta(t): park at D_IDLE, dip to d_gate on the flat top (quarter-sine edges)."""
    return D_IDLE + (d_gate - D_IDLE) * env(t, t_hold)


def U_pulse(t_hold, d_gate):
    """Propagator across one flux swap pulse (ramp + hold + ramp)."""
    T  = 2 * EDGE + t_hold
    ns = int(round(T / DT))
    dt = T / ns
    U  = np.eye(DIM, dtype=complex)
    for k in range(ns):
        U = expmh(detuning((k + 0.5) * dt, t_hold, d_gate) * n1 + H0, dt) @ U
    return U


def U_wait(t_delay):
    """Propagator for parking at Delta_idle (exact -- H is time independent)."""
    return expmh(D_IDLE * n1 + H0, t_delay)


def U_rep(d_gate):
    """One repetition of the qc_n_swap_amp sequence: swap pulse + park delay."""
    U = U_pulse(T_HOLD, d_gate)
    if T_DELAY > 0:
        U = U_wait(T_DELAY) @ U
    return U


# -------------------------------------------------------------- extraction
def evolve(C, N, init):
    """States after 0..N applications of C."""
    psi = np.zeros(DIM, complex); psi[init] = 1.0
    U, out = np.eye(DIM, dtype=complex), []
    for _ in range(N + 1):
        out.append(U @ psi)
        U = C @ U
    return np.array(out)


# ================================================================ run info
t_pulse = 2 * EDGE + T_HOLD
t_rep   = t_pulse + T_DELAY
print("=" * 66)
print(f"swap pulse : edge={EDGE:.2f} ns/side  hold={T_HOLD:.2f} ns  -> {t_pulse:.2f} ns")
print(f"delay      : {T_DELAY:.2f} ns at park   (one repetition = {t_rep:.2f} ns)")
print(f"detuning   : [{DMIN:.1f}, {DMAX:.1f}] MHz  x {NDET} pts   (ctrl_amp axis)")
print(f"repetitions: N = 0 .. {NMAX}")
print("=" * 66)

# ================================================================== figure
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

# --- (a) one swap repetition: pulse + delay ------------------------------
tt = np.linspace(0, t_rep, 1400)
pulse_dd = lambda dg: np.where(tt <= t_pulse,
                               detuning(np.clip(tt, 0, t_pulse), T_HOLD, dg),
                               D_IDLE) / TWOPI * 1e3
ax[0].plot(tt, pulse_dd(0.0), lw=2.2, color="#1f4e79")
ax[0].axhline(0, color="crimson", ls="--", lw=1.1)
ax[0].axhspan(-8, 8, color="crimson", alpha=0.10)
# faint outlines of the swept flat-top levels (= the panel-(b) x-axis)
for dg in (DMIN, DMAX):
    ax[0].plot(tt, pulse_dd(TWOPI * dg * 1e-3), lw=1.0, ls=":", color="0.55")
if T_DELAY > 0:
    ax[0].axvspan(t_pulse, t_rep, color="0.85", alpha=0.7)
    ax[0].text(t_pulse + T_DELAY / 2, D_IDLE / TWOPI * 1e3 * 0.55,
               "delay\n(park)", ha="center", fontsize=9)
ax[0].text(EDGE + T_HOLD / 2, 16, "resonant\nexchange", ha="center",
           fontsize=9, color="crimson")
ax[0].text(0.02 * t_rep, D_IDLE / TWOPI * 1e3 - 6, "park\n(dispersive)",
           fontsize=9, va="top")
ax[0].set_xlabel("t  [ns]"); ax[0].set_ylabel(r"$\Delta/2\pi$  [MHz]")
ax[0].set_title("(a) one swap repetition: pulse + delay", loc="left", fontsize=11)
ax[0].set_ylim(min(DMIN, 0.0) - 15, D_IDLE / TWOPI * 1e3 + 22)
ax[0].grid(alpha=0.25)

# --- (b) qc_n_swap_amp map: P(|01>) vs N and control detuning ------------
d_axis = np.linspace(DMIN, DMAX, NDET)           # MHz
d_rad  = TWOPI * d_axis * 1e-3                    # rad/ns
P01 = np.array([abs(evolve(U_rep(dg), NMAX, S10)[:, S01])**2
                for dg in d_rad]).T              # (NMAX+1, NDET): rows = N, cols = detuning
im = ax[1].imshow(P01, origin="lower", aspect="auto",
                  extent=[DMIN, DMAX, -0.5, NMAX + 0.5],
                  cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
ax[1].axvline(0, color="w", ls="--", lw=1.0, alpha=0.6)      # resonance
cbar = fig.colorbar(im, ax=ax[1], pad=0.02)
cbar.set_label(r"$P(|01\rangle)$   (target, from $|10\rangle$)")
ax[1].set_xlabel(r"control detuning  $\Delta_{gate}/2\pi$   [MHz]   (= ctrl_amp)")
ax[1].set_ylabel("N   (number of swaps)")
ax[1].set_yticks(np.arange(0, NMAX + 1, 2))
ax[1].set_title(r"(b) qc_n_swap_amp map: $P(|01\rangle)$ vs $N$ and detuning",
                loc="left", fontsize=11)

fig.suptitle(
    rf"qc_n_swap_amp simulation  ($g/2\pi$={G*1e3:.1f} MHz,  "
    rf"$\Delta_{{idle}}/2\pi$={D_IDLE/TWOPI*1e3:.0f} MHz,  hold={T_HOLD:.1f} ns,  "
    rf"edge={EDGE:.1f} ns,  delay={T_DELAY:.1f} ns)", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("dc_flux_partial_swap.png", dpi=155)

# ------------------------------------------------------------------ summary
print("on-resonance (Delta_gate = 0) joint populations from |10>:")
print(f"{'N':>3} {'P10':>8} {'P01':>8} {'P10+P01':>9}")
sr = evolve(U_rep(0.0), NMAX, S10)
for N in range(0, NMAX + 1, 2):
    p10, p01 = abs(sr[N, S10])**2, abs(sr[N, S01])**2
    print(f"{N:>3} {p10:>8.4f} {p01:>8.4f} {p10+p01:>9.4f}")
print("=" * 66)
