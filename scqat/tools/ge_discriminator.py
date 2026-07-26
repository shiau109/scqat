"""Readout-discriminator knobs from g/e IQ blobs — the rotation + threshold solve.

The single-shot companion of :mod:`scqat.tools.discriminate`, and deliberately a
DIFFERENT job: ``discriminate_states`` trains a 2-D GMM and assigns each shot to
its nearest centre (an OFFLINE classification, in the IQ plane), while this module
solves for the two numbers an FPGA needs to do the same thing in REAL TIME on one
rotated quadrature — a demod rotation and a scalar threshold. Both instrument
backends express discrimination that way (QM ``integration_weights_angle`` +
``threshold``; Qblox ``acq_rotation`` + ``acq_threshold``), which is why the math
lives here rather than in either driver.

Mirrors the qualibrate ``07_iq_blobs`` analysis so a ``single_shot_readout`` run can
propose the discriminator without the GUI. Pure math — the caller turns these into
governed suggestions on the neutral ``readout_rotation_rad`` / ``readout_threshold``
/ ``readout_rus_threshold`` fields; nothing here touches an instrument.

UNITS: everything here is in the ACQUISITION frame's native units — whatever the
probe acquired in, the results come back in. The returned ``ge_threshold`` /
``rus_exit_threshold`` are therefore ALREADY in the units the vendor's threshold
field stores; the caller writes them as-is. Do NOT apply 07's ``* length / 2**12``
— that factor inverts a Volt conversion that this path never performs (it would be
a double-division).

ANGLE: a single final angle (with 07's ``+pi`` fix so the rotated excited blob sits
above ground) drives BOTH the returned rotation and the threshold frame.
``delta_angle_rad`` is the correction measured IN THE FRAME THE DATA ARRIVED IN —
it is not an absolute instrument setting, and turning it into one is the CALLER's
job, because the right rule depends on the vendor:

* If the rotation knob feeds back into acquisition — the demod itself is rotated,
  so the next run's blobs arrive already turned — the delta shrinks toward zero
  each round and the caller ACCUMULATES: ``new = current - delta``. QM's
  ``integration_weights_angle`` (folded into the integration weights) works this way.
* If the knob is applied only somewhere downstream — Qblox reads ``acq_rotation``
  only on the thresholded path, so plain integrated acquisitions come back in the
  raw frame forever — there is no feedback, the same delta is measured every run,
  and the caller must write it ABSOLUTELY: ``new = -delta``. Accumulating in this
  case has no fixed point and walks away by one full delta per run (chipA,
  2026-07-26: three accepted runs took the knob to 3x its correct value).

Radians here; a backend whose knob is in degrees converts at its own boundary.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

__all__ = ["compute_ge_discriminator"]


def _false_detections(threshold: float, ig_rot: np.ndarray, ie_rot: np.ndarray) -> float:
    """Total misclassification count at a rotated-I threshold (mirrors 07's helper)."""
    if np.mean(ig_rot) < np.mean(ie_rot):
        return float(np.sum(ig_rot > threshold) + np.sum(ie_rot < threshold))
    return float(np.sum(ig_rot < threshold) + np.sum(ie_rot > threshold))


def compute_ge_discriminator(mean_g, mean_e, shots_g, shots_e) -> dict:
    """Discriminator knobs from the g/e blob centers and the per-shot |g>/|e> clouds.

    Parameters
    ----------
    mean_g, mean_e : (I, Q)
        The ground / excited blob centers (e.g. the GMM means
        :func:`~scqat.tools.discriminate.discriminate_states` returns),
        acquisition-frame units.
    shots_g, shots_e : (I_array, Q_array)
        Per-shot clouds for the |g> / |e> preparations (same units).

    Returns
    -------
    dict with (all acquisition-frame units):
        delta_angle_rad : the rotation putting the g->e axis on +I with Ie > Ig,
            relative to the current weights rotation; the caller proposes
            ``readout_rotation_rad = current - delta_angle_rad``.
        ge_threshold : rotated-I decision threshold (false-detection minimum).
        rus_exit_threshold : rotated-I ground-blob mode (active-reset exit).
    """
    ig_c, qg_c = float(mean_g[0]), float(mean_g[1])
    ie_c, qe_c = float(mean_e[0]), float(mean_e[1])
    ig = np.asarray(shots_g[0], dtype=float).ravel()
    qg = np.asarray(shots_g[1], dtype=float).ravel()
    ie = np.asarray(shots_e[0], dtype=float).ravel()
    qe = np.asarray(shots_e[1], dtype=float).ravel()

    # 07's angle: put the g->e separation on the I axis (from the blob centers).
    angle = float(np.arctan2(qe_c - qg_c, ig_c - ie_c))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # +pi fix so the rotated EXCITED blob sits ABOVE ground (Ie_rot > Ig_rot).
    if (ig_c - ie_c) * cos_a - (qg_c - qe_c) * sin_a > 0:
        angle += np.pi
        cos_a, sin_a = np.cos(angle), np.sin(angle)

    ig_rot = ig * cos_a - qg * sin_a
    ie_rot = ie * cos_a - qe * sin_a

    # RUS exit = the mode (tallest histogram bin) of the rotated ground blob.
    counts, edges = np.histogram(ig_rot, bins=100)
    rus_exit_threshold = float(edges[1:][int(np.argmax(counts))])

    # ge threshold = the rotated-I cut minimizing total misclassification, seeded at
    # the midpoint of the two rotated blob means.
    seed = 0.5 * (float(np.mean(ig_rot)) + float(np.mean(ie_rot)))
    res = minimize(_false_detections, seed, args=(ig_rot, ie_rot), method="Nelder-Mead")
    ge_threshold = float(res.x[0])

    return {
        "delta_angle_rad": float(angle),
        "ge_threshold": ge_threshold,
        "rus_exit_threshold": rus_exit_threshold,
    }
