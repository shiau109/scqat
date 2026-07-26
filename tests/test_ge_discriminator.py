"""Pure-numpy tests for the readout-discriminator solve (no instrument, no dataset).

Verifies compute_ge_discriminator against the 07_iq_blobs semantics: the rotation
lands the g->e axis on I with the excited blob ABOVE ground, the threshold beats the
naive midpoint on asymmetric clouds, RUS sits at the ground blob, and everything is
returned in the input (acquisition-frame) units with no hidden scaling.
"""

import numpy as np
import pytest

from scqat.tools.ge_discriminator import _false_detections, compute_ge_discriminator


def _clouds(theta, sep=6.0, sig_g=1.0, sig_e=1.0, n=4000, seed=0, origin=(0.0, 0.0)):
    """Two Gaussian blobs separated by `sep` along a line at angle `theta`, plus their
    centers. `sig_e` != `sig_g` makes the optimal threshold differ from the midpoint."""
    rng = np.random.default_rng(seed)
    ox, oy = origin
    g_c = np.array([ox, oy])
    e_c = np.array([ox + sep * np.cos(theta), oy + sep * np.sin(theta)])
    ig = g_c[0] + rng.normal(0, sig_g, n)
    qg = g_c[1] + rng.normal(0, sig_g, n)
    ie = e_c[0] + rng.normal(0, sig_e, n)
    qe = e_c[1] + rng.normal(0, sig_e, n)
    return tuple(g_c), tuple(e_c), (ig, qg), (ie, qe)


@pytest.mark.parametrize("theta", np.linspace(-np.pi, np.pi, 9))
def test_rotation_puts_excited_above_ground(theta):
    g_c, e_c, sg, se = _clouds(theta, seed=1)
    d = compute_ge_discriminator(g_c, e_c, sg, se)
    a = d["delta_angle_rad"]
    c, s = np.cos(a), np.sin(a)
    ig_rot_mean = np.mean(sg[0]) * c - np.mean(sg[1]) * s
    ie_rot_mean = np.mean(se[0]) * c - np.mean(se[1]) * s
    # +pi fix guarantees the excited blob sits above ground on the rotated I axis
    assert ie_rot_mean > ig_rot_mean
    # threshold lies strictly between the two rotated blob centers
    assert ig_rot_mean < d["ge_threshold"] < ie_rot_mean


def test_threshold_beats_midpoint_on_asymmetric_blobs():
    g_c, e_c, sg, se = _clouds(0.7, sig_g=0.8, sig_e=2.2, seed=2)
    d = compute_ge_discriminator(g_c, e_c, sg, se)
    a = d["delta_angle_rad"]
    c, s = np.cos(a), np.sin(a)
    ig_rot = sg[0] * c - sg[1] * s
    ie_rot = se[0] * c - se[1] * s
    midpoint = 0.5 * (np.mean(ig_rot) + np.mean(ie_rot))
    # the optimizer's threshold misclassifies no more than the midpoint (usually fewer)
    assert _false_detections(d["ge_threshold"], ig_rot, ie_rot) <= _false_detections(midpoint, ig_rot, ie_rot)


def test_rus_exit_at_ground_blob():
    g_c, e_c, sg, se = _clouds(1.3, sep=8.0, seed=3)
    d = compute_ge_discriminator(g_c, e_c, sg, se)
    a = d["delta_angle_rad"]
    c, s = np.cos(a), np.sin(a)
    ig_rot_mean = float(np.mean(sg[0] * c - sg[1] * s))
    ie_rot_mean = float(np.mean(se[0] * c - se[1] * s))
    sep_rot = ie_rot_mean - ig_rot_mean
    # the RUS exit is the ground-blob mode -> near the ground center, far from excited
    assert abs(d["rus_exit_threshold"] - ig_rot_mean) < 0.2 * sep_rot


def test_units_are_preserved_no_hidden_scaling():
    """Scaling the input clouds by a constant scales threshold + RUS by the same
    constant (no length/2**12-style factor), and leaves the angle unchanged."""
    g_c, e_c, sg, se = _clouds(0.9, sep=6.0, seed=4)
    base = compute_ge_discriminator(g_c, e_c, sg, se)

    k = 137.0
    g_k = (g_c[0] * k, g_c[1] * k)
    e_k = (e_c[0] * k, e_c[1] * k)
    sg_k = (sg[0] * k, sg[1] * k)
    se_k = (se[0] * k, se[1] * k)
    scaled = compute_ge_discriminator(g_k, e_k, sg_k, se_k)

    assert scaled["delta_angle_rad"] == pytest.approx(base["delta_angle_rad"], abs=1e-9)
    assert scaled["ge_threshold"] == pytest.approx(base["ge_threshold"] * k, rel=1e-6)
    assert scaled["rus_exit_threshold"] == pytest.approx(base["rus_exit_threshold"] * k, rel=1e-6)


def test_exported_from_the_tools_package():
    """The drivers import it from `scqat.tools`, not the module path."""
    from scqat.tools import compute_ge_discriminator as exported

    assert exported is compute_ge_discriminator
