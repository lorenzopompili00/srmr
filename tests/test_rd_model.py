"""Checks of the standalone ringdown model and of the NR extraction."""

import os

import numpy as np
import pytest
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from srmr.nr_dataset import find_peak_time
from srmr.qnm_utils import mu, qnm_omega_tau, spheroidal_from_spherical, spherical_from_spheroidal
from srmr.rd_model import Attachment, V5Ansatz, evaluate


def _attachment(lm=(2, 2), af=0.68, Mf=0.95):
    return Attachment.from_nr(
        lm, nu=0.2, af=af, Mf=Mf, amp0=0.35, damp0=-1e-3, phi0=0.7, omega0=-0.36
    )


def _pyseobnr_reference(tau, att, c1f, c2f, d1f, d2f):
    """Literal transcription of ``pyseobnr.eob.waveform.compute_MR.compute_MR_mode_free``.

    Kept independent of :mod:`srmr.rd_model` so that it actually tests the
    rescaled reformulation used there.
    """
    # pyseobnr works with the conjugated QNM: Re > 0, Im > 0.
    omega_complex = att.omega_qnm + 1j / att.tau_qnm
    sigmaR = -np.imag(omega_complex)
    sigmaI = -np.real(omega_complex)
    eta, nra, nraDot, nromega = att.nu, att.amp0, att.damp0, att.omega0

    c1c = 1 / (c1f * eta) * (nraDot - sigmaR * nra) * np.cosh(c2f) ** 2
    c2c = nra / eta - 1 / (c1f * eta) * (nraDot - sigmaR * nra) * np.cosh(c2f) * np.sinh(c2f)
    d1c = (nromega - sigmaI) * (1 + d2f) / (d1f * d2f)

    Alm = c1c * np.tanh(c1f * tau + c2f) + c2c
    philm = att.phi0 - d1c * np.log((1 + d2f * np.exp(-d1f * tau)) / (1 + d2f))
    test_omega = -np.real(omega_complex) + 1j * np.imag(omega_complex)
    return eta * Alm * np.exp(1j * philm) * np.exp(1j * test_omega * tau)


@pytest.mark.parametrize("lm", [(2, 2), (3, 3), (2, 1), (4, 4)])
def test_matches_pyseobnr_ansatz(lm):
    """The rescaled formulation reproduces the SEOBNRv5 merger-ringdown exactly."""
    att = _attachment(lm)
    coeffs = dict(c1f=0.1, c2f=-0.5, d1f=0.15, d2f=2.0)
    tau = np.linspace(0.0, 90.0, 500)

    amp, phase = evaluate(tau, att, V5Ansatz(enforce_peak=False), coeffs)
    reference = _pyseobnr_reference(tau, att, **coeffs)

    assert np.allclose(amp, np.abs(reference), rtol=1e-12)
    assert np.allclose(np.exp(1j * phase), reference / np.abs(reference), atol=1e-12)


def test_attachment_conditions():
    """Amplitude, its derivative, phase and frequency at tau=0 reproduce the input values."""
    att = _attachment()
    tau = np.linspace(0.0, 20.0, 2001)
    amp, phase = evaluate(tau, att, V5Ansatz(enforce_peak=False),
                          dict(c1f=0.1, c2f=-0.5, d1f=0.15, d2f=2.0))

    assert amp[0] == pytest.approx(att.amp0, rel=1e-12)
    assert phase[0] == pytest.approx(att.phi0, rel=1e-12)
    assert Spline(tau, amp, k=5).derivative()(0.0) == pytest.approx(att.damp0, rel=1e-6)
    assert Spline(tau, phase, k=5).derivative()(0.0) == pytest.approx(att.omega0, rel=1e-6)


def test_qnm_late_time_limit():
    """Far from the attachment the model decays at the QNM rate and frequency."""
    att = _attachment()
    tau = np.linspace(300.0, 400.0, 1001)
    amp, phase = evaluate(tau, att, V5Ansatz(enforce_peak=False),
                          dict(c1f=0.1, c2f=-0.5, d1f=0.15, d2f=2.0))

    decay = -Spline(tau, np.log(amp), k=3).derivative()(350.0)
    assert decay == pytest.approx(1 / att.tau_qnm, rel=1e-6)
    assert Spline(tau, phase, k=3).derivative()(350.0) == pytest.approx(-att.omega_qnm, rel=1e-6)


def test_peak_finding_beats_the_sampling_rate():
    """Root-finding recovers a peak that sits between two samples."""
    t = np.arange(-20.0, 20.0, 0.5)  # samples at multiples of 0.5
    t_true = 0.1234
    amp = np.exp(-((t - t_true) ** 2) / 50.0)

    assert abs(t[np.argmax(amp)] - t_true) > 0.1  # the grid cannot resolve it
    assert find_peak_time(t, amp) == pytest.approx(t_true, abs=1e-6)


@pytest.mark.parametrize("lm", [(3, 2), (4, 3)])
def test_mode_mixing_round_trip(lm):
    ell, m = lm
    af = 0.7
    rng = np.random.default_rng(0)
    h_lm = rng.normal(size=50) + 1j * rng.normal(size=50)
    h_mm = rng.normal(size=50) + 1j * rng.normal(size=50)

    s_h_lm0 = spheroidal_from_spherical(ell, m, af, h_lm, h_mm)
    s_h_mm0 = h_mm / np.conj(mu(m, m, m, af))
    assert np.allclose(spherical_from_spheroidal(ell, m, af, s_h_lm0, s_h_mm0), h_lm)


def test_truncate():
    """Truncating the stored window keeps the leading samples and nothing past t_end."""
    from srmr.dataset import ModeArrays

    tau = np.arange(0, 100.01, 0.1)
    m = ModeArrays(2, 2, "spherical", 0.0, tau, np.ones((3, tau.size)),
                   np.zeros((3, tau.size)), np.full(3, 150.0), None)

    assert m.truncate(None).tau[-1] == pytest.approx(100.0)
    assert m.truncate(200.0).tau[-1] == pytest.approx(100.0)
    short = m.truncate(40.0)
    assert short.tau[-1] == pytest.approx(40.0)
    assert short.amp.shape == (3, short.tau.size) == (3, 401)
    assert np.array_equal(short.tau, tau[:401])


def test_qnm_conventions():
    """Positive frequency and damping time, matching known Kerr values."""
    omega, tau = qnm_omega_tau(2, 2, af=0.68, Mf=1.0)
    assert omega > 0 and tau > 0
    assert omega == pytest.approx(0.5239751, rel=1e-5)
    assert tau == pytest.approx(1 / 0.0815126, rel=1e-5)


# The same simulation in both SXS catalogue formats, on the AEI cluster.
_OLD_0335 = "/work/sossokine/SimAnnex/Public/4d2sB/0011/Lev3"
_NEW_0335 = "/work/hpfeiffer/SimulationAnnex/PublicLinks/SXS:BBH:0335/Lev3"


@pytest.mark.skipif(
    not (os.path.isdir(_OLD_0335) and os.path.isdir(_NEW_0335)), reason="needs the SXS annex"
)
def test_new_format_matches_old_once_memory_is_removed():
    """The new-format reader, with the memory subtracted, reproduces the old files."""
    from srmr.mismatch import mismatch
    from srmr.nr_dataset import extract_simulation

    def mm(a, b):
        return mismatch(a.amp, a.phase - a.phase[0], b.amp, b.phase - b.phase[0], 0.1)

    old = extract_simulation(_OLD_0335, modes=[(2, 2)]).mode((2, 2))
    new = extract_simulation(_NEW_0335, modes=[(2, 2)]).mode((2, 2))
    with_memory = extract_simulation(_NEW_0335, modes=[(2, 2)], remove_memory=False).mode((2, 2))
    assert mm(old, new) < 1e-6
    assert mm(old, with_memory) > 10 * mm(old, new)
