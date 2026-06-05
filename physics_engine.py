"""
physics_engine.py
-----------------
AFM contact mechanics: Dimitriadis finite-thickness correction
and Hertz spherical-indenter model.

All lengths in metres, forces in Newtons, moduli in Pascals.

References
----------
[4] Dimitriadis EK et al. Biophys J 2002; 82:2798-2810.
    DOI: 10.1016/S0006-3495(02)75620-8
[5] Abuhattum S et al. iScience 2022; 25:104016.
    DOI: 10.1016/j.isci.2022.104016
"""

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeGeometry:
    """Immutable descriptor for the AFM colloidal probe.

    Parameters
    ----------
    R : float
        Sphere radius (m).  Verified by SEM before experiments.
    delta_0 : float
        Fixed indentation depth for stress-relaxation hold (m).
        Should be < 10 % of cell thickness to minimise substrate
        artefacts.
    """
    R: float
    delta_0: float

    def __post_init__(self):
        if self.R <= 0 or self.delta_0 <= 0:
            raise ValueError("R and delta_0 must be positive.")


class DimitriadisCorrection:
    """Substrate-effect correction for a spherical indenter on a finite
    soft layer bonded to a rigid substrate.

    The standard Hertz model assumes an infinitely deep half-space.
    For cells (h ≈ 5–15 µm) the rigid glass substrate stiffens the
    apparent response by 10–20 %; ignoring this overestimates E*.

    Parameters
    ----------
    probe : ProbeGeometry
    h : float
        Cell thickness at the indentation site (m).
    bonded : bool
        True  → cell adhered to substrate (bonded boundary, typical).
        False → free-slip boundary (gel on non-adhesive coating).

    Reference
    ---------
    [4] Dimitriadis et al., Biophys J 2002.
        Eq. (12) bonded / Eq. (11) unbonded.
    """

    # Polynomial coefficients for bonded and unbonded cases [4]
    _COEFF_BONDED   = (1.0, -0.884,  0.781, -0.386,  0.0179)
    _COEFF_UNBONDED = (1.0, -0.626,  0.288,  0.157, -0.0524)

    def __init__(self, probe: ProbeGeometry, h: float, bonded: bool = True):
        if h <= 0:
            raise ValueError("Cell thickness h must be positive.")
        self.probe   = probe
        self.h       = h
        self.bonded  = bonded
        self._chi    = np.sqrt(probe.R * probe.delta_0) / h
        self._factor = self._compute_factor()

    def _compute_factor(self) -> float:
        coeffs = self._COEFF_BONDED if self.bonded else self._COEFF_UNBONDED
        return sum(c * self._chi**i for i, c in enumerate(coeffs))

    @property
    def chi(self) -> float:
        """Dimensionless geometry parameter χ = √(R·δ₀) / h."""
        return self._chi

    @property
    def factor(self) -> float:
        """Correction multiplier f_dim ∈ (0, 1].
        Values below 0.85 indicate a strong substrate influence."""
        return self._factor

    @property
    def substrate_influence_pct(self) -> float:
        """Percentage overestimation if correction is ignored."""
        return (1.0 - self._factor) * 100.0

    def summary(self) -> str:
        bc = "bonded" if self.bonded else "unbonded"
        return (f"Dimitriadis ({bc}): χ={self._chi:.4f}, "
                f"f_dim={self._factor:.4f}, "
                f"substrate effect={self.substrate_influence_pct:.1f}%")


class HertzContact:
    """Hertz spherical-indenter model with optional Dimitriadis correction.

    Converts between measured force F(t) and reduced modulus E*(t):
        F(t) = (4/3) · E*(t) · √R · δ₀^(3/2) · f_dim

    The algebraic inversion is valid for ramp-hold AFM experiments when
    t_ramp ≪ t_hold, so the Ting convolution integral reduces to this
    closed form.  See [5], Appendix A.

    Parameters
    ----------
    probe : ProbeGeometry
    correction : DimitriadisCorrection or None
        Pass None to use uncorrected Hertz (not recommended for cells).
    """

    def __init__(self,
                 probe: ProbeGeometry,
                 correction: DimitriadisCorrection | None = None):
        self.probe      = probe
        self.correction = correction
        self._prefactor = self._build_prefactor()

    def _build_prefactor(self) -> float:
        f = self.correction.factor if self.correction else 1.0
        return (4.0 / 3.0) * np.sqrt(self.probe.R) * self.probe.delta_0**1.5 * f

    def modulus_to_force(self, E_star: np.ndarray) -> np.ndarray:
        """F(t) = prefactor · E*(t).  Vectorised over time."""
        return self._prefactor * np.asarray(E_star)

    def force_to_modulus(self, F: np.ndarray) -> np.ndarray:
        """E*(t) = F(t) / prefactor.  Inverse Hertz-Dimitriadis."""
        F = np.asarray(F)
        if np.any(F <= 0):
            raise ValueError("All force values must be positive (compressive).")
        return F / self._prefactor
