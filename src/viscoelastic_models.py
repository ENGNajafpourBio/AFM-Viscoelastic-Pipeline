"""
viscoelastic_models.py
----------------------
Viscoelastic relaxation models for AFM stress-relaxation experiments.

All models implement the ViscoelasticModel interface, so new models
can be added without touching the optimizer or the main script.

References
----------
[1] Efremov YM, Okajima T, Raman A. Soft Matter 2020; 16:64-81.
    DOI: 10.1039/c9sm01020c
[2] Efremov YM et al. Sci Rep 2017; 7:1541.
    DOI: 10.1038/s41598-017-01784-3
[3] Weber A et al. Microsc Res Tech 2022; 85:3284-3295.
    DOI: 10.1002/jemt.24184
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class ParamSpec:
    """Specification for a single model parameter."""
    name: str        # e.g. 'E0'
    unit: str        # e.g. 'Pa'
    lower: float     # lower search bound
    upper: float     # upper search bound
    description: str = ""


class ViscoelasticModel(ABC):
    """Abstract base class for viscoelastic relaxation models.

    Subclass and implement:
        • param_specs  – list of ParamSpec objects
        • predict()    – E*(t, params) → ndarray

    Everything else (bounds extraction, pretty-printing) is inherited.
    """

    @property
    @abstractmethod
    def param_specs(self) -> list[ParamSpec]:
        """Ordered list of parameter specifications."""

    @abstractmethod
    def predict(self, t: np.ndarray, params: Sequence[float]) -> np.ndarray:
        """Compute E*(t) for given parameter vector.

        Parameters
        ----------
        t      : 1-D array of time points (s)
        params : sequence matching param_specs order

        Returns
        -------
        E_model : 1-D array (Pa), same length as t
        """

    # ----------------------------------------------------------------
    # Convenience helpers – no need to override in subclasses
    # ----------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def n_params(self) -> int:
        return len(self.param_specs)

    @property
    def bounds_lower(self) -> list[float]:
        return [p.lower for p in self.param_specs]

    @property
    def bounds_upper(self) -> list[float]:
        return [p.upper for p in self.param_specs]

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """List of (lower, upper) tuples for scipy optimizers."""
        return [(p.lower, p.upper) for p in self.param_specs]

    def param_dict(self, values: Sequence[float]) -> dict:
        """Map a parameter vector to a {name: value} dictionary."""
        return {ps.name: v for ps, v in zip(self.param_specs, values)}

    def is_physically_valid(self, params: Sequence[float]) -> bool:
        """Check whether parameters satisfy hard physical constraints.
        Subclasses may override for additional checks."""
        for ps, v in zip(self.param_specs, params):
            if not (ps.lower <= v <= ps.upper):
                return False
        return True

    def summary_equation(self) -> str:
        """Return the model equation as a string (for reports)."""
        return f"{self.name}: [equation not defined]"


# ====================================================================
# Concrete models
# ====================================================================

class SLSModel(ViscoelasticModel):
    """Standard Linear Solid (three-element Maxwell-Voigt network).

    E*(t) = E_inf + (E0 - E_inf) · exp(−t / τ)

    Parameters
    ----------
    E0   : instantaneous (glassy) modulus  [Pa]
    Einf : equilibrium (rubbery) modulus   [Pa];  E_inf < E0
    tau  : relaxation time                 [s]

    Physical meaning
    ----------------
    τ = η / E2 (dashpot viscosity over spring constant).
    Healthy cells: τ ≈ 0.5–2 s  [3]
    Infected cells: τ increases due to cross-link disruption  [ref 6]

    References: [1] Eq. 3, [2] Table 2, [3] Fig. 5
    """

    @property
    def param_specs(self) -> list[ParamSpec]:
        return [
            ParamSpec("E0",   "Pa", 50.0,  5000.0, "Instantaneous modulus"),
            ParamSpec("Einf", "Pa", 10.0,  2000.0, "Equilibrium modulus"),
            ParamSpec("tau",  "s",  0.01,  20.0,   "Relaxation time"),
        ]

    def is_physically_valid(self, params: Sequence[float]) -> bool:
        E0, Einf, tau = params
        return (E0 > Einf > 0) and (tau > 0)

    def predict(self, t: np.ndarray, params: Sequence[float]) -> np.ndarray:
        E0, Einf, tau = params
        return Einf + (E0 - Einf) * np.exp(-np.asarray(t) / tau)

    def summary_equation(self) -> str:
        return "E*(t) = E_inf + (E0 - E_inf)·exp(−t/τ)"


class FKVModel(ViscoelasticModel):
    """Fractional Kelvin-Voigt model (spring + springpot in parallel).

    E*(t) = E_inf + C · t^(−α)

    Parameters
    ----------
    Einf  : long-time elastic modulus (equilibrium spring)  [Pa]
    C     : springpot prefactor                             [Pa·s^α]
    alpha : fractional order ∈ (0, 1)
            α → 0 : purely elastic solid
            α → 1 : Newtonian viscous fluid

    Physical meaning
    ----------------
    α encodes the viscoelastic memory of the cytoskeleton.
    Healthy cells:  α ≈ 0.15–0.22  (structured actin network)  [1, 2]
    Infected cells: α increases due to actin depolymerization   [ref 6]

    Note on bounds
    --------------
    α is bounded to [0.05, 0.95] instead of (0, 1) to avoid
    the degenerate collinearity between Einf and C that arises
    when t^(−α) ≈ const (α → 0).

    References: [1] Eq. 18, [2] PLR model
    """

    @property
    def param_specs(self) -> list[ParamSpec]:
        return [
            ParamSpec("Einf",  "Pa",      10.0, 2000.0, "Equilibrium modulus"),
            ParamSpec("C",     "Pa·s^α",   0.01, 5e4,   "Springpot prefactor"),
            ParamSpec("alpha", "—",        0.05, 0.95,  "Fractional order"),
        ]

    def predict(self, t: np.ndarray, params: Sequence[float]) -> np.ndarray:
        Einf, C, alpha = params
        return Einf + C * np.asarray(t) ** (-alpha)

    def summary_equation(self) -> str:
        return "E*(t) = E_inf + C·t^(−α)"


# ====================================================================
# Registry – makes it trivial to add new models in the future
# ====================================================================

_MODEL_REGISTRY: dict[str, type[ViscoelasticModel]] = {
    "SLS": SLSModel,
    "FKV": FKVModel,
}


def get_model(name: str) -> ViscoelasticModel:
    """Instantiate a model by its registry key.

    Usage
    -----
    >>> model = get_model("FKV")
    >>> E_pred = model.predict(t, [350.0, 520.0, 0.18])
    """
    name = name.upper()
    if name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY)
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return _MODEL_REGISTRY[name]()


def list_models() -> list[str]:
    """Return all registered model names."""
    return list(_MODEL_REGISTRY.keys())
