"""
hybrid_optimizer.py
-------------------
Two-stage hybrid optimizer:
  Stage 1 – Differential Evolution (global, avoids local minima)
  Stage 2 – Levenberg-Marquardt / TRF (local, tight convergence)

Works with any ViscoelasticModel subclass.

References
----------
[2] Efremov YM et al. Sci Rep 2017; 7:1541.
    DOI: 10.1038/s41598-017-01784-3
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import differential_evolution, least_squares

from viscoelastic_models import ViscoelasticModel


@dataclass
class FitResult:
    """Stores everything produced by one model-fit run."""
    model_name  : str
    params      : np.ndarray          # fitted parameter vector
    std_errors  : np.ndarray          # ± standard errors (from Jacobian)
    R2          : float
    E_fit       : np.ndarray          # model prediction on the same t-grid
    de_seed     : np.ndarray          # DE's best guess before LM refinement
    converged   : bool = True

    def param_table(self, specs) -> str:
        lines = [f"  {'Parameter':<12} {'Value':>12}  {'±SE':>10}  {'Unit'}"]
        lines.append("  " + "-" * 46)
        for ps, v, se in zip(specs, self.params, self.std_errors):
            lines.append(f"  {ps.name:<12} {v:>12.4f}  {se:>10.4f}  {ps.unit}")
        lines.append(f"  {'R²':<12} {self.R2:>12.6f}")
        return "\n".join(lines)


class HybridOptimizer:
    """Fit a ViscoelasticModel to E*(t) data using DE → LM/TRF.

    Parameters
    ----------
    model   : any ViscoelasticModel instance
    t       : 1-D time array (s)
    E_data  : 1-D modulus array (Pa), same length as t
    de_kw   : extra kwargs forwarded to differential_evolution
    lm_kw   : extra kwargs forwarded to least_squares
    """

    _DE_DEFAULTS = dict(
        maxiter=4000, tol=1e-10, seed=42,
        popsize=20, mutation=(0.5, 1.2),
        recombination=0.85, polish=False,
        init="latinhypercube",
    )
    _LM_DEFAULTS = dict(
        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=50_000
    )

    def __init__(self,
                 model   : ViscoelasticModel,
                 t       : np.ndarray,
                 E_data  : np.ndarray,
                 de_kw   : dict | None = None,
                 lm_kw   : dict | None = None):
        self.model  = model
        self.t      = np.asarray(t,      dtype=float)
        self.E_data = np.asarray(E_data, dtype=float)
        self._de_kw = {**self._DE_DEFAULTS, **(de_kw or {})}
        self._lm_kw = {**self._LM_DEFAULTS, **(lm_kw or {})}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sse(self, params: np.ndarray) -> float:
        if not self.model.is_physically_valid(params):
            return 1e14
        residuals = self.E_data - self.model.predict(self.t, params)
        return float(np.dot(residuals, residuals))

    def _residuals(self, params: np.ndarray) -> np.ndarray:
        return self.E_data - self.model.predict(self.t, params)

    def _data_seed(self) -> np.ndarray | None:
        """Heuristic initial guess from the data itself.
        Einf ≈ median of last 10 % of points (long-time plateau).
        Works for both SLS and FKV shapes."""
        specs = self.model.param_specs
        if len(specs) < 2:
            return None
        Einf_est = float(np.median(self.E_data[-max(1, len(self.E_data)//10):]))
        seed = []
        for ps in specs:
            if ps.name in ("Einf", "E_inf"):
                seed.append(np.clip(Einf_est, ps.lower, ps.upper))
            elif ps.name == "E0":
                E0_est = float(self.E_data[0])
                seed.append(np.clip(E0_est, ps.lower, ps.upper))
            elif ps.name in ("C",):
                drop = max(float(self.E_data[0]) - Einf_est, 1.0)
                seed.append(np.clip(drop, ps.lower, ps.upper))
            elif ps.name in ("alpha", "α"):
                seed.append(0.18)
            elif ps.name == "tau":
                seed.append(1.0)
            else:
                seed.append(0.5 * (ps.lower + ps.upper))
        return np.array(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, verbose: bool = True) -> FitResult:
        """Run the two-stage optimization and return a FitResult."""

        # ---- Stage 1: Differential Evolution ----
        x0 = self._data_seed()
        de_kwargs = dict(self._de_kw)
        if x0 is not None:
            de_kwargs["x0"] = x0

        de_result = differential_evolution(
            self._sse,
            bounds=self.model.bounds,
            **de_kwargs,
        )
        theta0 = de_result.x

        # ---- Stage 2: Local refinement ----
        # Choose LM (unbounded, faster) when bounds are loose,
        # TRF (bounded) when parameter collinearity is a known risk.
        use_trf = any(
            ps.name in ("alpha", "C", "Einf")
            for ps in self.model.param_specs
        )
        method = "trf" if use_trf else "lm"
        lm_kw  = dict(self._lm_kw)

        if method == "trf":
            lm_result = least_squares(
                self._residuals, theta0, method="trf",
                bounds=(self.model.bounds_lower, self.model.bounds_upper),
                **lm_kw,
            )
        else:
            lm_result = least_squares(
                self._residuals, theta0, method="lm", **lm_kw
            )

        params  = lm_result.x
        E_fit   = self.model.predict(self.t, params)
        ss_res  = float(np.sum((self.E_data - E_fit) ** 2))
        ss_tot  = float(np.sum((self.E_data - self.E_data.mean()) ** 2))
        R2      = 1.0 - ss_res / ss_tot

        # Covariance from Jacobian
        J   = lm_result.jac
        cov = np.linalg.pinv(J.T @ J) * ss_res / max(len(self.E_data) - len(params), 1)
        se  = np.sqrt(np.abs(np.diag(cov)))

        result = FitResult(
            model_name = self.model.name,
            params     = params,
            std_errors = se,
            R2         = R2,
            E_fit      = E_fit,
            de_seed    = theta0,
            converged  = lm_result.success if hasattr(lm_result, "success") else True,
        )

        if verbose:
            print(f"\n[{self.model.name}]")
            print(f"  DE seed : {dict(zip([p.name for p in self.model.param_specs], theta0.round(4)))}")
            print(result.param_table(self.model.param_specs))

        return result
