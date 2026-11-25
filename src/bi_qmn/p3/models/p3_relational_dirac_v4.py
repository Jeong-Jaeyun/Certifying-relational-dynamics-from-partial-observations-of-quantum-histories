"""Model C architecture v4: v3 with a bispectral energy-basis readout (development).

The only architectural change from v3 is the translation-invariant readout used for energy-basis
(``F``) outcomes. v3 used the DFT power spectrum |X_m(r)|^2 of the tau-pooled features, which discards
relative phase. v4 adds the bispectrum slice

    B_m(r) = X_m(1) X_m(r - 1) conj(X_m(r)),      r in Z_K,

which is invariant under clock translation (X_r -> X_r w^{rs} leaves B unchanged) but keeps relative
phase. For a generic signal with X(1) != 0 the slice determines the signal up to translation
(Kakarala 1992/2012; used as a learnable invariant in Sanborn et al. 2023). Its cubic magnitude is
mapped to degree one by B * (|B|^2 + eps^2)^(-1/3).

``readout="power"`` reproduces the v3 power-spectrum readout inside the v4 family (same
initialisation), so the readout hypothesis is tested with common random numbers.
"""

from __future__ import annotations

import numpy as np

from bi_qmn.p3.models.p3_relational_dirac_v3 import P3EquivariantClockDiracEstimator


MODEL_FAMILY_V4 = "p3_clock_equivariant_bispectral_dirac_free_table_v4"
READOUTS = ("bispectrum", "power")


class P3BispectralClockDiracEstimator(P3EquivariantClockDiracEstimator):
    model_family = MODEL_FAMILY_V4

    def __init__(self, *args, readout: str = "bispectrum", bispectrum_epsilon: float = 1.0e-6, **kwargs):
        if readout not in READOUTS:
            raise ValueError(f"unknown v4 readout: {readout}")
        if bispectrum_epsilon <= 0.0:
            raise ValueError("bispectrum epsilon must be positive")
        self.readout = readout
        self.bispectrum_epsilon = float(bispectrum_epsilon)
        super().__init__(*args, **kwargs)
        self.spectral_dim = self.m * (3 if readout == "bispectrum" else 1)
        self.params.C = np.zeros((self.static_dim, self.spectral_dim, 2))
        self.previous_index = (np.arange(self.K) - 1) % self.K

    # ------------------------------------------------------------------ spectral readout
    def _bispectrum(self, Fhat: np.ndarray) -> dict:
        X1 = Fhat[:, 1:2, :]
        Xprev = Fhat[:, self.previous_index, :]
        B = X1 * Xprev * np.conj(Fhat)
        s = np.abs(B) ** 2
        g = (s + self.bispectrum_epsilon**2) ** (-1.0 / 3.0)
        return {"X1": X1, "Xprev": Xprev, "B": B, "s": s, "g": g, "Bt": B * g}

    def spectral_features(self, Fhat: np.ndarray) -> tuple[np.ndarray, dict]:
        power = np.abs(Fhat) ** 2
        if self.readout == "power":
            return power, {}
        bis = self._bispectrum(Fhat)
        return np.concatenate([power, bis["Bt"].real, bis["Bt"].imag], axis=2), bis

    def forward(self, psi: np.ndarray, gates: np.ndarray, a: np.ndarray, eta: float) -> dict:
        cache = super().forward(psi, gates, a, eta)
        features, bis = self.spectral_features(cache["Fhat"])
        cache["power"] = cache["Pw"]
        cache["Pw"] = features  # the v3 decoder/gradient code reads the F readout from this slot
        cache["bispectrum"] = bis
        return cache

    def _tangent(self, cache: dict, gates: np.ndarray, eta: float, *, rho: int | None):
        dF, d_power, d_dirac = super()._tangent(cache, gates, eta, rho=rho)
        if self.readout == "power":
            return dF, d_power, d_dirac
        bis = cache["bispectrum"]
        Fhat = cache["Fhat"]
        dX = np.einsum("rk,tkm->trm", self.dft, dF, optimize=True)
        dB = dX[:, 1:2, :] * bis["Xprev"] * np.conj(Fhat) + bis["X1"] * dX[:, self.previous_index, :] * np.conj(Fhat) + bis["X1"] * bis["Xprev"] * np.conj(dX)
        ds = 2.0 * np.real(np.conj(bis["B"]) * dB)
        dg = -(1.0 / 3.0) * (bis["s"] + self.bispectrum_epsilon**2) ** (-4.0 / 3.0) * ds
        dBt = dB * bis["g"] + bis["B"] * dg
        return dF, np.concatenate([d_power, dBt.real, dBt.imag], axis=2), d_dirac

    def manifest(self) -> dict:
        manifest = super().manifest()
        manifest["model_family"] = self.model_family
        manifest["readout"] = (
            "tau_pooled_equivariant_D_and_power_plus_bispectrum_slice_invariant_F_v4"
            if self.readout == "bispectrum"
            else "tau_pooled_equivariant_D_and_power_spectrum_invariant_F_v4_control"
        )
        manifest["f_readout_kind"] = self.readout
        manifest["bispectrum_epsilon"] = self.bispectrum_epsilon
        return manifest
