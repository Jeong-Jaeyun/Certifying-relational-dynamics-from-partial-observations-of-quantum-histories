"""Model C architecture v3: Z_K-equivariant clock-relational Dirac free-table estimator (development).

Motivation (``documents/P3-model-c-development-run2.md``): in v2 every edge/node feature had its
own free decoder weight, so a permuted or rewired graph was largely a reparameterisation of the
same feature family. v3 makes the relational structure act through weight sharing and symmetry:

1. Scaffold: all K anchors (P1 atlas convention) so clock translation tau -> tau + 1 is a graph
   automorphism (a, u) -> (a + 1, u).
2. Transport strength shared per relation type; one gauge-covariant Dirac propagation step
   chi = psi + eta * gamma * N(psi), N_u = sum_e kappa_e s_e (U_e psi_v - psi_u).
3. Gauge-invariant node features are pooled per relation type, then per clock coordinate tau.
4. Readout is equivariant: time-basis (``D``) outcome tau reads the tau-pooled features; energy-basis
   (``F``) outcome r reads the translation-invariant DFT power spectrum plus an r-specific bias.
   Clock translation is diagonal in the energy basis, so the ideal map is exactly this equivariant
   form for a phase-uniform generator family (group-equivariant design, Cohen & Welling 2016;
   Kondor & Trivedi 2018).

Latent field, transports, residuals, and propagated fields are estimator-internal, never physical.
Gradients: decoder parameters analytically; the few graph scalars (one transport strength per
relation type and eta) by exact forward-mode tangents.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from bi_qmn.core.observations import ShotBatch
from bi_qmn.p3.p3_clock_graph import ClockCoordinateGraph, hermitian_unitary_gamma, relation_generator_v2
from bi_qmn.p3.p3_records import P3ObservationRecord, p3_target_split_valid
from bi_qmn.p3.models.p3_relational_dirac import P3_OUTCOMES, frozen_context_descriptor
from bi_qmn.core.types import MeasurementContext, MeasurementTable


MODEL_FAMILY_V3 = "p3_clock_equivariant_relational_dirac_free_table_v3"


def _table_matrix(table: MeasurementTable, context_id: str, K: int) -> np.ndarray | None:
    probabilities = table.probabilities.get(context_id)
    if probabilities is None:
        return None
    return np.asarray([[float(probabilities.get(f"c{tau}_s{bit}", 0.0)) for bit in range(2)] for tau in range(K)], dtype=float)


@dataclass
class _Params:
    a: np.ndarray  # (n_types,) transport strength per relation type
    eta: float
    A: np.ndarray  # (J, m, 2) D readout
    betaD: np.ndarray  # (J, 2)
    C: np.ndarray  # (J, m, 2) F readout
    gammaF: np.ndarray  # (J, K, 2)

    def decoder_arrays(self) -> tuple[np.ndarray, ...]:
        return self.A, self.betaD, self.C, self.gammaF


class P3EquivariantClockDiracEstimator:
    model_family = MODEL_FAMILY_V3
    trained_target_channels = ("shot", "comp")
    report_only_target_channels = ("op",)

    def __init__(
        self,
        input_contexts: list[MeasurementContext],
        graph: ClockCoordinateGraph,
        *,
        latent_dim: int = 4,
        lambda_dirac: float = 0.05,
        lambda_mask: float = 0.25,
        lambda_proxy: float = 0.0,
        l2: float = 1.0e-4,
        learning_rate: float = 0.03,
        epochs: int = 120,
        init_seed: int = 0,
        propagate: bool = True,
    ):
        if lambda_dirac < 0.0 or lambda_mask < 0.0 or lambda_proxy != 0.0 or l2 < 0.0 or learning_rate <= 0.0 or epochs <= 0:
            raise ValueError("invalid Model C v3 hyperparameters")
        self.input_contexts = list(input_contexts)
        self.graph = graph
        self.K = int(graph.K)
        self.latent_dim = int(latent_dim)
        self.lambda_dirac = float(lambda_dirac)
        self.lambda_mask = float(lambda_mask)
        self.lambda_proxy = float(lambda_proxy)
        self.l2 = float(l2)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.init_seed = int(init_seed)
        self.propagate = bool(propagate)
        self.d_contexts = [c for c in self.input_contexts if c.clock == "D"]
        self.f_contexts = [c for c in self.input_contexts if c.clock == "F"]
        self.node_dim = 3 * len(self.d_contexts) + len(self.f_contexts) + 2
        self.edge_dim = 4 * len(self.f_contexts) + 1
        self.relation_types = graph.relation_types
        self.lags = tuple(sorted({edge.lag for edge in graph.edges}))
        n_types = len(self.relation_types)

        seed = int(hashlib.sha256(f"{self.model_family}:{self.latent_dim}:{self.init_seed}".encode("ascii")).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        shape = (self.latent_dim, self.node_dim)
        self.encoder = (rng.normal(size=shape) + 1.0j * rng.normal(size=shape)) / math.sqrt(2.0 * self.node_dim)
        self.gate_weights = rng.normal(size=(n_types, self.edge_dim)) / math.sqrt(self.edge_dim)
        self.phase_basis = np.linspace(-1.0, 1.0, self.latent_dim, dtype=float)
        self.gamma = hermitian_unitary_gamma(self.latent_dim)

        E, V = len(graph.edges), len(graph.nodes)
        self.n_edges, self.n_nodes = E, V
        self.src = np.asarray([e.source for e in graph.edges], dtype=int)
        self.tgt = np.asarray([e.target for e in graph.edges], dtype=int)
        self.coef = np.asarray([e.sign * e.scale for e in graph.edges], dtype=float)
        type_index = {name: i for i, name in enumerate(self.relation_types)}
        lag_index = {lag: i for i, lag in enumerate(self.lags)}
        self.edge_type = np.asarray([type_index[e.relation] for e in graph.edges], dtype=int)
        self.edge_lag = np.asarray([lag_index[e.lag] for e in graph.edges], dtype=int)
        self.node_tau = np.asarray(graph.node_tau, dtype=int)
        self.incidence = np.zeros((V, E), dtype=float)
        self.incidence[self.src, np.arange(E)] = 1.0
        self.type_mean = np.zeros((n_types, V, E), dtype=float)
        for rho in range(n_types):
            mask = self.edge_type == rho
            for u in range(V):
                out = np.flatnonzero(mask & (self.src == u))
                if out.size:
                    self.type_mean[rho, u, out] = 1.0 / out.size
        self.pool = np.zeros((self.K, V), dtype=float)
        for tau in range(self.K):
            members = np.flatnonzero(self.node_tau == tau)
            if members.size:
                self.pool[tau, members] = 1.0 / members.size
        k = np.arange(self.K)
        self.dft = np.exp(-2.0j * math.pi * np.outer(k, k) / self.K) / self.K
        self.m = 2 + 2 * n_types
        self.static_dim = len(frozen_context_descriptor(self.input_contexts[0])) + 3
        J = self.static_dim
        self.params = _Params(
            a=np.zeros(n_types),
            eta=0.0,
            A=np.zeros((J, self.m, 2)),
            betaD=np.zeros((J, 2)),
            C=np.zeros((J, self.m, 2)),
            gammaF=np.zeros((J, self.K, 2)),
        )
        self.checkpoint_hash = "unfitted"
        self.training_summary: dict = {}

    # ------------------------------------------------------------------ inputs
    def tau_features(self, table: MeasurementTable) -> np.ndarray:
        n_shots = float(table.metadata.get("n_shots", 0))
        features = np.zeros((self.K, self.node_dim), dtype=float)
        column = 0
        for context in self.d_contexts:
            matrix = _table_matrix(table, context.context_id, self.K)
            if matrix is not None:
                features[:, column : column + 2] = self.K * matrix
                features[:, column + 2] = 1.0
            column += 3
        for context in self.f_contexts:
            features[:, column] = 1.0 if context.context_id in table.probabilities else 0.0
            column += 1
        features[:, column] = 1.0
        features[:, column + 1] = math.log1p(n_shots) / 10.0
        return features

    def lag_features(self, table: MeasurementTable) -> np.ndarray:
        features = np.zeros((len(self.lags), self.edge_dim), dtype=float)
        r = np.arange(self.K)
        for position, lag in enumerate(self.lags):
            phase = np.exp(2.0j * math.pi * r * lag / self.K)
            column = 0
            for context in self.f_contexts:
                matrix = _table_matrix(table, context.context_id, self.K)
                if matrix is not None:
                    coherence = phase @ matrix
                    features[position, column : column + 4] = [coherence[0].real, coherence[0].imag, coherence[1].real, coherence[1].imag]
                column += 4
            features[position, column] = 1.0
        return features

    def table_inputs(self, table: MeasurementTable) -> tuple[np.ndarray, np.ndarray]:
        raw = self.tau_features(table)[self.node_tau] @ self.encoder.T
        psi = raw / (np.sqrt(np.sum(np.abs(raw) ** 2, axis=1, keepdims=True)) + 1.0e-12)
        gates = np.tanh(np.sum(self.gate_weights[self.edge_type] * self.lag_features(table)[self.edge_lag], axis=1))
        return psi, gates

    def static_features(self, table: MeasurementTable, context: MeasurementContext) -> np.ndarray:
        observed = sum(c.context_id in table.probabilities for c in self.input_contexts) / float(len(self.input_contexts))
        n_shots = float(table.metadata.get("n_shots", 0))
        return np.concatenate([frozen_context_descriptor(context), [observed, math.log1p(n_shots) / 10.0, 1.0]])

    # ------------------------------------------------------------------ graph forward
    def _theta(self, a: np.ndarray, gates: np.ndarray) -> np.ndarray:
        return a[self.edge_type][None, :, None] * gates[:, :, None] * self.phase_basis[None, None, :]

    def _node_features(self, chi, Tchi, N, res):
        bil = np.sum(np.conj(chi[:, self.src, :]) * Tchi, axis=2)
        chinorm = np.sum(np.abs(chi) ** 2, axis=2)
        parts = [chinorm[..., None], res[..., None]]
        for rho in range(len(self.relation_types)):
            parts.append((bil.real @ self.type_mean[rho].T)[..., None])
            parts.append((bil.imag @ self.type_mean[rho].T)[..., None])
        return np.concatenate(parts, axis=2), bil

    def forward(self, psi: np.ndarray, gates: np.ndarray, a: np.ndarray, eta: float) -> dict:
        P = np.exp(1.0j * self._theta(a, gates))
        Tpsi = P * psi[:, self.tgt, :]
        hop = self.coef[None, :, None] * (Tpsi - psi[:, self.src, :])
        N = np.einsum("ve,ter->tvr", self.incidence, hop, optimize=True)
        res = np.sum(np.abs(N) ** 2, axis=2)
        GN = N @ self.gamma.T
        chi = psi + eta * GN
        Tchi = P * chi[:, self.tgt, :]
        f, bil = self._node_features(chi, Tchi, N, res)
        F = np.einsum("kv,tvm->tkm", self.pool, f, optimize=True)
        Fhat = np.einsum("rk,tkm->trm", self.dft, F, optimize=True)
        return {"P": P, "Tpsi": Tpsi, "N": N, "res": res, "GN": GN, "chi": chi, "Tchi": Tchi, "F": F, "Fhat": Fhat, "Pw": np.abs(Fhat) ** 2, "dirac": np.mean(res, axis=1), "bil": bil}

    def _tangent(self, cache: dict, gates: np.ndarray, eta: float, *, rho: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact forward-mode derivative of (F, Pw, dirac) w.r.t. a[rho] (rho given) or eta (rho None)."""
        P, Tchi, chi, N = cache["P"], cache["Tchi"], cache["chi"], cache["N"]
        if rho is not None:
            dtheta = (self.edge_type == rho)[None, :, None] * gates[:, :, None] * self.phase_basis[None, None, :]
            dTpsi = 1.0j * dtheta * cache["Tpsi"]
            dN = np.einsum("ve,ter->tvr", self.incidence, self.coef[None, :, None] * dTpsi, optimize=True)
            dres = 2.0 * np.real(np.sum(np.conj(N) * dN, axis=2))
            dchi = eta * (dN @ self.gamma.T)
            dTchi = 1.0j * dtheta * Tchi + P * dchi[:, self.tgt, :]
        else:
            dres = np.zeros_like(cache["res"])
            dchi = cache["GN"]
            dTchi = P * dchi[:, self.tgt, :]
        dbil = np.sum(np.conj(dchi[:, self.src, :]) * Tchi + np.conj(chi[:, self.src, :]) * dTchi, axis=2)
        dchinorm = 2.0 * np.real(np.sum(np.conj(chi) * dchi, axis=2))
        parts = [dchinorm[..., None], dres[..., None]]
        for r in range(len(self.relation_types)):
            parts.append((dbil.real @ self.type_mean[r].T)[..., None])
            parts.append((dbil.imag @ self.type_mean[r].T)[..., None])
        df = np.concatenate(parts, axis=2)
        dF = np.einsum("kv,tvm->tkm", self.pool, df, optimize=True)
        dFhat = np.einsum("rk,tkm->trm", self.dft, dF, optimize=True)
        dPw = 2.0 * np.real(np.conj(cache["Fhat"]) * dFhat)
        return dF, dPw, np.mean(dres, axis=1)

    # ------------------------------------------------------------------ decoder
    def _logits(self, cache: dict, static: np.ndarray, table_index: np.ndarray, is_F: np.ndarray, params: _Params) -> np.ndarray:
        logits = np.zeros((static.shape[0], self.K, 2), dtype=float)
        d, f = ~is_F, is_F
        if np.any(d):
            logits[d] = np.einsum("nj,nkm,jms->nks", static[d], cache["F"][table_index[d]], params.A, optimize=True) + (static[d] @ params.betaD)[:, None, :]
        if np.any(f):
            logits[f] = np.einsum("nj,nkm,jms->nks", static[f], cache["Pw"][table_index[f]], params.C, optimize=True) + np.einsum("nj,jks->nks", static[f], params.gammaF, optimize=True)
        return logits.reshape(static.shape[0], 2 * self.K)

    # ------------------------------------------------------------------ training
    @staticmethod
    def _counts(batch: ShotBatch, context: MeasurementContext) -> np.ndarray:
        try:
            return np.asarray([batch.counts[context.context_id][o] for o in P3_OUTCOMES], dtype=float)
        except KeyError as error:
            raise ValueError(f"missing target count for {context.context_id}") from error

    @staticmethod
    def _masked(table: MeasurementTable, hidden: str) -> MeasurementTable:
        return MeasurementTable(
            probabilities={k: dict(v) for k, v in table.probabilities.items() if k != hidden},
            contexts=[c for c in table.contexts if c.context_id != hidden],
            metadata={**table.metadata, "p3_masked_context": hidden},
            source_density=None,
        )

    def _design(self, records: list[P3ObservationRecord]) -> dict:
        tables, table_index, statics, counts, weights, is_F = [], [], [], [], [], []
        for record in records:
            if record.gamma != 1.0 or not p3_target_split_valid(record):
                raise ValueError("Model C v3 requires valid gamma=1 records")
            tables.append(record.support.table)
            position = len(tables) - 1
            for batch in (record.query_shot, record.query_comp):
                for context in batch.table.contexts:
                    table_index.append(position)
                    statics.append(self.static_features(record.support.table, context))
                    counts.append(self._counts(batch, context))
                    weights.append(1.0)
                    is_F.append(context.clock == "F")
            if self.lambda_mask > 0.0 and record.support.table.contexts:
                hidden = sorted(record.support.table.contexts, key=lambda c: c.context_id)[0]
                masked = self._masked(record.support.table, hidden.context_id)
                tables.append(masked)
                table_index.append(len(tables) - 1)
                statics.append(self.static_features(masked, hidden))
                counts.append(self._counts(record.query_shot, hidden))
                weights.append(self.lambda_mask)
                is_F.append(hidden.clock == "F")
        inputs = [self.table_inputs(table) for table in tables]
        return {
            "psi": np.stack([x[0] for x in inputs]),
            "gates": np.stack([x[1] for x in inputs]),
            "table_index": np.asarray(table_index, dtype=int),
            "static": np.vstack(statics),
            "counts": np.vstack(counts),
            "weights": np.asarray(weights, dtype=float),
            "is_F": np.asarray(is_F, dtype=bool),
            "table_count": len(tables),
        }

    def _loss_and_gradient(self, design: dict, params: _Params) -> tuple[float, dict, dict]:
        cache = self.forward(design["psi"], design["gates"], params.a, params.eta)
        static, tix, is_F = design["static"], design["table_index"], design["is_F"]
        logits = self._logits(cache, static, tix, is_F, params)
        logits -= np.max(logits, axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= np.sum(prob, axis=1, keepdims=True)
        weighted = design["weights"][:, None] * design["counts"]
        totals = np.sum(weighted, axis=1)
        W = float(np.sum(totals))
        nll = -float(np.sum(weighted * np.log(np.maximum(prob, 1.0e-15)))) / W
        err = ((totals[:, None] * prob - weighted) / W).reshape(-1, self.K, 2)

        grads = {"A": np.zeros_like(params.A), "betaD": np.zeros_like(params.betaD), "C": np.zeros_like(params.C), "gammaF": np.zeros_like(params.gammaF)}
        dF = np.zeros_like(cache["F"])
        dPw = np.zeros_like(cache["Pw"])
        d, f = ~is_F, is_F
        if np.any(d):
            grads["A"] = np.einsum("nj,nkm,nks->jms", static[d], cache["F"][tix[d]], err[d], optimize=True)
            grads["betaD"] = np.einsum("nj,nks->js", static[d], err[d], optimize=True)
            np.add.at(dF, tix[d], np.einsum("nj,jms,nks->nkm", static[d], params.A, err[d], optimize=True))
        if np.any(f):
            grads["C"] = np.einsum("nj,nkm,nks->jms", static[f], cache["Pw"][tix[f]], err[f], optimize=True)
            grads["gammaF"] = np.einsum("nj,nks->jks", static[f], err[f], optimize=True)
            np.add.at(dPw, tix[f], np.einsum("nj,jms,nks->nkm", static[f], params.C, err[f], optimize=True))
        l2_loss = 0.0
        for name in grads:
            value = getattr(params, name)
            grads[name] += self.l2 * value
            l2_loss += 0.5 * self.l2 * float(np.sum(value * value))

        dirac = float(np.mean(cache["dirac"]))
        graph_grads = {"a": np.zeros_like(params.a), "eta": 0.0}
        for rho in range(len(self.relation_types)):
            tF, tPw, tdirac = self._tangent(cache, design["gates"], params.eta, rho=rho)
            graph_grads["a"][rho] = float(np.sum(dF * tF) + np.sum(dPw * tPw) + self.lambda_dirac * np.mean(tdirac))
        if self.propagate:
            tF, tPw, _ = self._tangent(cache, design["gates"], params.eta, rho=None)
            graph_grads["eta"] = float(np.sum(dF * tF) + np.sum(dPw * tPw))
        loss = nll + self.lambda_dirac * dirac + l2_loss
        return loss, {**grads, **graph_grads}, {"nll": nll, "dirac": dirac, "l2": l2_loss, "weighted_target_shots": W}

    def fit_records(self, records: list[P3ObservationRecord]) -> "P3EquivariantClockDiracEstimator":
        design = self._design(records)
        params = self.params
        names = ("a", "A", "betaD", "C", "gammaF") + (("eta",) if self.propagate else ())
        moments = {n: np.zeros_like(np.asarray(getattr(params, n), dtype=float)) for n in names}
        variances = {n: np.zeros_like(np.asarray(getattr(params, n), dtype=float)) for n in names}
        first_loss = None
        for epoch in range(1, self.epochs + 1):
            loss, grads, _ = self._loss_and_gradient(design, params)
            if not math.isfinite(loss):
                raise FloatingPointError("Model C v3 loss became non-finite")
            if first_loss is None:
                first_loss = loss
            for n in names:
                g = np.asarray(grads[n], dtype=float)
                moments[n] = 0.9 * moments[n] + 0.1 * g
                variances[n] = 0.999 * variances[n] + 0.001 * g * g
                step = self.learning_rate * (moments[n] / (1.0 - 0.9**epoch)) / (np.sqrt(variances[n] / (1.0 - 0.999**epoch)) + 1.0e-8)
                setattr(params, n, np.asarray(getattr(params, n), dtype=float) - step if n != "eta" else float(params.eta - step))
        final_loss, _, breakdown = self._loss_and_gradient(design, params)
        digest = hashlib.sha256(f"{self.model_family}:{self.graph.graph_hash}:{self.init_seed}:{self.propagate}".encode("ascii"))
        for array in (params.a, np.asarray([params.eta]), *params.decoder_arrays()):
            digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
        self.checkpoint_hash = f"{self.model_family}:{digest.hexdigest()[:16]}"
        self.training_summary = {
            "record_count": len(records),
            "target_example_count": int(design["counts"].shape[0]),
            "support_table_count": int(design["table_count"]),
            "first_loss": first_loss,
            "final_loss": final_loss,
            "final_breakdown": breakdown,
            "transport_strength_by_relation": dict(zip(self.relation_types, params.a.tolist(), strict=True)),
            "eta": params.eta,
            "uses_exact_density_or_state_target": False,
            "uses_certificate_selector_atlas_oracle_target": False,
        }
        return self

    # ------------------------------------------------------------------ prediction
    def predict_vectors(self, table: MeasurementTable, contexts: list[MeasurementContext]) -> np.ndarray:
        psi, gates = self.table_inputs(table)
        cache = self.forward(psi[None], gates[None], self.params.a, self.params.eta)
        static = np.vstack([self.static_features(table, c) for c in contexts])
        logits = self._logits(cache, static, np.zeros(len(contexts), dtype=int), np.asarray([c.clock == "F" for c in contexts]), self.params)
        logits -= np.max(logits, axis=1, keepdims=True)
        prob = np.exp(logits)
        return prob / np.sum(prob, axis=1, keepdims=True)

    def predict(self, table: MeasurementTable, context: MeasurementContext) -> dict[str, float]:
        return dict(zip(P3_OUTCOMES, self.predict_vectors(table, [context])[0].tolist(), strict=True))

    def channel_nll(self, record: P3ObservationRecord, channel: str, *, support_table: MeasurementTable | None = None) -> float:
        batch = {"shot": record.query_shot, "comp": record.query_comp, "op": record.query_op}[channel]
        contexts = list(batch.table.contexts)
        prediction = self.predict_vectors(record.support.table if support_table is None else support_table, contexts)
        counts = np.vstack([self._counts(batch, c) for c in contexts])
        return float(-np.sum(counts * np.log(np.maximum(prediction, 1.0e-15))) / np.sum(counts))

    def operator(self, psi: np.ndarray, transports: list[np.ndarray], generators: list[np.ndarray] | None = None) -> np.ndarray:
        active = generators if generators is not None else [relation_generator_v2(e, self.latent_dim) for e in self.graph.edges]
        output = np.zeros_like(psi)
        for index, edge in enumerate(self.graph.edges):
            output[edge.source] += edge.scale * active[index] @ (transports[index] @ psi[edge.target] - psi[edge.source])
        return output

    def parameter_count(self) -> int:
        decoder = sum(int(np.asarray(x).size) for x in self.params.decoder_arrays())
        return decoder + int(self.params.a.size) + (1 if self.propagate else 0)

    def manifest(self) -> dict:
        return {
            "model_family": self.model_family,
            "checkpoint_hash": self.checkpoint_hash,
            "graph_hash": self.graph.graph_hash,
            "graph_control": self.graph.control,
            "capacity_signature": self.graph.capacity_signature(),
            "coordinate_map_hash": self.graph.coordinate_map_hash,
            "init_seed": self.init_seed,
            "init_seed_depends_on_graph": False,
            "latent_dim": self.latent_dim,
            "propagate": self.propagate,
            "parameter_count": self.parameter_count(),
            "connection_parameterization": "relation_tied_lag_gated_diagonal_unitary_phase_v3",
            "finite_generator_representation": "signed_hermitian_unitary_central_difference_v2",
            "readout": "tau_pooled_equivariant_D_and_dft_power_invariant_F_v3",
            "loss": {"lambda_dirac": self.lambda_dirac, "lambda_mask": self.lambda_mask, "lambda_proxy": self.lambda_proxy, "l2": self.l2},
            "claim_boundary": {
                "latent_is_not_physical_state": True,
                "graph_is_not_atlas": True,
                "certificate_objects_optimized": False,
                "exact_state_target_used": False,
            },
            "training": dict(self.training_summary),
        }
