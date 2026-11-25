"""Model C architecture v2: clock-coordinate relational Dirac-type free-table estimator.

Development-only estimator. Differences from v1 (``p3_relational_dirac.py``, left untouched so
that the v1 artifact stays reproducible):

1. Node coordinates come from the Stage-A clock map. Node (a, u) reads only the tau-local
   time-basis (``D``) populations at tau(a, u); edges of signed lag h read the lag-h clock
   coherence obtained from the energy-basis (``F``) populations by the discrete Fourier identity
   sum_r p_F(r, s) exp(2 pi i r h / K) = sum_tau <tau| rho_s |tau - h> (finite clock duality).
2. Relation generators are +gamma / -gamma (naive lattice Dirac form); v1's cyclic generators
   coincide for forward and backward hops when r = 2, which made the v1 edge scramble a pure
   initialisation-seed comparison.
3. The random encoder/gate initialisation depends only on (family, latent_dim, init_seed), never
   on the graph hash, so primary and control graphs share common random numbers.
4. The decoder is a multiplicative-interaction (bilinear) readout between gauge-invariant graph
   features and the frozen operator descriptor (Jayakumar et al. 2020); in v1 graph features
   could only add a context-independent logit offset.

The latent field, transports and residuals are estimator-internal and never physical objects.
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


MODEL_FAMILY_V2 = "p3_clock_relational_dirac_free_table_v2"


def null_support_table() -> MeasurementTable:
    return MeasurementTable(probabilities={}, contexts=[], metadata={"role": "p3_support_null", "n_shots": 0}, source_density=None)


def _table_matrix(table: MeasurementTable, context_id: str, K: int) -> np.ndarray | None:
    probabilities = table.probabilities.get(context_id)
    if probabilities is None:
        return None
    return np.asarray([[float(probabilities.get(f"c{tau}_s{bit}", 0.0)) for bit in range(2)] for tau in range(K)], dtype=float)


@dataclass
class _TableState:
    psi: np.ndarray  # (V, r) complex, normalised
    gates: np.ndarray  # (E,)


class P3ClockRelationalDiracEstimator:
    model_family = MODEL_FAMILY_V2
    trained_target_channels = ("shot", "comp")
    report_only_target_channels = ("op",)

    def __init__(
        self,
        input_contexts: list[MeasurementContext],
        graph: ClockCoordinateGraph,
        *,
        latent_dim: int = 2,
        lambda_dirac: float = 0.05,
        lambda_mask: float = 0.25,
        lambda_proxy: float = 0.0,
        l2: float = 1.0e-4,
        learning_rate: float = 0.03,
        epochs: int = 120,
        init_seed: int = 0,
    ):
        if lambda_dirac < 0.0 or lambda_mask < 0.0 or lambda_proxy != 0.0 or l2 < 0.0 or learning_rate <= 0.0 or epochs <= 0:
            raise ValueError("invalid Model C v2 hyperparameters")
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
        self.d_contexts = [context for context in self.input_contexts if context.clock == "D"]
        self.f_contexts = [context for context in self.input_contexts if context.clock == "F"]
        self.node_dim = 3 * len(self.d_contexts) + len(self.f_contexts) + 2
        self.edge_dim = 4 * len(self.f_contexts) + 1
        self.relation_types = graph.relation_types
        self.lags = tuple(sorted({edge.lag for edge in graph.edges}))

        seed = int(hashlib.sha256(f"{self.model_family}:{self.latent_dim}:{self.init_seed}".encode("ascii")).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        shape = (self.latent_dim, self.node_dim)
        self.encoder = (rng.normal(size=shape) + 1.0j * rng.normal(size=shape)) / math.sqrt(2.0 * self.node_dim)
        self.gate_weights = rng.normal(size=(len(self.relation_types), self.edge_dim)) / math.sqrt(self.edge_dim)
        self.phase_basis = np.linspace(-1.0, 1.0, self.latent_dim, dtype=float)

        self.src = np.asarray([edge.source for edge in graph.edges], dtype=int)
        self.tgt = np.asarray([edge.target for edge in graph.edges], dtype=int)
        self.coef = np.asarray([edge.sign * edge.scale for edge in graph.edges], dtype=float)
        type_index = {name: position for position, name in enumerate(self.relation_types)}
        lag_index = {lag: position for position, lag in enumerate(self.lags)}
        self.edge_type = np.asarray([type_index[edge.relation] for edge in graph.edges], dtype=int)
        self.edge_lag = np.asarray([lag_index[edge.lag] for edge in graph.edges], dtype=int)
        self.node_tau = np.asarray(graph.node_tau, dtype=int)
        self.incidence = np.zeros((len(graph.nodes), len(graph.edges)), dtype=float)
        self.incidence[self.src, np.arange(len(graph.edges))] = 1.0
        self.n_edges = len(graph.edges)
        self.n_nodes = len(graph.nodes)
        self.graph_feature_dim = 2 * self.n_edges + self.n_nodes + 1
        self.static_dim = 0
        self.alpha = np.zeros(self.n_edges, dtype=float)
        self.B: np.ndarray | None = None
        self.checkpoint_hash = "unfitted"
        self.training_summary: dict = {}

    # ------------------------------------------------------------------ features
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
        for lag_position, lag in enumerate(self.lags):
            phase = np.exp(2.0j * math.pi * r * lag / self.K)
            column = 0
            for context in self.f_contexts:
                matrix = _table_matrix(table, context.context_id, self.K)
                if matrix is not None:
                    coherence = phase @ matrix  # (2,) complex
                    features[lag_position, column : column + 4] = [
                        coherence[0].real,
                        coherence[0].imag,
                        coherence[1].real,
                        coherence[1].imag,
                    ]
                column += 4
            features[lag_position, column] = 1.0
        return features

    def table_state(self, table: MeasurementTable) -> _TableState:
        node_inputs = self.tau_features(table)[self.node_tau]
        raw = node_inputs @ self.encoder.T
        norms = np.sqrt(np.sum(np.abs(raw) ** 2, axis=1, keepdims=True))
        psi = raw / (norms + 1.0e-12)
        lag = self.lag_features(table)
        gates = np.tanh(np.sum(self.gate_weights[self.edge_type] * lag[self.edge_lag], axis=1))
        return _TableState(psi=psi, gates=gates)

    def connections(self, state: _TableState, alpha: np.ndarray) -> np.ndarray:
        """Diagonal unitary transports U_e = diag(exp(i alpha_e g_e phi)), shape (E, r)."""
        return np.exp(1.0j * alpha[:, None] * state.gates[:, None] * self.phase_basis[None, :])

    def _graph_batch(self, psi: np.ndarray, gates: np.ndarray, alpha: np.ndarray) -> dict:
        """Vectorised graph features for T tables: psi (T, V, r), gates (T, E)."""
        theta = alpha[None, :, None] * gates[:, :, None] * self.phase_basis[None, None, :]
        phase = np.exp(1.0j * theta)
        source = psi[:, self.src, :]
        transported = phase * psi[:, self.tgt, :]
        bilinear = np.sum(np.conj(source) * transported, axis=2)
        d_transported = 1.0j * gates[:, :, None] * self.phase_basis[None, None, :] * transported
        d_bilinear = np.sum(np.conj(source) * d_transported, axis=2)
        hop = self.coef[None, :, None] * (transported - source)
        node_sum = np.einsum("ve,ter->tvr", self.incidence, hop, optimize=True)
        residual = np.sum(np.abs(node_sum) ** 2, axis=2)
        d_residual = 2.0 * np.real(
            np.sum(np.conj(node_sum[:, self.src, :]) * self.coef[None, :, None] * d_transported, axis=2)
        )
        ones = np.ones((psi.shape[0], 1), dtype=float)
        features = np.concatenate([bilinear.real, bilinear.imag, residual, ones], axis=1)
        return {
            "features": features,
            "d_bilinear": d_bilinear,
            "d_residual": d_residual,
            "dirac": np.mean(residual, axis=1),
        }

    def operator(self, psi: np.ndarray, transports: np.ndarray, generators: list[np.ndarray] | None = None) -> np.ndarray:
        """Explicit (D psi)_u = sum_e kappa_e Gamma_e (U_e psi_v - psi_u) with full matrices (for audits)."""
        active = generators if generators is not None else [relation_generator_v2(edge, self.latent_dim) for edge in self.graph.edges]
        output = np.zeros_like(psi)
        for index, edge in enumerate(self.graph.edges):
            output[edge.source] += edge.scale * active[index] @ (transports[index] @ psi[edge.target] - psi[edge.source])
        return output

    def static_features(self, table: MeasurementTable, context: MeasurementContext) -> np.ndarray:
        observed = sum(item.context_id in table.probabilities for item in self.input_contexts) / float(len(self.input_contexts))
        n_shots = float(table.metadata.get("n_shots", 0))
        return np.concatenate([frozen_context_descriptor(context), [observed, math.log1p(n_shots) / 10.0, 1.0]])

    # ------------------------------------------------------------------ training
    @staticmethod
    def _counts(batch: ShotBatch, context: MeasurementContext) -> np.ndarray:
        try:
            return np.asarray([batch.counts[context.context_id][outcome] for outcome in P3_OUTCOMES], dtype=float)
        except KeyError as error:
            raise ValueError(f"missing target count for {context.context_id}") from error

    @staticmethod
    def _masked(table: MeasurementTable, hidden_context_id: str) -> MeasurementTable:
        return MeasurementTable(
            probabilities={key: dict(value) for key, value in table.probabilities.items() if key != hidden_context_id},
            contexts=[context for context in table.contexts if context.context_id != hidden_context_id],
            metadata={**table.metadata, "p3_masked_context": hidden_context_id},
            source_density=None,
        )

    def _design(self, records: list[P3ObservationRecord]) -> dict:
        tables: list[MeasurementTable] = []
        table_index: list[int] = []
        statics: list[np.ndarray] = []
        counts: list[np.ndarray] = []
        weights: list[float] = []
        for record in records:
            if record.gamma != 1.0:
                raise ValueError("Model C coherent-core fitting requires gamma=1 records")
            if not p3_target_split_valid(record):
                raise ValueError("invalid Model C target split")
            tables.append(record.support.table)
            support_position = len(tables) - 1
            for batch in (record.query_shot, record.query_comp):
                for context in batch.table.contexts:
                    table_index.append(support_position)
                    statics.append(self.static_features(record.support.table, context))
                    counts.append(self._counts(batch, context))
                    weights.append(1.0)
            if self.lambda_mask > 0.0 and record.support.table.contexts:
                hidden = sorted(record.support.table.contexts, key=lambda context: context.context_id)[0]
                masked = self._masked(record.support.table, hidden.context_id)
                tables.append(masked)
                table_index.append(len(tables) - 1)
                statics.append(self.static_features(masked, hidden))
                counts.append(self._counts(record.query_shot, hidden))
                weights.append(self.lambda_mask)
        if not counts:
            raise ValueError("Model C requires at least one target")
        states = [self.table_state(table) for table in tables]
        return {
            "psi": np.stack([state.psi for state in states]),
            "gates": np.stack([state.gates for state in states]),
            "table_index": np.asarray(table_index, dtype=int),
            "static": np.vstack(statics),
            "counts": np.vstack(counts),
            "weights": np.asarray(weights, dtype=float),
            "table_count": len(tables),
        }

    def _loss_and_gradient(self, design: dict, B: np.ndarray, alpha: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, dict]:
        graph = self._graph_batch(design["psi"], design["gates"], alpha)
        table_features = graph["features"]  # (T, F)
        static = design["static"]  # (N, J)
        table_index = design["table_index"]
        # logits_n = sum_j static_nj * (g_t(n) . B_j): contract per support table, then per example.
        projected = np.einsum("tf,jfo->tjo", table_features, B, optimize=True)
        logits = np.einsum("nj,njo->no", static, projected[table_index], optimize=True)
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        weighted = design["weights"][:, None] * design["counts"]
        totals = np.sum(weighted, axis=1)
        total_weight = float(np.sum(totals))
        nll = -float(np.sum(weighted * np.log(np.maximum(probabilities, 1.0e-15)))) / total_weight
        error = (totals[:, None] * probabilities - weighted) / total_weight
        static_error = np.zeros((design["table_count"], static.shape[1], error.shape[1]), dtype=float)
        np.add.at(static_error, table_index, static[:, :, None] * error[:, None, :])
        gradient_B = np.einsum("tf,tjo->jfo", table_features, static_error, optimize=True) + self.l2 * B
        d_features = np.einsum("jfo,tjo->tf", B, static_error, optimize=True)
        E, V = self.n_edges, self.n_nodes
        gradient_alpha = np.sum(
            d_features[:, :E] * graph["d_bilinear"].real
            + d_features[:, E : 2 * E] * graph["d_bilinear"].imag
            + d_features[:, 2 * E + self.src] * graph["d_residual"],
            axis=0,
        )
        dirac = float(np.mean(graph["dirac"]))
        gradient_alpha += self.lambda_dirac * np.mean(graph["d_residual"], axis=0) / V
        l2_loss = 0.5 * self.l2 * float(np.sum(B * B))
        loss = nll + self.lambda_dirac * dirac + l2_loss
        return loss, gradient_B, gradient_alpha, {"nll": nll, "dirac": dirac, "l2": l2_loss, "weighted_target_shots": total_weight}

    def fit_records(self, records: list[P3ObservationRecord]) -> "P3ClockRelationalDiracEstimator":
        design = self._design(records)
        self.static_dim = design["static"].shape[1]
        B = np.zeros((self.static_dim, self.graph_feature_dim, len(P3_OUTCOMES)), dtype=float)
        alpha = np.zeros(self.n_edges, dtype=float)
        m_B, v_B = np.zeros_like(B), np.zeros_like(B)
        m_a, v_a = np.zeros_like(alpha), np.zeros_like(alpha)
        first_loss = None
        for epoch in range(1, self.epochs + 1):
            loss, g_B, g_a, _ = self._loss_and_gradient(design, B, alpha)
            if not math.isfinite(loss):
                raise FloatingPointError("Model C v2 loss became non-finite")
            if first_loss is None:
                first_loss = loss
            m_B = 0.9 * m_B + 0.1 * g_B
            v_B = 0.999 * v_B + 0.001 * g_B * g_B
            m_a = 0.9 * m_a + 0.1 * g_a
            v_a = 0.999 * v_a + 0.001 * g_a * g_a
            B -= self.learning_rate * (m_B / (1.0 - 0.9**epoch)) / (np.sqrt(v_B / (1.0 - 0.999**epoch)) + 1.0e-8)
            alpha -= self.learning_rate * (m_a / (1.0 - 0.9**epoch)) / (np.sqrt(v_a / (1.0 - 0.999**epoch)) + 1.0e-8)
        final_loss, _, _, breakdown = self._loss_and_gradient(design, B, alpha)
        self.B, self.alpha = B, alpha
        digest = hashlib.sha256()
        digest.update(f"{self.model_family}:{self.graph.graph_hash}:{self.init_seed}".encode("ascii"))
        digest.update(np.ascontiguousarray(B, dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(alpha, dtype=np.float64).tobytes())
        self.checkpoint_hash = f"{self.model_family}:{digest.hexdigest()[:16]}"
        self.training_summary = {
            "record_count": len(records),
            "target_example_count": int(design["counts"].shape[0]),
            "support_table_count": int(design["table_count"]),
            "first_loss": first_loss,
            "final_loss": final_loss,
            "final_breakdown": breakdown,
            "uses_exact_density_or_state_target": False,
            "uses_certificate_selector_atlas_oracle_target": False,
        }
        return self

    # ------------------------------------------------------------------ prediction
    def predict_vectors(self, table: MeasurementTable, contexts: list[MeasurementContext]) -> np.ndarray:
        if self.B is None:
            raise RuntimeError("Model C v2 estimator is not fitted")
        state = self.table_state(table)
        graph = self._graph_batch(state.psi[None], state.gates[None], self.alpha)
        features = graph["features"][0]
        static = np.vstack([self.static_features(table, context) for context in contexts])
        logits = np.einsum("nj,f,jfo->no", static, features, self.B, optimize=True)
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / np.sum(probabilities, axis=1, keepdims=True)

    def predict(self, table: MeasurementTable, context: MeasurementContext) -> dict[str, float]:
        return dict(zip(P3_OUTCOMES, self.predict_vectors(table, [context])[0].tolist(), strict=True))

    def channel_nll(self, record: P3ObservationRecord, channel: str, *, support_table: MeasurementTable | None = None) -> float:
        batch = {"shot": record.query_shot, "comp": record.query_comp, "op": record.query_op}[channel]
        contexts = list(batch.table.contexts)
        table = record.support.table if support_table is None else support_table
        predictions = self.predict_vectors(table, contexts)
        counts = np.vstack([self._counts(batch, context) for context in contexts])
        return float(-np.sum(counts * np.log(np.maximum(predictions, 1.0e-15))) / np.sum(counts))

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
            "parameter_count": int(self.static_dim * self.graph_feature_dim * len(P3_OUTCOMES) + self.n_edges),
            "connection_parameterization": "lag_coherence_gated_diagonal_unitary_phase_v2",
            "finite_generator_representation": "signed_hermitian_unitary_central_difference_v2",
            "decoder": "descriptor_graph_bilinear_interaction_v2",
            "loss": {
                "lambda_dirac": self.lambda_dirac,
                "lambda_mask": self.lambda_mask,
                "lambda_proxy": self.lambda_proxy,
                "l2": self.l2,
                "trained_target_channels": list(self.trained_target_channels),
                "report_only_target_channels": list(self.report_only_target_channels),
            },
            "claim_boundary": {
                "latent_is_not_physical_state": True,
                "graph_is_not_atlas": True,
                "certificate_objects_optimized": False,
                "exact_state_target_used": False,
                "stage_a_proxy_training_effect_claimed": False,
            },
            "training": dict(self.training_summary),
        }


def gamma_matrix(latent_dim: int) -> np.ndarray:
    return hermitian_unitary_gamma(latent_dim)
