from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from bi_qmn.core.noise import outcome_key
from bi_qmn.core.observations import ShotBatch
from bi_qmn.p3.p3_records import P3ObservationRecord, p3_target_split_valid
from bi_qmn.core.types import MeasurementContext, MeasurementTable


P3_OUTCOMES = tuple(outcome_key(clock_label, system_bit) for clock_label in range(8) for system_bit in range(2))


@dataclass(frozen=True)
class RelationalEdge:
    source: int
    target: int
    relation: str
    scale: float = 1.0


@dataclass(frozen=True)
class FrozenRelationalGraph:
    nodes: tuple[tuple[int, int], ...]
    edges: tuple[RelationalEdge, ...]
    stage_a_reference: str

    @property
    def graph_hash(self) -> str:
        payload = {
            "nodes": self.nodes,
            "edges": [edge.__dict__ for edge in self.edges],
            "stage_a_reference": self.stage_a_reference,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()[:16]


def build_frozen_local_graph(
    *,
    anchor_count: int,
    local_nodes: tuple[int, ...] | list[int],
    stage_a_reference: str,
) -> FrozenRelationalGraph:
    if anchor_count <= 0 or not stage_a_reference:
        raise ValueError("a frozen Stage-A reference and positive anchor count are required")
    offsets = tuple(int(value) for value in local_nodes)
    if len(offsets) < 2 or tuple(sorted(set(offsets))) != offsets:
        raise ValueError("local graph offsets must be strictly increasing with at least two nodes")
    nodes = tuple((anchor, offset) for anchor in range(anchor_count) for offset in offsets)
    node_index = {node: index for index, node in enumerate(nodes)}
    edges: list[RelationalEdge] = []
    for anchor in range(anchor_count):
        for left, right in zip(offsets, offsets[1:]):
            edges.append(RelationalEdge(node_index[(anchor, left)], node_index[(anchor, right)], "clock_forward"))
            edges.append(RelationalEdge(node_index[(anchor, right)], node_index[(anchor, left)], "clock_backward"))
    return FrozenRelationalGraph(nodes=nodes, edges=tuple(edges), stage_a_reference=stage_a_reference)


def finite_relation_generator(relation: str, latent_dim: int) -> np.ndarray:
    if latent_dim < 2:
        raise ValueError("the finite relational latent dimension must be at least two")
    shift = np.zeros((latent_dim, latent_dim), dtype=complex)
    for index in range(latent_dim):
        shift[index, (index + 1) % latent_dim] = 1.0
    if relation == "clock_forward":
        return shift
    if relation == "clock_backward":
        return shift.conj().T
    raise ValueError(f"unknown finite relation type: {relation}")


def frozen_context_descriptor(context: MeasurementContext) -> np.ndarray:
    symbols = ("I", "X", "Y", "Z")
    values = [1.0 if context.clock == "D" else 0.0, 1.0 if context.clock == "F" else 0.0]
    for symbol in context.system:
        values.extend(1.0 if symbol == candidate else 0.0 for candidate in symbols)
    return np.asarray(values, dtype=float)


def model_c_support_vector(table: MeasurementTable, input_contexts: list[MeasurementContext]) -> np.ndarray:
    n_shots = float(table.metadata.get("n_shots", 0))
    values: list[float] = []
    for context in input_contexts:
        probabilities = table.probabilities.get(context.context_id)
        if probabilities is None:
            values.extend([0.0] * len(P3_OUTCOMES))
            values.extend([0.0, 0.0])
            continue
        values.extend(float(probabilities.get(outcome, 0.0)) for outcome in P3_OUTCOMES)
        values.extend([1.0, math.log1p(n_shots) / 10.0])
    return np.asarray(values, dtype=float)


def _normalized(vector: np.ndarray, epsilon: float = 1.0e-12) -> np.ndarray:
    norm = math.sqrt(float(np.real(np.vdot(vector, vector))))
    return vector / (norm + epsilon)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    values = np.exp(shifted)
    return values / float(np.sum(values))


class FiniteRelationalDiracLayer:
    """Frozen graph encoder with support-conditioned unitary latent transports."""

    def __init__(self, graph: FrozenRelationalGraph, support_dimension: int, latent_dim: int):
        self.graph = graph
        self.support_dimension = int(support_dimension)
        self.latent_dim = int(latent_dim)
        if self.support_dimension <= 0:
            raise ValueError("support dimension must be positive")
        seed = int(hashlib.sha256(f"{graph.graph_hash}:{latent_dim}:v1".encode("ascii")).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        shape = (len(graph.nodes), latent_dim, self.support_dimension)
        self.encoder = (
            rng.normal(size=shape) + 1.0j * rng.normal(size=shape)
        ) / math.sqrt(2.0 * self.support_dimension)
        self.gates = rng.normal(size=(len(graph.edges), self.support_dimension)) / math.sqrt(self.support_dimension)
        self.phase_basis = np.linspace(-1.0, 1.0, latent_dim, dtype=float)
        self.generators = tuple(finite_relation_generator(edge.relation, latent_dim) for edge in graph.edges)

    def encode(self, support_vector: np.ndarray) -> np.ndarray:
        if support_vector.shape != (self.support_dimension,):
            raise ValueError("unexpected Model C support vector shape")
        raw = np.einsum("vrd,d->vr", self.encoder, support_vector, optimize=True)
        return np.vstack([_normalized(row) for row in raw])

    def connections(self, support_vector: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, ...]:
        if alpha.shape != (len(self.graph.edges),):
            raise ValueError("unexpected transport parameter shape")
        gates = np.tanh(self.gates @ support_vector)
        return tuple(
            np.diag(np.exp(1.0j * alpha[index] * gates[index] * self.phase_basis))
            for index in range(len(self.graph.edges))
        )

    def operator(
        self,
        psi: np.ndarray,
        connections: tuple[np.ndarray, ...],
        generators: tuple[np.ndarray, ...] | None = None,
    ) -> np.ndarray:
        if psi.shape != (len(self.graph.nodes), self.latent_dim):
            raise ValueError("unexpected latent field shape")
        active_generators = self.generators if generators is None else generators
        output = np.zeros_like(psi)
        for index, edge in enumerate(self.graph.edges):
            output[edge.source] += edge.scale * active_generators[index] @ (
                connections[index] @ psi[edge.target] - psi[edge.source]
            )
        return output

    def gauge_transform(
        self,
        psi: np.ndarray,
        connections: tuple[np.ndarray, ...],
        gauges: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        if len(gauges) != len(self.graph.nodes):
            raise ValueError("one gauge matrix is required for every graph node")
        transformed_psi = np.vstack([gauges[index] @ psi[index] for index in range(len(gauges))])
        transformed_connections = tuple(
            gauges[edge.source] @ connections[index] @ gauges[edge.target].conj().T
            for index, edge in enumerate(self.graph.edges)
        )
        transformed_generators = tuple(
            gauges[edge.source] @ self.generators[index] @ gauges[edge.source].conj().T
            for index, edge in enumerate(self.graph.edges)
        )
        return transformed_psi, transformed_connections, transformed_generators

    def feature_bundle(
        self,
        support_vector: np.ndarray,
        alpha: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Return invariant graph features, parameter derivatives, and Dirac loss data."""
        psi = self.encode(support_vector)
        connections = self.connections(support_vector, alpha)
        gates = np.tanh(self.gates @ support_vector)
        bilinears: list[float] = []
        derivatives = np.zeros((len(self.graph.edges), 2 * len(self.graph.edges) + len(self.graph.nodes)), dtype=float)
        d_operator = np.zeros(
            (len(self.graph.edges), len(self.graph.nodes), self.latent_dim),
            dtype=complex,
        )
        for index, edge in enumerate(self.graph.edges):
            transported = connections[index] @ psi[edge.target]
            bilinear = np.vdot(psi[edge.source], transported)
            bilinears.extend([float(np.real(bilinear)), float(np.imag(bilinear))])
            derivative_transport = 1.0j * gates[index] * self.phase_basis * transported
            derivative_bilinear = np.vdot(psi[edge.source], derivative_transport)
            derivatives[index, 2 * index] = float(np.real(derivative_bilinear))
            derivatives[index, 2 * index + 1] = float(np.imag(derivative_bilinear))
            d_operator[index, edge.source] = (
                edge.scale * self.generators[index] @ derivative_transport
            )

        operator = self.operator(psi, connections)
        residuals = np.asarray([float(np.real(np.vdot(row, row))) for row in operator], dtype=float)
        for parameter in range(len(self.graph.edges)):
            for node in range(len(self.graph.nodes)):
                derivatives[parameter, 2 * len(self.graph.edges) + node] = 2.0 * float(
                    np.real(np.vdot(operator[node], d_operator[parameter, node]))
                )
        features = np.asarray(bilinears + residuals.tolist(), dtype=float)
        dirac_loss = float(np.mean(residuals))
        dirac_gradient = np.mean(derivatives[:, 2 * len(self.graph.edges) :], axis=1)
        return features, derivatives, dirac_loss, dirac_gradient


@dataclass(frozen=True)
class _TargetExample:
    support_table: MeasurementTable
    context: MeasurementContext
    counts: np.ndarray
    weight: float
    channel: str


class P3RelationalDiracFreeTableEstimator:
    """Count-NLL free-table estimator with a finite relational Dirac-type regularizer."""

    model_family = "p3_relational_dirac_free_table_v1"
    trained_target_channels = ("shot", "comp")
    report_only_target_channels = ("op",)

    def __init__(
        self,
        input_contexts: list[MeasurementContext],
        graph: FrozenRelationalGraph,
        *,
        latent_dim: int = 2,
        lambda_dirac: float = 0.05,
        lambda_mask: float = 0.25,
        lambda_proxy: float = 0.0,
        l2: float = 1.0e-4,
        learning_rate: float = 0.03,
        epochs: int = 120,
    ):
        if lambda_dirac < 0.0 or lambda_mask < 0.0 or lambda_proxy != 0.0 or l2 < 0.0 or learning_rate <= 0.0 or epochs <= 0:
            raise ValueError("invalid Model C training hyperparameters")
        self.input_contexts = list(input_contexts)
        self.graph = graph
        self.lambda_dirac = float(lambda_dirac)
        self.lambda_mask = float(lambda_mask)
        self.lambda_proxy = float(lambda_proxy)
        self.l2 = float(l2)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        support_dimension = len(self.input_contexts) * (len(P3_OUTCOMES) + 2)
        self.layer = FiniteRelationalDiracLayer(graph, support_dimension, latent_dim)
        self.alpha = np.zeros(len(graph.edges), dtype=float)
        self.beta: np.ndarray | None = None
        self.checkpoint_hash = "unfitted"
        self.training_summary: dict = {}

    def _features(self, support_table: MeasurementTable, context: MeasurementContext, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        support_vector = model_c_support_vector(support_table, self.input_contexts)
        graph_features, graph_derivative, dirac_loss, dirac_gradient = self.layer.feature_bundle(support_vector, alpha)
        observed_fraction = sum(
            context.context_id in support_table.probabilities for context in self.input_contexts
        ) / float(len(self.input_contexts))
        n_shots = float(support_table.metadata.get("n_shots", 0))
        static = np.concatenate(
            [
                frozen_context_descriptor(context),
                np.asarray([observed_fraction, math.log1p(n_shots) / 10.0, 1.0], dtype=float),
            ]
        )
        features = np.concatenate([graph_features, static])
        derivative = np.zeros((len(self.graph.edges), features.size), dtype=float)
        derivative[:, : graph_features.size] = graph_derivative
        return features, derivative, dirac_loss, dirac_gradient

    def _examples(self, records: list[P3ObservationRecord]) -> tuple[list[_TargetExample], list[MeasurementTable]]:
        examples: list[_TargetExample] = []
        dirac_supports: list[MeasurementTable] = []
        for record in records:
            if record.gamma != 1.0:
                raise ValueError("Model C coherent-core fitting requires gamma=1 records")
            if not p3_target_split_valid(record):
                raise ValueError("invalid Model C target split")
            dirac_supports.append(record.support.table)
            for channel, batch, weight in (
                ("shot", record.query_shot, 1.0),
                ("comp", record.query_comp, 1.0),
            ):
                for context in batch.table.contexts:
                    examples.append(
                        _TargetExample(
                            support_table=record.support.table,
                            context=context,
                            counts=self._counts(batch, context),
                            weight=weight,
                            channel=channel,
                        )
                    )
            if self.lambda_mask > 0.0 and record.support.table.contexts:
                hidden = sorted(record.support.table.contexts, key=lambda context: context.context_id)[0]
                masked = self._masked_support_table(record.support.table, hidden.context_id)
                examples.append(
                    _TargetExample(
                        support_table=masked,
                        context=hidden,
                        counts=self._counts(record.query_shot, hidden),
                        weight=self.lambda_mask,
                        channel="mask",
                    )
                )
                dirac_supports.append(masked)
        if not examples:
            raise ValueError("Model C requires at least one shot or compositional target")
        return examples, dirac_supports

    @staticmethod
    def _masked_support_table(table: MeasurementTable, hidden_context_id: str) -> MeasurementTable:
        return MeasurementTable(
            probabilities={
                context_id: dict(probabilities)
                for context_id, probabilities in table.probabilities.items()
                if context_id != hidden_context_id
            },
            contexts=[context for context in table.contexts if context.context_id != hidden_context_id],
            metadata={**table.metadata, "p3_masked_context": hidden_context_id},
            source_density=None,
        )

    @staticmethod
    def _counts(batch: ShotBatch, context: MeasurementContext) -> np.ndarray:
        try:
            return np.asarray([batch.counts[context.context_id][outcome] for outcome in P3_OUTCOMES], dtype=float)
        except KeyError as error:
            raise ValueError(f"missing target count for {context.context_id}") from error

    def _loss_and_gradient(
        self,
        examples: list[_TargetExample],
        dirac_supports: list[MeasurementTable],
        beta: np.ndarray,
        alpha: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray, dict]:
        gradient_beta = np.zeros_like(beta)
        gradient_alpha = np.zeros_like(alpha)
        nll = 0.0
        total_weight = 0.0
        for example in examples:
            features, feature_derivative, _, _ = self._features(example.support_table, example.context, alpha)
            logits = features @ beta
            probabilities = _softmax(logits)
            weighted_counts = example.weight * example.counts
            count_total = float(np.sum(weighted_counts))
            error = count_total * probabilities - weighted_counts
            nll -= float(np.sum(weighted_counts * np.log(np.maximum(probabilities, 1.0e-15))))
            gradient_beta += np.outer(features, error)
            gradient_alpha += feature_derivative @ (beta @ error)
            total_weight += count_total

        nll /= total_weight
        gradient_beta /= total_weight
        gradient_alpha /= total_weight

        dirac_loss = 0.0
        dirac_gradient = np.zeros_like(alpha)
        for support in dirac_supports:
            support_vector = model_c_support_vector(support, self.input_contexts)
            _, _, value, gradient = self.layer.feature_bundle(support_vector, alpha)
            dirac_loss += value / len(dirac_supports)
            dirac_gradient += gradient / len(dirac_supports)

        l2_loss = 0.5 * self.l2 * float(np.sum(beta * beta))
        gradient_beta += self.l2 * beta
        loss = nll + self.lambda_dirac * dirac_loss + l2_loss
        gradient_alpha += self.lambda_dirac * dirac_gradient
        return loss, gradient_beta, gradient_alpha, {
            "nll": nll,
            "dirac": dirac_loss,
            "l2": l2_loss,
            "weighted_target_shots": total_weight,
        }

    def fit_records(self, records: list[P3ObservationRecord]) -> "P3RelationalDiracFreeTableEstimator":
        examples, dirac_supports = self._examples(records)
        feature_size = self._features(examples[0].support_table, examples[0].context, self.alpha)[0].size
        beta = np.zeros((feature_size, len(P3_OUTCOMES)), dtype=float)
        alpha = np.zeros(len(self.graph.edges), dtype=float)
        moment_beta = np.zeros_like(beta)
        moment_alpha = np.zeros_like(alpha)
        variance_beta = np.zeros_like(beta)
        variance_alpha = np.zeros_like(alpha)
        first_loss = None
        last_breakdown: dict = {}
        for epoch in range(1, self.epochs + 1):
            loss, gradient_beta, gradient_alpha, breakdown = self._loss_and_gradient(
                examples,
                dirac_supports,
                beta,
                alpha,
            )
            if not math.isfinite(loss):
                raise FloatingPointError("Model C loss became non-finite")
            if first_loss is None:
                first_loss = loss
            moment_beta = 0.9 * moment_beta + 0.1 * gradient_beta
            variance_beta = 0.999 * variance_beta + 0.001 * gradient_beta * gradient_beta
            moment_alpha = 0.9 * moment_alpha + 0.1 * gradient_alpha
            variance_alpha = 0.999 * variance_alpha + 0.001 * gradient_alpha * gradient_alpha
            beta_hat = moment_beta / (1.0 - 0.9**epoch)
            alpha_hat = moment_alpha / (1.0 - 0.9**epoch)
            beta -= self.learning_rate * beta_hat / (np.sqrt(variance_beta / (1.0 - 0.999**epoch)) + 1.0e-8)
            alpha -= self.learning_rate * alpha_hat / (np.sqrt(variance_alpha / (1.0 - 0.999**epoch)) + 1.0e-8)
            last_breakdown = breakdown

        final_loss, _, _, final_breakdown = self._loss_and_gradient(examples, dirac_supports, beta, alpha)
        self.beta = beta
        self.alpha = alpha
        digest = hashlib.sha256()
        digest.update(self.model_family.encode("ascii"))
        digest.update(self.graph.graph_hash.encode("ascii"))
        digest.update(np.ascontiguousarray(self.beta, dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(self.alpha, dtype=np.float64).tobytes())
        self.checkpoint_hash = f"{self.model_family}:{digest.hexdigest()[:16]}"
        self.training_summary = {
            "record_count": len(records),
            "target_example_count": len(examples),
            "trained_target_channels": list(self.trained_target_channels),
            "report_only_target_channels": list(self.report_only_target_channels),
            "first_loss": first_loss,
            "last_preupdate_loss": loss,
            "final_loss": final_loss,
            "final_breakdown": final_breakdown,
            "last_preupdate_breakdown": last_breakdown,
            "uses_exact_density_or_state_target": False,
            "uses_certificate_selector_atlas_oracle_target": False,
        }
        return self

    def predict(self, support_table: MeasurementTable, context: MeasurementContext) -> dict[str, float]:
        if self.beta is None:
            raise RuntimeError("Model C estimator is not fitted")
        features, _, _, _ = self._features(support_table, context, self.alpha)
        probabilities = _softmax(features @ self.beta)
        return dict(zip(P3_OUTCOMES, probabilities.tolist(), strict=True))

    def predict_table(
        self,
        support_table: MeasurementTable,
        contexts: list[MeasurementContext],
        metadata: dict | None = None,
    ) -> MeasurementTable:
        return MeasurementTable(
            probabilities={context.context_id: self.predict(support_table, context) for context in contexts},
            contexts=list(contexts),
            metadata=dict(metadata or support_table.metadata),
            source_density=None,
        )

    def channel_nll(self, record: P3ObservationRecord, channel: str) -> float:
        batches = {
            "shot": record.query_shot,
            "comp": record.query_comp,
            "op": record.query_op,
        }
        if channel not in batches:
            raise ValueError(f"unknown Model C target channel: {channel}")
        batch = batches[channel]
        total = 0.0
        count_total = 0.0
        for context in batch.table.contexts:
            counts = self._counts(batch, context)
            prediction = self.predict(record.support.table, context)
            values = np.asarray([prediction[outcome] for outcome in P3_OUTCOMES], dtype=float)
            total -= float(np.sum(counts * np.log(np.maximum(values, 1.0e-15))))
            count_total += float(np.sum(counts))
        return total / count_total

    def manifest(self) -> dict:
        return {
            "model_family": self.model_family,
            "checkpoint_hash": self.checkpoint_hash,
            "graph_hash": self.graph.graph_hash,
            "stage_a_reference": self.graph.stage_a_reference,
            "latent_dim": self.layer.latent_dim,
            "connection_parameterization": "support_conditioned_diagonal_unitary_phase_v1",
            "finite_generator_representation": "cyclic_relation_generators_v1",
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


def evaluate_p3_target_channels(
    estimator: P3RelationalDiracFreeTableEstimator,
    records: list[P3ObservationRecord],
) -> dict:
    per_record = {channel: [estimator.channel_nll(record, channel) for record in records] for channel in ("shot", "comp", "op")}
    return {
        "record_count": len(records),
        "mean_nll": {channel: float(np.mean(values)) if values else float("nan") for channel, values in per_record.items()},
        "per_record_nll": per_record,
        "trained_target_channels": list(estimator.trained_target_channels),
        "report_only_target_channels": list(estimator.report_only_target_channels),
    }


def evaluate_p3_anti_collapse(
    estimator: P3RelationalDiracFreeTableEstimator,
    records: list[P3ObservationRecord],
) -> dict:
    """Development diagnostics only; none is a certificate or identifiability proof."""
    if not records:
        raise ValueError("anti-collapse diagnostics require records")
    support_nll = [estimator.channel_nll(record, "comp") for record in records]
    null_nll: list[float] = []
    permuted_nll: list[float] = []
    for index, record in enumerate(records):
        null_support = MeasurementTable(
            probabilities={},
            contexts=[],
            metadata={"role": "p3_support_null", "n_shots": 0},
            source_density=None,
        )
        rotated_support = records[(index + 1) % len(records)].support.table
        for support_table, values in ((null_support, null_nll), (rotated_support, permuted_nll)):
            total = 0.0
            count_total = 0.0
            for context in record.query_comp.table.contexts:
                counts = estimator._counts(record.query_comp, context)
                prediction = estimator.predict(support_table, context)
                probabilities = np.asarray([prediction[outcome] for outcome in P3_OUTCOMES], dtype=float)
                total -= float(np.sum(counts * np.log(np.maximum(probabilities, 1.0e-15))))
                count_total += float(np.sum(counts))
            values.append(total / count_total)
    return {
        "channel": "comp",
        "mean_nll": {
            "support_conditioned": float(np.mean(support_nll)),
            "support_null": float(np.mean(null_nll)),
            "support_permuted": float(np.mean(permuted_nll)),
        },
        "deltas": {
            "null_minus_support": float(np.mean(null_nll) - np.mean(support_nll)),
            "permuted_minus_support": float(np.mean(permuted_nll) - np.mean(support_nll)),
        },
        "interpretation": "development retention diagnostic only; no hard mutual-information floor",
    }
