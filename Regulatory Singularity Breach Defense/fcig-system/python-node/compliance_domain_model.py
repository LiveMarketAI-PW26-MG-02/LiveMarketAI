#!/usr/bin/env python3
"""
FCIG Compliance Intelligence Grid
Trains on audit trail consistency, encryption enforcement metrics,
and trade reporting statistical alignment per jurisdiction.
Produces a compliance health probability index.
"""

import numpy as np
from enum import Enum


class Jurisdiction(Enum):
    SEBI  = "Securities and Exchange Board of India"
    SEC   = "U.S. Securities and Exchange Commission"
    FCA   = "UK Financial Conduct Authority"
    ESMA  = "European Securities and Markets Authority"


class ComplianceDomainModel:
    """
    Local compliance model per jurisdiction node.
    Features: [audit_consistency(4), encryption_score(4),
               reporting_alignment(4), latency_compliance(4)]
    """

    INPUT_DIM  = 16
    HIDDEN_DIM = 32

    def __init__(self, node_id: str, jurisdiction: Jurisdiction):
        self.node_id      = node_id
        self.jurisdiction = jurisdiction
        rng = np.random.default_rng(abs(hash(node_id)) % (2**31))
        self.W1   = rng.standard_normal((self.INPUT_DIM, self.HIDDEN_DIM)).astype(np.float32) * 0.1
        self.b1   = np.zeros(self.HIDDEN_DIM, dtype=np.float32)
        self.W2   = rng.standard_normal((self.HIDDEN_DIM, 1)).astype(np.float32) * 0.1
        self.b2   = np.zeros(1, dtype=np.float32)
        self.loss = float("inf")

    @staticmethod
    def extract_compliance_features(audit_record: dict) -> np.ndarray:
        """Extract 16-dim compliance feature vector from audit record."""
        # Audit trail consistency (4 features)
        aud_completeness = float(audit_record.get("completeness", 1.0))
        aud_timestamp_gap = float(audit_record.get("max_timestamp_gap_s", 0))
        aud_hash_valid    = float(audit_record.get("hash_chain_valid", True))
        aud_seq_breaks    = float(audit_record.get("sequence_breaks", 0))

        # Encryption metrics (4 features)
        enc_tls_version   = float(audit_record.get("tls_version", 1.3))
        enc_cipher_score  = float(audit_record.get("cipher_score", 1.0))
        enc_key_rotation  = float(audit_record.get("key_rotation_days", 30) / 365)
        enc_cert_expiry   = float(audit_record.get("cert_days_remaining", 365) / 365)

        # Trade reporting alignment (4 features)
        rpt_completeness  = float(audit_record.get("report_completeness", 1.0))
        rpt_latency_ms    = float(audit_record.get("report_latency_ms", 50))
        rpt_error_rate    = float(audit_record.get("report_error_rate", 0))
        rpt_rejection_rate = float(audit_record.get("rejection_rate", 0))

        # Latency compliance (4 features)
        lat_p50_ms        = float(audit_record.get("latency_p50_ms", 10))
        lat_p99_ms        = float(audit_record.get("latency_p99_ms", 50))
        lat_max_ms        = float(audit_record.get("latency_max_ms", 100))
        lat_sla_breach    = float(audit_record.get("sla_breach_rate", 0))

        return np.array([
            aud_completeness, min(aud_timestamp_gap/60, 1), aud_hash_valid,
            min(aud_seq_breaks/10, 1),
            min(enc_tls_version/1.3, 1), enc_cipher_score, enc_key_rotation,
            enc_cert_expiry,
            rpt_completeness, min(rpt_latency_ms/1000, 1), rpt_error_rate,
            rpt_rejection_rate,
            min(lat_p50_ms/100, 1), min(lat_p99_ms/500, 1), min(lat_max_ms/1000, 1),
            lat_sla_breach,
        ], dtype=np.float32)

    def forward(self, x: np.ndarray) -> float:
        h = np.tanh(x @ self.W1 + self.b1)
        return float(1.0 / (1.0 + np.exp(-(h @ self.W2 + self.b2)[0])))

    def train(self, records: list, lr: float = 1e-3) -> float:
        X = np.array([self.extract_compliance_features(r[0]) for r in records],
                     dtype=np.float32)
        y = np.array([float(r[1]) for r in records]).reshape(-1, 1)
        preds = np.array([[self.forward(x)] for x in X], dtype=np.float32)
        err   = preds - y
        n     = len(records)
        hh    = np.tanh(X @ self.W1 + self.b1)
        dW2   = (hh.T @ err) / n
        db2   = err.mean(axis=0)
        dh    = (err @ self.W2.T) * (1 - hh**2)
        dW1   = (X.T @ dh) / n
        db1   = dh.mean(axis=0)
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.loss = float(np.mean(err**2))
        return self.loss

    def compliance_health_score(self, record: dict) -> float:
        """Returns probability of compliance: 1.0 = fully compliant."""
        feats = self.extract_compliance_features(record)
        return self.forward(feats)


def simulate_compliant_record() -> dict:
    rng = np.random.default_rng()
    return {
        "completeness": 1.0, "max_timestamp_gap_s": rng.uniform(0, 5),
        "hash_chain_valid": True, "sequence_breaks": 0,
        "tls_version": 1.3, "cipher_score": 1.0,
        "key_rotation_days": rng.integers(15, 45),
        "cert_days_remaining": rng.integers(200, 365),
        "report_completeness": 1.0, "report_latency_ms": rng.uniform(10, 80),
        "report_error_rate": 0.0, "rejection_rate": 0.0,
        "latency_p50_ms": rng.uniform(5, 15), "latency_p99_ms": rng.uniform(30, 60),
        "latency_max_ms": rng.uniform(80, 120), "sla_breach_rate": 0.0,
    }


def simulate_violation_record() -> dict:
    rng = np.random.default_rng()
    return {
        "completeness": rng.uniform(0.5, 0.85),
        "max_timestamp_gap_s": rng.uniform(120, 600),
        "hash_chain_valid": False, "sequence_breaks": rng.integers(1, 20),
        "tls_version": 1.0, "cipher_score": rng.uniform(0.3, 0.7),
        "key_rotation_days": rng.integers(200, 365),
        "cert_days_remaining": rng.integers(0, 30),
        "report_completeness": rng.uniform(0.4, 0.8),
        "report_latency_ms": rng.uniform(2000, 10000),
        "report_error_rate": rng.uniform(0.1, 0.5),
        "rejection_rate": rng.uniform(0.05, 0.3),
        "latency_p50_ms": rng.uniform(200, 500),
        "latency_p99_ms": rng.uniform(1000, 5000),
        "latency_max_ms": rng.uniform(5000, 20000),
        "sla_breach_rate": rng.uniform(0.1, 0.5),
    }


if __name__ == "__main__":
    model = ComplianceDomainModel("fcig-sebi-01", Jurisdiction.SEBI)
    records = [(simulate_compliant_record(), True) for _ in range(60)]
    records += [(simulate_violation_record(), False) for _ in range(20)]

    for rnd in range(10):
        loss = model.train(records)
        print(f"Round {rnd+1:2d} | loss={loss:.6f}")

    print(f"\nCompliant health  : {model.compliance_health_score(simulate_compliant_record()):.4f}")
    print(f"Violation health  : {model.compliance_health_score(simulate_violation_record()):.4f}")
