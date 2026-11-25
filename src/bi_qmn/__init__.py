"""BI-QMN research package.

Layout (phase first, then role):

- ``core``   phase-independent foundations and the P0 physical layer
- ``ilgc``   indexed local-global certificate engine (SDP-free verification)
- ``stats``  versioned paired-bound statistical engine
- ``p0``     P0 mechanical-correctness harness (SDP era)
- ``p1``     P1 model, Model A/B estimators, SDP-legacy pipeline, ILGC adapters, frozen runs
- ``p2``     P2 typed-path/oracle model and frozen runs
- ``p3``     P3 Model C estimators and development protocols

The flat pre-2026-11-19 layout is archived with an old-to-new path map under
``archive/2026-11-19-flat-layout``.
"""

__all__ = ["core", "ilgc", "stats", "p0", "p1", "p2", "p3"]
