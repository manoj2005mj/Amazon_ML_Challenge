"""Pairwise matcher: features, models and the F0.5 decision rule.

Consumes the candidate files written by ``blocking_strategies.export_candidates``
(``candidates/<split>/parts/<country>_<source>.parquet``) and the normalization
caches written by ``blocking_strategies.harness.dataio``.
"""
