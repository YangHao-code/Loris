"""LORIS analysis utilities.

Post-hoc reporting over experiment artifacts:
  * ``rule_stats``      — predicate/rule statistics over discovered ``rules.json``
                          (counts by type, ML-vs-logic split, body length, roles,
                          representative rules).
  * ``dataset_screen``  — dataset suitability screen (raw-text vs tokenized,
                          label cardinality, sizes) to select usable datasets.

These are read-only aggregators; they do not touch the pipeline.
"""
