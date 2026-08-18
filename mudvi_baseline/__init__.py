"""Baseline continual-learning pipeline for BCI IV 2a, per Duan et al. (MUDVI).

Only the baseline described in the task is implemented here:
sequential subject training (A01 -> A09) with a class-balanced replay
memory buffer. See IMPLEMENTATION_MAPPING.md (repo root) for the
paper-to-code traceability and explicit list of implementation decisions.

Not implemented (left as modular extension points for future work):
- MMD-based subject-shift detection (Duan et al. Sec 3.3): the subject
  order is known here, so `shift_detection.OracleShiftDetector` is used
  instead of the real detector.
- Gradient projection / confidence-based shift detection (both are the
  user's own proposed additions from mudvi_gp_report.pdf, explicitly out
  of scope for this baseline).
"""
