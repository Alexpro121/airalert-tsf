# Reflection (~100 слів)

What went wrong? The first pipeline tuned its threshold on train predictions and
collapsed to always-alert on hot months; single-class folds crashed it; a distance
feature measured the farthest node, not the nearest.

How did I adjust? Adversarial audits instead of trusting green tests: 3h embargo,
Kyiv-local calendar, strict degenerate guards with a flagged 0.5 fallback, 8 threat
corridors plus path features, sha256 provenance, honest bootstrap intervals.

Why is it better? It is causal and honest about drift (0.22 vs 0.58). LightGBM
reaches PR-AUC 0.630 against 0.600 for the best baseline, plus a working web demo.
