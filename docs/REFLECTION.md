# Reflection (~100 слів)

What went wrong? Early pipeline tuned the F2 threshold on train predictions and
used a fixed precision guardrail, collapsing to an always-alert rule on hot months.
Target edges were NaN-masked, but train/test had no embargo, and calendar features
used UTC instead of Kyiv time.

How did I adjust? Added a 3h embargo on the split and in folds, Kyiv-local calendar
features, a data-relative guardrail max(0.25, 1.2×prevalence), clipped hours_since_last
with a has_history flag, persistence/climatology/always baselines, and onset metrics.

Why is it better? The pipeline is causal within tested checks, honest about drift
(OOF base 0.22 vs test 0.58, CIs overlap), and LightGBM still leads on onset PR-AUC
(0.622 vs 0.556 continuation) with interpretable SHAP attributions.
