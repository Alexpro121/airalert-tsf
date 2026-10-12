# RESULTS (generated from reports/metrics.json - do not hand-edit)

data_source=real use_text_signals=False target=Kyiv backend=lightgbm seed=42
prevalence train=0.2172 test=0.5792
threshold_rule: max precision s.t. recall>=0.85 on pooled OOF (guardrail precision>=0.250); DEGENERATE FALLBACK to 0.5
versions: {'sklearn': '1.9.1', 'lightgbm': '4.7.0', 'pandas': '3.0.6', 'python': '3.13.14'}

| model | pr_auc | roc_auc | brier | bss | precision | recall | f2 | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heuristic | 0.5809 | 0.5036 | 0.4208 | — | 0.581 | 0.981 | 0.8621 | 0.5 |
| persistence | 0.6031 | 0.5435 | 0.4917 | — | 0.654 | 0.321 | 0.3577 | 0.5 |
| always | 0.5792 | 0.5000 | 0.4208 | — | 0.579 | 1.000 | 0.8731 | 0.5 |
| trailing | 0.5992 | 0.5169 | 0.3639 | — | 0.654 | 0.321 | 0.3577 | 0.5 |
| climatology | 0.5339 | 0.4391 | 0.3791 | -0.4730 | 0.579 | 1.000 | 0.8731 | 0.16 |
| laghour | 0.5999 | 0.5201 | 0.3499 | -0.3598 | 0.579 | 1.000 | 0.8731 | 0.1 |
| dummy_logistic | 0.5377 | 0.4635 | 0.3784 | -0.4706 | 0.579 | 1.000 | 0.8731 | 0.14 |
| main | 0.6381 | 0.5504 | 0.2575 | -0.0008 | 0.590 | 0.712 | 0.6840 | 0.5 |

paired delta PR-AUC (main-laghour): {'point': 0.0382, 'mean': 0.0381, 'lo': -0.0221, 'hi': 0.0959, 'n_boot': 1000}
paired delta PR-AUC (main-persistence): {'point': 0.035, 'mean': 0.0367, 'lo': -0.0024, 'hi': 0.076, 'n_boot': 1000}
paired delta by split: {'onset': {'laghour': {'point': 0.0712, 'mean': 0.0711, 'lo': -0.0431, 'hi': 0.1636, 'n_boot': 1000}, 'persistence': {'point': 0.0208, 'mean': 0.0256, 'lo': -0.0311, 'hi': 0.0852, 'n_boot': 1000}, 'always': {'point': 0.0208, 'mean': 0.0256, 'lo': -0.0311, 'hi': 0.0852, 'n_boot': 1000}}, 'cont': {'laghour': {'point': 0.1007, 'mean': 0.0975, 'lo': -0.0002, 'hi': 0.1944, 'n_boot': 1000}, 'persistence': {'point': 0.0828, 'mean': 0.0852, 'lo': 0.0249, 'hi': 0.1448, 'n_boot': 1000}, 'always': {'point': 0.0828, 'mean': 0.0852, 'lo': 0.0249, 'hi': 0.1448, 'n_boot': 1000}}}
onset: {'n': 515, 'pos_rate': 0.5495, 'pr_auc': 0.5703112700377575, 'lift': 0.0208, 'f2': 0.5968077723802915}
cont: {'n': 205, 'pos_rate': 0.6537, 'pr_auc': 0.7364364281051317, 'lift': 0.0828, 'f2': 0.8561643835616438}

## Verification (raw tails)
```
tests/test_e2e.py: 10 warnings
  C:\Users\User\Documents\Default Project\airalert-tsf-fixed\.venv\Lib\site-packages\shap\explainers\_tree.py:579: UserWarning: LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
83 passed, 11 warnings in 188.32s (0:03:08)
```
```
All checks passed!
```
