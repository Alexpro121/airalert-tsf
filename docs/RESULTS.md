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
| main | 0.6299 | 0.5452 | 0.2593 | -0.0075 | 0.595 | 0.719 | 0.6906 | 0.5 |

paired delta PR-AUC (main-laghour): {'point': 0.0301, 'mean': 0.0304, 'lo': -0.0335, 'hi': 0.0906, 'n_boot': 1000}
paired delta PR-AUC (main-persistence): {'point': 0.0268, 'mean': 0.029, 'lo': -0.0131, 'hi': 0.0692, 'n_boot': 1000}
paired delta by split: {'onset': {'laghour': {'point': 0.062, 'mean': 0.0627, 'lo': -0.0482, 'hi': 0.1517, 'n_boot': 1000}, 'persistence': {'point': 0.0116, 'mean': 0.0172, 'lo': -0.0364, 'hi': 0.0742, 'n_boot': 1000}, 'always': {'point': 0.0116, 'mean': 0.0172, 'lo': -0.0364, 'hi': 0.0742, 'n_boot': 1000}}, 'cont': {'laghour': {'point': 0.0952, 'mean': 0.0931, 'lo': -0.0006, 'hi': 0.1887, 'n_boot': 1000}, 'persistence': {'point': 0.0773, 'mean': 0.0807, 'lo': 0.0253, 'hi': 0.1413, 'n_boot': 1000}, 'always': {'point': 0.0773, 'mean': 0.0807, 'lo': 0.0253, 'hi': 0.1413, 'n_boot': 1000}}}
onset: {'n': 515, 'pos_rate': 0.5495, 'pr_auc': 0.5611104049223918, 'lift': 0.0116, 'f2': 0.6033287101248266}
cont: {'n': 205, 'pos_rate': 0.6537, 'pr_auc': 0.7309811711134789, 'lift': 0.0773, 'f2': 0.863013698630137}

## Verification (raw tails)
```
tests/test_e2e.py: 10 warnings
  C:\Users\User\Documents\Default Project\airalert-tsf-fixed\.venv\Lib\site-packages\shap\explainers\_tree.py:579: UserWarning: LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
84 passed, 11 warnings in 507.43s (0:08:27)
```
```
All checks passed!
```
