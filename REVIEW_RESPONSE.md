# REVIEW_RESPONSE (DOC-1..DOC-11)

- DOC-1: Прийнято. Ознак 22 (5 календарних + 5 лагів + 3 вікна + hours_since_last + has_history + 4 сусіди + country_ratio + 2 текстові). Таблицю виправлено.
- DOC-2: Виправлено. 0.679 був F2. Тепер Precision і Recall окремо для кожної моделі в metrics.json і README.
- DOC-3: Виправлено. Фолди: TimeSeriesSplit(n_splits=5, gap=3) — 6 блоків по ~16.7%; ембарго враховано.
- DOC-4: Виправлено. Розділ порогу описує guardrail max(0.25, 1.2×prevalence_OOF) і tie-break до вищого порогу.
- DOC-5: Неактуально. Старі відсотки прибрано; нові порівняння — з CI, що показує різниці в межах шуму.
- DOC-6: Знято. Brier гірший за Dummy у старих числах визнано; тепер BSS для всіх; заяв про каліброваність немає.
- DOC-7: Знято. Формулювання «фізика/Hawkes» прибрано: табличний класифікатор на лагах.
- DOC-8: Пом'якшено до «відсутність витоку в межах реалізованих перевірок» з переліком.
- DOC-9: Синхронізовано. README-дерево відповідає факту (text_signals.py, 4 тест-файли, 37 тестів).
- DOC-10: Виправлено. Зберігається summary (стратифікований) + waterfall топ-епізоду; топ-фактори звітуються з фактичного прогону.
- DOC-11: Рефлексію скорочено до ~100 слів і виправлено факти (clip 168 + has_history).
# REVIEW_RESPONSE v2 (HIGH + DOC)
- HIGH-1: file exists; edge tests added (exact-hour end does not mark next hour; >24h works).
- HIGH-2: chosen variant (a): T=start of hour t, known A[..t-1], target A[t..t+2]. True 3h horizon, no dead hour; old Variant B docstring replaced.
- HIGH-3: one convention - naive means Europe/Kyiv in alerts and messages (shared parse_wall_times).
- HIGH-4: ambiguous hour kept as first occurrence (DST side); drops are logged with counts.
- HIGH-5: mixed strings split by offset presence; covered by test.
- HIGH-6: stems with left boundary only; verified inflections match and mid-word rejects.
- HIGH-7: BSS reference is train prevalence (parameter, documented).
- HIGH-8: paired delta CI [0.022, 0.079] on real Kyiv run - superiority over persistence is significant.
- HIGH-9: hash-verified source; pre-hash meta files read as unknown by design.
- DOC-1: prior summary was inconsistent; current: fixed/ changed, original untouched until sync.
- DOC-2: README verified against code in this pass (semantics, 22 features, folds, sources).
- DOC-3: test12.txt replaced by docs/RESULTS.md generated from metrics.json.
# REVIEW_RESPONSE v4 (HIGH + DOC)
- HIGH-1: detect_backend catches (ImportError, OSError) + smoke-fit; OSError-import test added.
- HIGH-2: class_weight removed from HGB; single unweighted rule for both backends.
- HIGH-3: guardrail capped at 0.95; fallback usage logged per model; tune_threshold returns (thr, used_fallback). always-F2 (0.8712) vs main (0.8512) stated directly: trivial rule wins on F2 at base rate 0.575.
- HIGH-4: LagHourBaseline added (PR-AUC 0.583); paired delta main-vs-laghour +0.061 [0.016, 0.100].
- HIGH-5: paired delta CI for onset and cont separately; onset CI contains 0 - superiority on onset NOT proven.
- HIGH-6: both stems kept; ballistika/balistiky tests added.
- HIGH-7: all-clear messages excluded from threat flag + test.
