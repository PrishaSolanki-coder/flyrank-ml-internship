Capstone Report —
Author: Prisha Solanki
Lane:2 
Repo:https://github.com/PrishaSolanki-coder/flyrank-ml-internship
Date:25-09-2026

0. Abstract
Content teams can't manually review every page in a large inventory, so this project asks which pages are likely to see a real decline in search visibility over the next 30 days, using only signals observable before that window begins. The FlyRank ML Internship starter dataset (30,000 content items, 32 pseudonymized clients) was decomposed by its own trailing-90-day structure into a genuine feature window (the 60 days prior) and a genuine future label window (the most recent 30 days), avoiding same-window leakage entirely. A frozen rule-based baseline, a logistic regression model, and a gradient-boosted tree model were compared on an identical, client-grouped validation split. The gradient-boosted model reached an AUC of 0.680 and beat both the baseline and the base rate at every K tested, while the baseline — built around raw visibility rather than forecasting decline — actually underperformed random selection. The output is a fully ranked queue of all 30,000 pages with an action label (refresh / expand / protect / prune / monitor) and a reason code attached to each, meant as decision support for a content team's next review cycle.

1. Problem framing
The decision this supports is which content pages a review team looks at first, and what to do with each one. The unit of analysis is a single content page (content_id), grouped by client. The output is a ranked list of all pages by predicted risk of near-term decline, with an action label and reason code attached to each row. The action a human takes is to open the top of the queue and refresh, expand, protect, prune, or monitor accordingly — the model doesn't act on anything itself. The cost of a wrong call runs in both directions: reviewing a stable page wastes editor time, while missing a genuinely declining page means the loss compounds before anyone notices it in monthly reporting. With 30,000 pages across 32 clients, no team can eyeball all of them each cycle, which is why a ranked, evidence-based queue is more useful than a flat report.

3. Data safety
Used: data/raw/content_refresh_anonymized.csv — pseudonymized content-level rows with trailing-90-day GSC/GA4 metrics. No client names, domains, URLs, or raw exports appear anywhere in this repository, including work/outputs/.
Deliberately excluded, and why:
trend_direction, trend_pct — the original label-derived fields; never used as features or to define the new label.
impressions_last_30d, clicks_last_30d, sessions_last_30d — these are the label window; used only to compute is_future_decline_label, never as model inputs.
impressions_90d, clicks_90d, sessions_90d, and every other 90d-only aggregate (ctr, avg_position, engagement_rate, scroll_rate, ai_traffic_pct, pageviews_90d, users_90d, engaged_sessions_90d, ai_sessions_90d, scroll_events_90d, days_with_impressions, days_with_sessions) — each of these blends the label window into itself and can't be cleanly split with the columns available, so all are excluded from features even though this shrinks the feature set.
content_id, client_id — used only for joining and for the client-grouped split, never passed to the model.
Confirmed via pl.assert_no_leakage() (raises if any banned column appears in the feature list) and a full label-correlation scan — the maximum absolute correlation between any retained feature and the label is 0.108, consistent with the model's modest (not suspiciously perfect) AUC.

3. Baseline
A transparent, rule-based score using only pre-cutoff signals: impressions_hist60, ctr_hist60, sessions_hist60 (min-max normalized, weighted 0.4 / 0.3 / 0.3), each row also getting a plain-language reason code. It's a fair comparison because it's scored on exactly the same validation rows and the same Precision@K metric as the model, and it was frozen (work/outputs/baseline_action_score.csv) before any model training began.

Its numbers on the validation split (base rate 0.740):

K	10	20	50	100	200
Baseline precision	0.30	0.35	0.48	0.53	0.55

Notably, the baseline underperforms the base rate at every K — a genuine finding, not an error: ranking by raw visibility tends to surface established, stable pages, which are less likely to be the ones declining.

4. Model / analysis
Method: logistic regression as an interpretable check, and HistGradientBoostingClassifier (scikit-learn's native gradient boosting) as the primary model — a good fit for a moderate-size tabular ranking problem with mixed numeric/categorical features and nonlinear interactions.

Feature list: impressions_hist60, clicks_hist60, sessions_hist60, ctr_hist60 (all recomputed from the 60-day history window only), search_volume, cpc, word_count, char_count, content_age_days, age_tier_order, days_since_last_update, plus one-hot-encoded content_type, main_intent, provider_used, model_used, competition_level, age_tier, freshness_tier, word_count_tier, char_count_tier. Left out on purpose: every 90d-only engagement/traffic aggregate and both trend fields (Section 2).

Target, in one sentence: is_future_decline_label = 1 if a page's last-30-day daily impression rate falls more than 20% below its prior-60-day daily impression rate, else 0 — a genuinely forward-looking outcome, not a same-window proxy.

5. Evaluation
Split: grouped by client_id (GroupKFold, 5 folds, fold 1 selected), so no client's pages appear in both sets — 25 clients / 24,269 rows train, 7 clients / 5,731 rows validate. This tests generalization to clients the model has never seen, which matters more than generalizing to new pages from familiar clients. It is not time-aware across multiple calendar cutoffs, since the starter CSV has no report_date to support that — each row instead carries its own internal 60-day-feature / 30-day-label cutoff (Section 2).

Metrics, same split for all three:

K	Baseline	Logistic Regression	Gradient Boosting	Base rate
10	0.30	1.00	1.00	0.740
20	0.35	0.95	0.95	0.740
50	0.48	0.84	0.86	0.740
100	0.53	0.77	0.88	0.740
200	0.55	0.73	0.88	0.740

AUC: logistic regression 0.509 (chance-level — it also failed to converge, ConvergenceWarning), gradient boosting 0.680 (real signal).

Error analysis: logistic regression's near-chance AUC alongside a seemingly perfect Precision@10 is a red flag, not a win — at a 0.740 base rate and only 10–20 samples, a near-random ranker can land a lucky top-K by chance; that number shouldn't be trusted. Gradient boosting's precision stays well above both the baseline and the base rate across every K, which is the more credible result given its AUC.

6. Interpretation
By permutation importance, the top signals for the gradient-boosted model are clicks_hist60, ctr_hist60, char_count, days_since_last_update, and impressions_hist60 — in plain words, a page's recent click behavior and freshness matter more to the model than its raw visibility. That lines up with the correlation scan, where age_tier_order and content_age_days (both negative) and days_since_last_update (positive) are the strongest single-feature relationships with the label — older, longer-untouched pages skew toward future decline, though no individual feature exceeds a 0.108 correlation, so no single signal dominates.

Negative result, stated plainly: the baseline's core assumption — that high visibility signals which pages need attention — doesn't hold for predicting decline specifically; visibility and click-through data are useful to the model, but only when combined, and a visibility-only rule actively misleads (underperforming random). That's a legitimate, well-understood "no effect" for the baseline's original framing, not a shortcoming of the exercise.

7. Recommendation
The full ranked output (work/outputs/final_ranked_recommendations.csv) assigns every one of the 30,000 pages an action: 23,746 monitor, 4,017 refresh, 906 expand, 735 prune, 596 protect. A FlyRank editor would open the file, work top-down through refresh and expand first (these carry the highest predicted decline risk plus either strong current visibility or search demand), spot-check a handful of prune candidates before removing anything, and otherwise leave monitor pages alone until the next cycle.

Confidence, stated explicitly: trust the gradient-boosted model's ranking (AUC 0.680, consistent lift over baseline and base rate) over the logistic regression numbers, which are not reliable at this convergence state. This is decision support, not a guarantee — no individual page's action should be taken without a human glance, especially prune.

8. Reproducibility
This regenerates every file in work/outputs/, including metrics_summary.json, from data/raw/content_refresh_anonymized.csv. The same pipeline is walked through cell by cell in notebooks/lane2_capstone.ipynb.

Random seeds: HistGradientBoostingClassifier(random_state=42) and permutation_importance(..., random_state=42); GroupKFold is deterministic (no seed needed) given the same input row order; LogisticRegression has no explicit seed set — noting this as a gap rather than omitting it.

Environment: requirements.txt — pandas, numpy, scikit-learn, matplotlib.

On "sealed" evaluation: this project does not claim a sealed/blind holdout in the strict sense (that would require the warehouse's daily fact table and a genuine calendar-based holdout month, which the starter CSV doesn't provide). What's here is a client-grouped validation split, fully reproducible and checkable: the exact split logic is in pl.grouped_split() (src/pipeline.py), and its output metrics are committed in work/outputs/metrics_summary.json and precision_at_k_report.csv — nothing about the validation numbers is taken on faith.

9. Acknowledgments & data credit
Built on the FlyRank ML Internship dataset. Credit to FlyRank for the internship program and dataset.
