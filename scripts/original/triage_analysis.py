"""
triage_analysis.py -- turns a disease-detection score into an agricultural
decision-support claim.

The premise: the pipeline is not accurate enough to REPLACE an agronomist, but
at a high-sensitivity operating point it can AUTO-CLEAR obviously-healthy leaves
and escalate everything else for human review. The quantity an extension service
actually cares about is therefore not AUC but:

    "What fraction of images does a human never have to look at,
     and what fraction of diseased leaves do we miss in exchange?"

Metrics
  workload_reduction : fraction of ALL images auto-cleared (never seen by human)
  missed_disease     : fraction of diseased leaves wrongly auto-cleared (= 1-sens)
  npv                : of the auto-cleared pile, what fraction really was healthy
  review_burden      : fraction escalated to human review

A screener is USEFUL when workload_reduction is large at an acceptable
missed_disease. Note this works even for a mediocre AUC: a 0.75-AUC model can
still safely clear a meaningful share of images.
"""

import numpy as np


def triage_at_sensitivity(y_true, scores, target_sens):
    """
    y_true : 1 = diseased, 0 = healthy
    scores : higher = more likely diseased
    Pick the threshold that achieves >= target_sens on diseased cases, then
    report what that buys us operationally.
    Auto-clear rule: score < threshold  ->  cleared as healthy, no human review.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    diseased = scores[y_true == 1]
    if len(diseased) == 0:
        raise ValueError("no diseased cases")

    # threshold = the (1-target_sens) quantile of diseased scores, so that
    # target_sens of diseased cases score >= threshold and get escalated.
    thr = np.quantile(diseased, 1.0 - target_sens)

    cleared = scores < thr
    escalated = ~cleared

    n = len(y_true)
    n_dis = int((y_true == 1).sum())

    missed = int(((y_true == 1) & cleared).sum())          # diseased, auto-cleared
    sens_actual = 1.0 - missed / n_dis
    n_cleared = int(cleared.sum())
    npv = float(((y_true == 0) & cleared).sum() / n_cleared) if n_cleared else float('nan')

    return {
        "target_sensitivity": float(target_sens),
        "achieved_sensitivity": float(sens_actual),
        "threshold": float(thr),
        "workload_reduction": float(n_cleared / n),
        "review_burden": float(escalated.sum() / n),
        "missed_disease_rate": float(missed / n_dis),
        "missed_disease_count": missed,
        "npv_of_cleared": npv,
        "n_total": int(n), "n_diseased": n_dis,
    }


def triage_table(y_true, scores, sens_levels=(0.90, 0.95, 0.98, 0.99)):
    return [triage_at_sensitivity(y_true, scores, s) for s in sens_levels]


def print_triage(rows, label=""):
    print(f"\n{'='*78}")
    print(f"TRIAGE / SCREENING ANALYSIS  {label}")
    print(f"{'='*78}")
    print(f"{'target sens':>11s} {'achieved':>9s} {'auto-cleared':>13s} "
          f"{'human review':>13s} {'missed dis.':>12s} {'NPV':>7s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['target_sensitivity']*100:>10.0f}% "
              f"{r['achieved_sensitivity']*100:>8.1f}% "
              f"{r['workload_reduction']*100:>12.1f}% "
              f"{r['review_burden']*100:>12.1f}% "
              f"{r['missed_disease_rate']*100:>11.1f}% "
              f"{r['npv_of_cleared']*100:>6.1f}%")
    print("-" * 78)
    print("Read: at 95% sensitivity we auto-clear X% of images (never reviewed by")
    print("a human) while missing 5% of diseased leaves.")


# ---------------------------------------------------------------------------
# Annotation-cost model
# ---------------------------------------------------------------------------
def annotation_cost(n_images,
                    sec_per_image=8.0,
                    expert_hourly_usd=15.0,
                    label="dataset"):
    """
    Expert labelling cost avoided by a zero-annotation pipeline.
    sec_per_image: time for a trained agronomist to assign a disease class.
                   8s is a conservative mid-estimate for fine-grained leaf
                   pathology (report the range, don't oversell a point value).
    expert_hourly_usd: local agronomist/extension officer rate. Adjust to your
                   region and CITE the source you use.
    """
    hours = n_images * sec_per_image / 3600.0
    cost = hours * expert_hourly_usd
    return {
        "dataset": label,
        "n_images": int(n_images),
        "sec_per_image": sec_per_image,
        "expert_hours": round(hours, 1),
        "expert_days_8h": round(hours / 8.0, 1),
        "cost_usd": round(cost, 0),
        "hourly_rate_usd": expert_hourly_usd,
    }


def print_annotation_cost(rows):
    print(f"\n{'='*78}")
    print("ANNOTATION COST AVOIDED (our pipeline requires ZERO of this)")
    print(f"{'='*78}")
    print(f"{'dataset':<22s} {'images':>8s} {'expert-hrs':>11s} "
          f"{'8h-days':>8s} {'cost USD':>10s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['dataset']:<22s} {r['n_images']:>8,d} {r['expert_hours']:>11.1f} "
              f"{r['expert_days_8h']:>8.1f} {r['cost_usd']:>10,.0f}")
    print("-" * 78)
    print(f"Assumes {rows[0]['sec_per_image']}s/image at "
          f"${rows[0]['hourly_rate_usd']}/hr. Report as a RANGE in the paper "
          f"(e.g. 5-15 s/image) and cite your rate source.")
