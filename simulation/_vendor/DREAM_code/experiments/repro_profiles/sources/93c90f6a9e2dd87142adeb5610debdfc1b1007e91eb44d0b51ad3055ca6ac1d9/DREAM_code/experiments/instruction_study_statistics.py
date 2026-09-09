"""Prespecified paired-house summary kernel, not an experiment/audit loader.

Only call after binding every outcome to a complete frozen protocol and auditing
successful task records. Synthetic unit-test outputs are not experimental data.
"""
import copy


_PLAN = {
    "schema_version": 1,
    "primary_endpoint": "evaluator_task_success",
    "video_qualification_endpoint": "evaluator_protocol_success",
    "house_count": 10,
    "paired_seeds_per_house": 3,
    "variants_in_order": ["dynamic", "static"],
    "primary_contrast": "dynamic minus static task-completion proportion",
    "resampling_unit": "house, retaining all three paired seeds and both variants",
    "bootstrap_replicates": 20000,
    "bootstrap_seed": 20260908,
    "interval_probability": 0.95,
    "interval_method": "percentile with linear quantiles",
    "missing_outcomes": "not imputed; paired comparison remains unavailable",
    "technical_failures": "retained and reported separately, never silently dropped",
    "successful_outcomes": "require independent physical and primary-record audits",
    "selection_boundary": "Development-selected houses are not a random held-out population; house resampling does not remove selection bias.",
    "interpretation_boundary": "A finite-house descriptive uncertainty interval, not proof of arbitrary-house, outdoor or hardware-system generalization.",
}


def study_analysis_plan():
    """Return a fresh copy for inclusion BEFORE the first formal task starts."""
    return copy.deepcopy(_PLAN)


def summarize_paired_outcomes(rows):
    """Summarize exactly 60 already-validated outcomes; reject missing endpoints.

    Each row supplies scene, seed, variant, task_success (a genuine bool).
    This mathematical kernel does NOT itself certify source provenance, physical
    success or research-release readiness. The caller must perform those checks.
    """
    import numpy as np

    plan=study_analysis_plan();variants=plan["variants_in_order"]
    if len(rows)!=60:raise ValueError("Exactly 60 paired outcomes are required")
    indexed={}
    for row in rows:
        if not isinstance(row["scene"],str) or not row["scene"]:
            raise ValueError("Each outcome needs a house identity")
        if type(row["seed"]) is not int or row["variant"] not in variants:
            raise ValueError("Invalid paired seed or memory variant")
        if type(row["task_success"]) is not bool:
            raise ValueError("Missing/non-boolean outcomes cannot be imputed as failures")
        key=(row["scene"],row["seed"],row["variant"])
        if key in indexed:raise ValueError("Duplicate paired outcome")
        indexed[key]=row["task_success"]
    houses=sorted({row["scene"] for row in rows});seeds=sorted({row["seed"] for row in rows})
    if len(houses)!=plan["house_count"] or len(seeds)!=plan["paired_seeds_per_house"]:
        raise ValueError("Expected ten houses and the same three seeds in each house")
    expected={(house,seed,variant) for house in houses for seed in seeds for variant in variants}
    if set(indexed)!=expected:raise ValueError("Every house/seed needs both memory variants")
    outcomes=np.asarray([[[indexed[house,seed,variant] for variant in variants]
                          for seed in seeds] for house in houses],dtype=bool)
    house_means=outcomes.mean(axis=1)
    rng=np.random.default_rng(plan["bootstrap_seed"])
    selected=rng.integers(0,len(houses),size=(plan["bootstrap_replicates"],len(houses)))
    samples=house_means[selected].mean(axis=1)
    alpha=(1-plan["interval_probability"])/2
    interval=lambda values:(100*np.quantile(values,[alpha,1-alpha],method="linear")).tolist()
    by_variant={variant:dict(successes=int(outcomes[:,:,i].sum()),attempts=30,
        completion_percent=float(100*house_means[:,i].mean()),
        house_bootstrap_interval_percent=interval(samples[:,i]))
        for i,variant in enumerate(variants)}
    return dict(analysis_plan=plan,house_order=houses,seed_order=seeds,
        mathematical_summary_only=True,source_and_physical_validation_performed=False,
        research_release_ready=False,variants=by_variant,
        contrast=dict(difference_percentage_points=float(100*(house_means[:,0]-house_means[:,1]).mean()),
            house_bootstrap_interval_percentage_points=interval(samples[:,0]-samples[:,1])),
        paired_counts={f"dynamic_{int(a)}_static_{int(b)}":int(((outcomes[:,:,0]==a)&(outcomes[:,:,1]==b)).sum())
            for a in (False,True) for b in (False,True)},
        houses=[dict(scene=house,dynamic_successes=int(outcomes[i,:,0].sum()),
            static_successes=int(outcomes[i,:,1].sum()),paired_seeds=3) for i,house in enumerate(houses)])
