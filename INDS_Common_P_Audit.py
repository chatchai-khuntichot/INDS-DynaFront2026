# ==============================================================================
# INDS: COMMON-METRIC CVR AUDIT
# AUTHOR: S.Wattanasuphinyakit Author(s)
# Purpose
#   Appendix G of the camera-ready says every baseline's CVR is audited "under the
#   exact same learned Lyapunov metric P" and "on the displacement between paired
#   trajectories". The benchmark script did neither:
#     * NODE / PINNs / NODE+SpecNorm have no auditor, so P fell back to identity.
#     * V-dot was computed on single latent states (V = z^T P z), not on pairs.
#   This script audits every model under one common P (the INDS P trained with the
#   same seed) with BOTH definitions, so the two can be compared directly.
#
# Run from the same folder as INDS_Verified_Benchmark_FIXED.py, models/, the CGM
# XML files and the XAUUSD CSV:   python INDS_Common_P_Audit.py
# ==============================================================================

import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from INDS_Verified_Benchmark_fixed import (
    SEEDS, DEVICE, enforce_precision_and_seed, load_comprehensive_datasets,
    train_model_env_aware, SpectralNormNODE, PINNBaseline,
)
from models.inds_core import INDS
from models.base_networks import BaselineNeuralODE

ALPHA = 0.5      # same alpha as the auditor and the original CVR check
TOL = 1e-4       # same tolerance as the original CVR check
STEPS = 10       # same Euler steps / dt as the original CVR check
DT = 0.1
MODELS = ["NODE", "PINNs", "SpectralNormNODE", "INDS"]


def build_model(name):
    # Same constructors as run_fair_evaluation (CGM model only).
    if name == "NODE":
        return BaselineNeuralODE(in_dim=100, latent_dim=10).to(DEVICE)
    if name == "PINNs":
        return PINNBaseline(in_dim=100, latent_dim=10).to(DEVICE)
    if name == "SpectralNormNODE":
        return SpectralNormNODE(in_dim=100, latent_dim=10).to(DEVICE)
    if name == "INDS":
        return INDS(in_dim=100, latent_dim=10, use_stability=True, beta=100.0).to(DEVICE)
    raise ValueError(name)


def train_for_seed(name, seed, data_package):
    """Seed, build and train the CGM model exactly as run_fair_evaluation does."""
    enforce_precision_and_seed(seed)
    X_envs, Y_envs = data_package[0], data_package[1]
    model = build_model(name)
    is_inds = name == "INDS"
    train_model_env_aware(model, X_envs, Y_envs, nn.MSELoss(), is_inds=is_inds, lambda_inv=0.1)
    model.eval()
    return model


@torch.no_grad()
def audit(model, X_ood, metrics, perm):
    """
    Integrate each OOD window with the model's own deployed dynamics (the INDS
    trajectory follows its auditor-safe field, as in the original check) and count
    violations of  V-dot + ALPHA * V > TOL  under every metric in `metrics`.

    single : V(z) = z^T P z,  V-dot = 2 z^T P f(z)                  (what the code did)
    pair   : r = z - z[perm], V = r^T P r, V-dot = 2 r^T P (f(z) - f(z[perm]))
                                                                     (what Appendix G says)
    pre    : raw vector field f_theta         post : field after the auditor projection
    """
    has_auditor = hasattr(model, "auditor") and getattr(model, "use_stability", False)
    n = X_ood.shape[0]
    counts = defaultdict(int)

    z = model.encoder(X_ood)
    for _ in range(STEPS):
        f_raw = model.vf(z)
        f_safe = model.auditor(z, f_raw)[0] if has_auditor else f_raw

        zb = z[perm]
        r = z - zb
        for mname, P in metrics.items():
            zP = z @ P
            V = torch.sum(z * zP, dim=-1)
            counts[(mname, "single", "pre")] += ((2.0 * torch.sum(zP * f_raw, -1) + ALPHA * V) > TOL).sum().item()
            counts[(mname, "single", "post")] += ((2.0 * torch.sum(zP * f_safe, -1) + ALPHA * V) > TOL).sum().item()

            rP = r @ P
            Vr = torch.sum(r * rP, dim=-1)
            d_raw = f_raw - f_raw[perm]
            d_safe = f_safe - f_safe[perm]
            counts[(mname, "pair", "pre")] += ((2.0 * torch.sum(rP * d_raw, -1) + ALPHA * Vr) > TOL).sum().item()
            counts[(mname, "pair", "post")] += ((2.0 * torch.sum(rP * d_safe, -1) + ALPHA * Vr) > TOL).sum().item()

        z = z + f_safe * DT

    return {k: 100.0 * v / (STEPS * n) for k, v in counts.items()}


def main():
    print("Loading data...")
    data_package = load_comprehensive_datasets()
    X_cgm_ood = data_package[2]
    n_ood = X_cgm_ood.shape[0]

    # results[model][(metric, definition, field)] -> list over seeds
    results = {m: defaultdict(list) for m in MODELS}
    kappas = []

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        # Fixed pairing of OOD windows, independent of the global RNG state.
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_ood, generator=gen).to(DEVICE)

        inds = train_for_seed("INDS", seed, data_package)
        P_common = inds.auditor.get_P().detach()
        P_common = 0.5 * (P_common + P_common.T)
        kappas.append(inds.auditor.kappa())
        metrics = {"identity": torch.eye(10, device=DEVICE), "commonP": P_common}

        for name in MODELS:
            model = inds if name == "INDS" else train_for_seed(name, seed, data_package)
            rates = audit(model, X_cgm_ood, metrics, perm)
            for key, val in rates.items():
                results[name][key].append(val)
            print(f"  {name:<17} commonP single pre/post = "
                  f"{rates[('commonP','single','pre')]:7.3f} / {rates[('commonP','single','post')]:7.3f}   "
                  f"commonP pair pre/post = "
                  f"{rates[('commonP','pair','pre')]:7.3f} / {rates[('commonP','pair','post')]:7.3f}")

    def ms(m, key):
        v = results[m][key]
        return f"{np.mean(v):8.4f} ± {np.std(v):7.4f}"

    for mname, title in [("identity", "METRIC = IDENTITY (what the benchmark used for baselines)"),
                         ("commonP", "METRIC = COMMON P (INDS trained P, same seed) for ALL models")]:
        for definition, dtitle in [("single", "single-state  V=z'Pz   (what the code computed)"),
                                   ("pair", "pairwise      r=z1-z2   (what Appendix G describes)")]:
            print("\n" + "=" * 92)
            print(f"{title}\n{dtitle}   [mean ± std over {len(SEEDS)} seeds, %]")
            print("=" * 92)
            print(f"{'Model':<18}{'Pre-Auditor CVR':>26}{'Post-Auditor CVR':>28}")
            for m in MODELS:
                print(f"{m:<18}{ms(m, (mname, definition, 'pre')):>26}{ms(m, (mname, definition, 'post')):>28}")

    print(f"\nkappa(P) of INDS over seeds: {np.mean(kappas):.4f} ± {np.std(kappas):.4f}  (per seed: {[round(k, 4) for k in kappas]})")

    # Sanity check: reproduce the numbers printed by INDS_Verified_Benchmark(_FIXED).py
    print("\n" + "=" * 92)
    print("REPRODUCTION CHECK vs. previously printed benchmark values (tolerance 0.01 percentage points)")
    print("=" * 92)
    reference = [
        ("NODE      identity single pre ", "NODE", ("identity", "single", "pre"), 98.0232),
        ("PINNs     identity single pre ", "PINNs", ("identity", "single", "pre"), 98.7920),
        ("SpecNorm  identity single pre ", "SpectralNormNODE", ("identity", "single", "pre"), 98.3314),
        ("INDS      identity single pre ", "INDS", ("identity", "single", "pre"), 6.6860),
        ("INDS      identity single post", "INDS", ("identity", "single", "post"), 3.2717),
        ("INDS      commonP  single pre ", "INDS", ("commonP", "single", "pre"), 5.7920),
        ("INDS      commonP  single post", "INDS", ("commonP", "single", "post"), 0.0000),
    ]
    for label, m, key, ref in reference:
        got = float(np.mean(results[m][key]))
        flag = "OK   " if abs(got - ref) < 0.01 else "CHECK"
        print(f"  [{flag}] {label}: got {got:8.4f}   reference {ref:8.4f}")

    out = {
        "seeds": SEEDS, "alpha": ALPHA, "tol": TOL, "kappa_P_per_seed": kappas,
        "per_seed_percent": {m: {"|".join(k): v for k, v in d.items()} for m, d in results.items()},
    }
    with open("common_P_audit_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved per-seed results to common_P_audit_results.json")


if __name__ == "__main__":
    main()
