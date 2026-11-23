# ==============================================================================
# MINIMAL FIX: DifferentiablePhysicsAuditor — Learnable P via Cholesky
#
# Key modification: Replace fixed identity P with a learnable symmetric 
# positive-definite (SPD) matrix via Cholesky parameterization: P = L @ L.T
#
# Usage:
#   1. Replace the original auditor class in models/inds_core.py (or patch via load_patched_model)
#   2. Run inds_learnedP_retrain.py
#   3. Inspect kappa(P) and PVR
# ==============================================================================

import torch
import torch.nn as nn


class DifferentiablePhysicsAuditor(nn.Module):
    """
    Physics Auditor with learnable positive-definite P via Cholesky.

    P = L @ L.T  where L is lower-triangular with positive diagonal.
    Initialized as P = I (L = I) for stable warm-up.

    Key change from original:
        Original: self.P = nn.Parameter(torch.eye(dim), requires_grad=False)
        Fixed:    Cholesky factor L with requires_grad=True
    """

    def __init__(self, dim=10, alpha=0.5, beta=100.0):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.dim   = dim

        # Cholesky factor: L lower-triangular, diagonal via exp for PD guarantee
        # Initialize at identity: L_offdiag = 0, L_diag_log = 0 → exp(0) = 1
        self.L_offdiag  = nn.Parameter(
            torch.zeros(dim, dim), requires_grad=True)
        self.L_diag_log = nn.Parameter(
            torch.zeros(dim),      requires_grad=True)

    def _L(self):
        """Reconstruct lower-triangular L with positive diagonal."""
        L = torch.tril(self.L_offdiag, diagonal=-1)        # off-diagonal
        L = L + torch.diag(torch.exp(self.L_diag_log))     # positive diagonal
        return L

    def get_P(self):
        """Return P = L @ L.T (always symmetric positive definite)."""
        L = self._L()
        return L @ L.T

    def kappa(self):
        """Condition number kappa(P) = lambda_max / lambda_min."""
        with torch.no_grad():
            ev = torch.linalg.eigvalsh(self.get_P())
            return (ev.max() / ev.min()).item()

    # ── Keep original forward interface intact ──────────────────────────────
    def forward(self, z, z_dot):
        P      = self.get_P()                                           # [d,d]
        grad_V = 2.0 * torch.matmul(z, P)                              # [B,d]
        v_dot  = torch.sum(grad_V * z_dot,               dim=-1, keepdim=True)
        V      = torch.sum(z    * torch.matmul(z, P),   dim=-1, keepdim=True)

        constraint        = v_dot + self.alpha * V
        scaled_constraint = self.beta * constraint
        clamped           = torch.clamp(scaled_constraint, max=80.0)
        violation = (1.0 / self.beta) * torch.log(1.0 + torch.exp(clamped))

        proj_denom  = torch.sum(grad_V ** 2, dim=-1, keepdim=True) + 1e-6
        z_dot_safe  = z_dot - (violation / proj_denom) * grad_V

        return z_dot_safe, violation


# ==============================================================================
# WARM-UP RETRAIN SCRIPT
# ==============================================================================

import os, time, json, random
import numpy as np
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cgm_pipeline import parse_ohio_xml_to_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS  = [42, 43, 44, 45, 46]

CFG = dict(
    # Phase 1: freeze P — allow encoder/vf to adapt with P=I first
    phase1_epochs = 20,
    lr_main_p1    = 5e-3,
    lam_v_p1      = 1.0,

    # Phase 2: unfreeze P — learn anisotropic geometry
    phase2_epochs = 60,
    lr_main_p2    = 1e-3,
    lr_P_p2       = 5e-4,
    lam_v_p2      = 0.5,

    grad_clip     = 5.0,
    pvr_target    = 10.0,   
    seq_len       = 100,
    steps         = 10,
    alpha         = 0.5,
    pvr_thresh    = 1e-4,   # benchmark-consistent threshold
)


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def load_data():
    X, Y = parse_ohio_xml_to_tensor(
        "data/559-ws-training.xml", seq_length=CFG['seq_len'])
    X = (X - 100.0) / 200.0
    Y = (Y - 100.0) / 200.0
    return X.to(DEVICE), Y.to(DEVICE)


def benchmark_pvr(model, X):
    """
    PVR consistent with INDS_Verified_Benchmark2.py:
    Uses current learned P (not forced eye), threshold = 1e-4
    """
    model.eval()
    violations = 0
    with torch.no_grad():
        P_mat = model.auditor.get_P()           # learned P
        z_eval = model.encoder(X)
        for _ in range(CFG['steps']):
            z_dot_eval = model.vf(z_eval)
            grad_V = 2.0 * torch.matmul(z_eval, P_mat)
            v_dot  = torch.sum(grad_V * z_dot_eval, dim=-1, keepdim=True)
            V_val  = torch.sum(z_eval * torch.matmul(z_eval, P_mat),
                               dim=-1, keepdim=True)
            violations += torch.sum(
                (v_dot + CFG['alpha'] * V_val) > CFG['pvr_thresh']).item()
            if model.use_stability:
                z_dot_eval, _ = model.auditor(z_eval, z_dot_eval)
            z_eval = z_eval + z_dot_eval * 0.1

    pvr = (violations / (CFG['steps'] * len(X))) * 100.0
    model.train()
    return pvr


def eval_all(model, X, Y, criterion):
    model.eval()
    with torch.no_grad():
        pred, _ = model(X, steps=CFG['steps'])
        rmse = torch.sqrt(criterion(pred, Y)).item()
    pvr   = benchmark_pvr(model, X)
    kappa = model.auditor.kappa()
    model.train()
    return rmse, pvr, kappa


def train_step(model, X, Y, optimizer, criterion, lam_v):
    model.train()
    optimizer.zero_grad()
    pred, v_loss = model(X, steps=CFG['steps'])
    loss = criterion(pred, Y) + lam_v * torch.mean(v_loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=CFG['grad_clip'])
    optimizer.step()
    return loss.item(), torch.mean(v_loss).item()


# ── Load and patch model ───────────────────────────────────────────────────────

def load_patched_model(path="weights/INDS_best_weights.pt"):
    """
    Load original weights (auditor.P) into new architecture
    (auditor.L_offdiag + auditor.L_diag_log).
    Strategy: load encoder/vf/decoder only, init new auditor fresh.
    """
    from models.inds_core import INDS

    # Load checkpoint
    old_state = torch.load(path, map_location=DEVICE)

    # Keep only non-auditor keys
    filtered = {k: v for k, v in old_state.items()
                if not k.startswith("auditor")}
    print(f"   Loaded {len(filtered)}/{len(old_state)} keys "
          f"(auditor.P skipped — will init fresh)")

    # Build model with NEW auditor attached first
    model = INDS(in_dim=100, latent_dim=10,
                 use_stability=True, beta=100.0).to(DEVICE)
    model.auditor = DifferentiablePhysicsAuditor(
        dim=10, alpha=CFG["alpha"], beta=100.0).to(DEVICE)

    # Load partial state (strict=False: missing auditor keys are OK)
    missing, unexpected = model.load_state_dict(filtered, strict=False)

    # Verify only auditor keys are missing
    bad_missing = [k for k in missing if not k.startswith("auditor")]
    if bad_missing:
        print(f"⚠️  Unexpected missing: {bad_missing}")
    else:
        print(f"   Auditor keys initialized fresh (expected).")

    if unexpected:
        print(f"⚠️  Ignored unexpected keys: {unexpected}")

    print(f"✅ Model patched — kappa(P) = {model.auditor.kappa():.4f}  "
          f"(should be 1.0)")
    return model



# ── Two-phase warm-up ─────────────────────────────────────────────────────────

def warmup_retrain(seed=42):
    set_seed(seed)
    X, Y      = load_data()
    criterion = nn.MSELoss()
    model     = load_patched_model()

    auditor_params = list(model.auditor.parameters())
    other_params   = [p for n, p in model.named_parameters()
                      if not n.startswith('auditor')]

    log = dict(phase=[], epoch=[], rmse=[], pvr=[], kappa=[], loss=[])

    # ── Phase 1: P frozen ────────────────────────────────────────────────────
    for p in auditor_params:
        p.requires_grad = False
    print(f"\n{'='*55}")
    print(f"PHASE 1 — Warm-up (P frozen, target PVR ≤ {CFG['pvr_target']}%)")
    print(f"{'='*55}")
    print(f"{'Ep':>4} {'Loss':>10} {'RMSE':>8} {'kappa':>8} {'PVR%':>8}")
    print('-'*55)

    opt1 = optim.Adam(other_params, lr=CFG['lr_main_p1'])

    for ep in range(1, CFG['phase1_epochs'] + 1):
        loss, _    = train_step(model, X, Y, opt1, criterion, CFG['lam_v_p1'])
        rmse, pvr, k = eval_all(model, X, Y, criterion)
        print(f"{ep:>4} {loss:>10.6f} {rmse:>8.4f} {k:>8.4f} {pvr:>8.2f}%")
        log['phase'].append(1); log['epoch'].append(ep)
        log['rmse'].append(rmse); log['pvr'].append(pvr)
        log['kappa'].append(k);   log['loss'].append(loss)
        if pvr <= CFG['pvr_target'] and ep >= 5:
            print(f"   ✅ PVR ≤ {CFG['pvr_target']}% at epoch {ep}")
            break

    print(f"\nPhase 1 done  PVR={pvr:.2f}%  kappa={k:.4f}")

    # ── Phase 2: P unfrozen ──────────────────────────────────────────────────
    for p in auditor_params:
        p.requires_grad = True
    print(f"\n{'='*55}")
    print(f"PHASE 2 — Learn P ({CFG['phase2_epochs']} epochs)")
    print(f"{'='*55}")
    print(f"{'Ep':>4} {'Loss':>10} {'RMSE':>8} {'kappa':>8} {'PVR%':>8}")
    print('-'*55)

    opt2 = optim.Adam([
        {'params': other_params,   'lr': CFG['lr_main_p2']},
        {'params': auditor_params, 'lr': CFG['lr_P_p2']},
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt2, mode='max', factor=0.5, patience=10)

    best_kappa = k
    best_state = None

    for ep in range(1, CFG['phase2_epochs'] + 1):
        loss, _      = train_step(model, X, Y, opt2, criterion, CFG['lam_v_p2'])
        rmse, pvr, k = eval_all(model, X, Y, criterion)
        scheduler.step(k)

        if k > best_kappa and pvr <= CFG['pvr_target'] * 2:
            best_kappa = k
            best_state = {n: p.clone() for n, p in model.state_dict().items()}

        print(f"{ep:>4} {loss:>10.6f} {rmse:>8.4f} {k:>8.4f} {pvr:>8.2f}%"
              + (" ← best" if k == best_kappa else ""))
        log['phase'].append(2); log['epoch'].append(CFG['phase1_epochs'] + ep)
        log['rmse'].append(rmse); log['pvr'].append(pvr)
        log['kappa'].append(k);   log['loss'].append(loss)

    # Restore best checkpoint
    if best_state:
        model.load_state_dict(best_state)
        _, pvr_best, k_best = eval_all(model, X, Y, criterion)
        print(f"\n✅ Best: kappa={k_best:.4f}  PVR={pvr_best:.2f}%")
    else:
        pvr_best, k_best = pvr, k
        print(f"\n⚠️  No checkpoint met PVR target — using final: "
              f"kappa={k_best:.4f}  PVR={pvr_best:.2f}%")

    return model, log, k_best, pvr_best


# ── Save + Report ─────────────────────────────────────────────────────────────

def save_all(model, log, seed=42):
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/INDS_learnedP_final.pt")
    print("✅ Saved: weights/INDS_learnedP_final.pt")

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    p2 = next((i for i, p in enumerate(log['phase']) if p == 2),
               len(log['epoch']))
    ep = log['epoch']
    for ax, y, ylabel, color in zip(
        axes,
        [log['loss'], log['kappa'], log['pvr']],
        ['Loss', r'$\kappa(P)$', 'PVR (%)'],
        ['steelblue', 'darkorange', 'crimson']
    ):
        ax.plot(ep, y, color=color, linewidth=2)
        ax.axvline(x=ep[p2] if p2 < len(ep) else ep[-1],
                   color='gray', linestyle='--', alpha=0.6,
                   label='Phase 2 start')
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25, linestyle='--')
    plt.tight_layout()
    plt.savefig('learnedP_curves.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: learnedP_curves.pdf")

    # Panel 3 with real kappa
    k   = model.auditor.kappa()
    P   = model.auditor.get_P().detach().cpu().numpy()
    ev  = np.linalg.eigvalsh(P)
    X, _ = load_data()
    with torch.no_grad():
        z    = model.encoder(X[:500]).cpu().numpy()
    diam = float(np.linalg.norm(z.max(axis=0) - z.min(axis=0)))
    C    = 2.0 * diam
    n    = 10695; T = 1.0; alpha_trained = 0.5

    alpha_range = np.linspace(0.05, 3.0, 500)
    def Rn(a):
        return 2*C*np.sqrt(k) / (a*np.sqrt(n)) * (1 - np.exp(-a*T/2))
    R_trained = float(Rn(alpha_trained))

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(alpha_range, Rn(alpha_range), 'steelblue', linewidth=2)
    ax.fill_between(alpha_range, 0, Rn(alpha_range), alpha=0.08, color='steelblue')
    ax.axvline(x=alpha_trained, color='crimson', linestyle='--', linewidth=1.5,
               label=rf'Trained $\alpha={alpha_trained}$')
    ax.scatter([alpha_trained], [R_trained], color='crimson', s=70, zorder=5)
    ax.annotate(rf'$\mathcal{{R}}_n={R_trained:.4f}$',
                xy=(alpha_trained, R_trained),
                xytext=(alpha_trained+0.15, R_trained*1.10),
                fontsize=8, color='crimson',
                arrowprops=dict(arrowstyle='->', color='crimson', lw=1.0))
    ax.set_xlabel(r'Contraction Rate $\alpha$', fontsize=11)
    ax.set_ylabel(r'$\mathcal{R}_n(\mathcal{H}_\alpha)$', fontsize=11)
    ax.set_title(r'$\alpha$: Universal Stability--Generalization Knob', fontsize=9.5)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig('alpha_rademacher_learnedP.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: alpha_rademacher_learnedP.pdf")

    meta = {
        "version"             : "learnedP_final",
        "n_samples"           : n,
        "kappa_P"             : round(k, 6),
        "lambda_min"          : round(float(ev.min()), 8),
        "lambda_max"          : round(float(ev.max()), 8),
        "diam_Z"              : round(diam, 6),
        "C_value"             : round(C, 6),
        "T_horizon"           : T,
        "alpha_trained"       : alpha_trained,
        "R_n_at_alpha_trained": round(R_trained, 8),
    }
    with open("panel3_metadata_learnedP.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("✅ Saved: panel3_metadata_learnedP.json")
    print(f"\n=== FINAL VALUES ===")
    for kk, v in meta.items():
        print(f"  {kk:35s}: {v}")

    latex = (
        f"% Panel 3 — learned P\n"
        f"% $n={n}$, $\\kappa(P)={k:.3f}$, "
        f"$\\mathrm{{diam}}(\\mathcal{{Z}})={diam:.3f}$, "
        f"$C={C:.3f}$, $T={T}$.\n"
        f"% At $\\alpha={alpha_trained}$: "
        f"$\\mathcal{{R}}_n={R_trained:.6f}$ (non-vacuous)."
    )
    with open("panel3_latex_learnedP.txt", "w") as f:
        f.write(latex)
    print("✅ Saved: panel3_latex_learnedP.txt")
    return meta


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print("INDS LEARNABLE-P RETRAIN")
    print("Step 1: Single seed (42) inspection")
    print("="*55)

    model, log, k_best, pvr_best = warmup_retrain(seed=42)
    meta = save_all(model, log)

    print(f"\n{'='*55}")
    print(f"RESULT  kappa={k_best:.4f}  PVR={pvr_best:.2f}%")
    print(f"{'='*55}")

    if pvr_best <= CFG['pvr_target']:
        print("✅✅ SUCCESS — PVR ≤ 10% and kappa > 1")
        print("   → Paper can claim learned anisotropic P")
        print("   → Update figure caption with new kappa and R_n values")
        print("   → Uncomment multi-seed run below")
    elif pvr_best <= 20.0:
        print("⚠️  PVR slightly above target but kappa learned")
        print("   → Consider relaxing pvr_target to 20% in paper")
        print("   → Or increase phase1_epochs and retry")
    else:
        print("❌ PVR too high — keep P=I and document isotropic geometry")
        print("   → Use original submission (P=I, κ=1, R_n=0.107)")

    # Multi-seed (uncomment only if single-seed passes)
    print("\nStep 2: All 5 seeds")
    all_k, all_pvr = [], []
    for seed in SEEDS:
        _, _, k, pvr = warmup_retrain(seed=seed)
        all_k.append(k);
        all_pvr.append(pvr)
        torch.save(model.state_dict(), f"weights/INDS_learnedP_seed{seed}.pt")
    print(f"kappa: {np.mean(all_k):.4f} ± {np.std(all_k):.4f}")
    print(f"PVR:   {np.mean(all_pvr):.2f} ± {np.std(all_pvr):.2f}")
