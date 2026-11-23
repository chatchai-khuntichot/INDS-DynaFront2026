# ==============================================================================
# INDS Panel 3 — Extract Real Values for alpha_rademacher.pdf (Autonomous Flow)
# AUTHOR: S.Wattanasuphinyakit Author(s)
# ==============================================================================

import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import os
from models.inds_core import INDS
from cgm_pipeline import parse_ohio_xml_to_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA_RANGE = np.linspace(0.05, 3.0, 500)

# ==============================================================================
# STEP 1 — Load trained model
# ==============================================================================
model = INDS(in_dim=100, latent_dim=10, use_stability=True, beta=100.0).to(DEVICE)
weight_path = "weights/INDS_learnedP_final.pt" if os.path.exists("weights/INDS_learnedP_final.pt") else "weights/INDS_best_weights.pt"
model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
model.eval()
print("✅ Model loaded.")

# ==============================================================================
# STEP 2 — Extract kappa(P)
# ==============================================================================
P_raw = model.auditor.get_P()
P_np = P_raw.detach().cpu().numpy()

eigvals = np.abs(np.linalg.eigvalsh(P_np))
kappa_P = eigvals.max() / eigvals.min()
lambda_min, lambda_max = eigvals.min(), eigvals.max()

print(f"✅ kappa(P)    = {kappa_P:.4f}")

# ==============================================================================
# STEP 3 — Extract n (training sample count)
# ==============================================================================
X_cgm, _ = parse_ohio_xml_to_tensor("data/559-ws-training.xml", seq_length=100)
X_cgm = (X_cgm - 100.0) / 200.0
n_samples = X_cgm.shape[0]
print(f"✅ n (training) = {n_samples}")

# ==============================================================================
# STEP 4 — Extract C = 2 * sqrt(d_z) * diam(Z) (Autonomous Flow - No 'u')
# ==============================================================================
z_all = []
d_z = 10
steps = 10
dt = 0.1

with torch.no_grad():
    chunk = 64
    for i in range(0, n_samples, chunk):
        x_batch = X_cgm[i:i + chunk].to(DEVICE)

        z = model.encoder(x_batch)
        z_seq = []

        for _ in range(steps):
            z_dot = model.vf(z)
            if model.use_stability:
                z_dot, _ = model.auditor(z, z_dot)
            z = z + z_dot * dt
            z_seq.append(z.cpu().numpy())

        z_all.append(np.stack(z_seq, axis=1))

z_all = np.concatenate(z_all, axis=0)  # shape: (N, seq, dz)

# diam(Z) upper-bounded by Euclidean norm of coordinate-wise ranges
z_flat = z_all.reshape(-1, z_all.shape[-1])
z_range = z_flat.max(axis=0) - z_flat.min(axis=0)
diam_Z = float(np.linalg.norm(z_range))

# NEW FORMULA: Exogenous input E[||u||^2] is removed.
C_value = 2.0 * np.sqrt(d_z) * diam_Z

print(f"✅ diam(Z)     = {diam_Z:.4f}")
print(f"✅ C           = {C_value:.4f}")

# ==============================================================================
# STEP 5 — T (horizon length)
# ==============================================================================
T_horizon = steps * dt
print(f"✅ T (horizon)  = {T_horizon:.2f}")

# ==============================================================================
# STEP 6 — Compute Rademacher bound
# ==============================================================================
def rademacher_bound(alpha, n, T, kappa, C):
    return (2.0 * C * np.sqrt(kappa)) / (alpha * np.sqrt(n)) * (1.0 - np.exp(-alpha * T / 2.0))

R_n_curve = rademacher_bound(ALPHA_RANGE, n_samples, T_horizon, kappa_P, C_value)
alpha_trained = 0.5
R_n_trained = rademacher_bound(alpha_trained, n_samples, T_horizon, kappa_P, C_value)

print(f"✅ R_n at alpha={alpha_trained} = {R_n_trained:.6f}")

# ==============================================================================
# STEP 7 & 8 — Plot and Save Metadata
# ==============================================================================
fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.plot(ALPHA_RANGE, R_n_curve, 'b-', linewidth=2.0, label=r'$\mathcal{R}_n(\mathcal{H}_\alpha)$ bound')
ax.axvline(x=alpha_trained, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=rf'Trained $\alpha={alpha_trained}$')
ax.scatter([alpha_trained], [R_n_trained], color='red', s=80, zorder=5)
ax.annotate(rf'$\mathcal{{R}}_n={R_n_trained:.4f}$', xy=(alpha_trained, R_n_trained),
            xytext=(alpha_trained + 0.25, R_n_trained + 0.002), fontsize=8, color='red')
ax.set_xlabel(r'Contraction Rate $\alpha$', fontsize=11)
ax.set_ylabel(r'$\mathcal{R}_n(\mathcal{H}_\alpha)$', fontsize=11)
ax.set_title('Conditional Rademacher Bound vs. Contraction Rate', fontsize=10)
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim([ALPHA_RANGE.min(), ALPHA_RANGE.max()])
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('alpha_rademacher.pdf', dpi=300, bbox_inches='tight')
plt.close()

metadata = {
    "n_samples": int(n_samples), "kappa_P": float(kappa_P), "diam_Z": float(diam_Z),
    "C_value": float(C_value), "T_horizon": float(T_horizon), "alpha_trained": float(alpha_trained),
    "R_n_at_trained_alpha": float(R_n_trained)
}
with open("panel3_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ Saved all outputs successfully!")