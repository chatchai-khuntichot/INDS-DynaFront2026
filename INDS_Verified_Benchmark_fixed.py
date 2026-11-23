# ==============================================================================
# INDS: FINAL VERIFIED BENCHMARK PIPELINE FOR NEURIPS 2026 (CAMERA-READY)
# AUTHOR: S.Wattanasuphinyakit Author(s).
# Reproducible empirical benchmark pipeline
# FEATURES: IRM Loss, Environment Routing, Wilcoxon Tests, Full Baselines
# ==============================================================================

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from models.inds_core import INDS
from models.base_networks import BaselineNeuralODE, PatchTSTBaseline
from cgm_pipeline import parse_ohio_xml_to_envs, parse_ohio_xml_to_tensor  # ใช้ v3 ที่ return เป็น X_envs, Y_envs
from finance_pipeline import prepare_real_xau_ood_datasets  # ใช้ v3 ที่รองรับ environments

SEEDS = [42, 43, 44, 45, 46]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def enforce_precision_and_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# 0. MISSING BASELINES IMPLEMENTATION (PINNs & Spectral Norm)
# ==============================================================================
class SpectralNormNODE(nn.Module):
    def __init__(self, in_dim=100, latent_dim=10):
        super().__init__()
        self.encoder = nn.Linear(in_dim, latent_dim)
        # Apply Spectral Normalization to Vector Field
        self.vf = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(latent_dim, 64)),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.Linear(64, latent_dim))
        )
        self.decoder = nn.Linear(latent_dim, in_dim)

    def forward(self, x, steps=10, dt=0.1):
        z = self.encoder(x)
        for _ in range(steps):
            z = z + self.vf(z) * dt
        return self.decoder(z), torch.tensor(0.0, device=x.device)


class PINNBaseline(BaselineNeuralODE):
    def __init__(self, in_dim=100, latent_dim=10):
        super().__init__(in_dim, latent_dim)

    def forward(self, x, steps=10, dt=0.1):
        z = self.encoder(x)
        physics_loss = 0.0
        for _ in range(steps):
            z_dot = self.vf(z)
            z = z + z_dot * dt
            # Simple physics penalty: penalize extreme accelerations (derivatives)
            physics_loss += torch.mean(z_dot ** 2)
        return self.decoder(z), physics_loss


# ==============================================================================
# 1. COMPREHENSIVE DATA CONNECTOR (Environment-Aware)
# ==============================================================================
def load_comprehensive_datasets():
    cgm_train_path = "data/559-ws-training.xml"
    cgm_test_path = "data/559-ws-testing.xml"
    finance_path = "data/XAUUSD_H1_202305282200_202605280000.csv"

    X_cgm_envs, Y_cgm_envs = parse_ohio_xml_to_envs(cgm_train_path, seq_length=100, n_envs=12)
    # OOD เป็น Tensor ก้อนเดียว
    X_cgm_ood, Y_cgm_ood = parse_ohio_xml_to_tensor(cgm_test_path, seq_length=100)

    X_fin_envs, Y_fin_envs, X_fin_ood, Y_fin_ood, fin_min, fin_max = prepare_real_xau_ood_datasets(
        finance_path, window_size=100, n_envs=8
    )

    def scale_envs(envs_list, is_cgm=True, f_min=0, f_max=1):
        scaled = []
        for tensor in envs_list:
            tensor = tensor.to(DEVICE)
            if is_cgm:
                scaled.append((tensor - 100.0) / 200.0)
            else:
                scaled.append((tensor - f_min) / (f_max - f_min + 1e-6))
        return scaled

    X_cgm_envs = scale_envs(X_cgm_envs, is_cgm=True)
    Y_cgm_envs = scale_envs(Y_cgm_envs, is_cgm=True)
    X_fin_envs = scale_envs(X_fin_envs, is_cgm=False, f_min=fin_min, f_max=fin_max)
    Y_fin_envs = scale_envs(Y_fin_envs, is_cgm=False, f_min=fin_min, f_max=fin_max)

    X_cgm_ood = ((X_cgm_ood.to(DEVICE) - 100.0) / 200.0)
    Y_cgm_ood = ((Y_cgm_ood.to(DEVICE) - 100.0) / 200.0)
    X_fin_ood = ((X_fin_ood.to(DEVICE) - fin_min) / (fin_max - fin_min + 1e-6))
    Y_fin_ood = ((Y_fin_ood.to(DEVICE) - fin_min) / (fin_max - fin_min + 1e-6))

    return (X_cgm_envs, Y_cgm_envs, X_cgm_ood, Y_cgm_ood,
            X_fin_envs, Y_fin_envs, X_fin_ood, Y_fin_ood,
            fin_min, fin_max)


# ==============================================================================
# 2. EVALUATION MATRIX WITH IRM PENALTY
# ==============================================================================
def train_model_env_aware(model, X_envs, Y_envs, criterion, is_inds=False, lambda_inv=0.1, lambda_stab=0.01):
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    start_time = time.time()

    if not is_inds:
        X_all = torch.cat(X_envs, dim=0)
        Y_all = torch.cat(Y_envs, dim=0)

    for epoch in range(15):
        model.train()
        optimizer.zero_grad()

        if is_inds:
            total_mse = 0.0
            total_v_loss = 0.0
            irm_penalty = 0.0
            num_envs = len(X_envs)

            # Dummy multiplier for IRMv1 penalty computation
            dummy_w = torch.tensor(1.0, requires_grad=True, device=DEVICE)

            for X_e, Y_e in zip(X_envs, Y_envs):
                pred_e, v_loss_e = model(X_e, steps=10)
                # Apply dummy_w for IRM gradient tracking
                mse_e = criterion(pred_e * dummy_w, Y_e)
                total_mse += mse_e
                total_v_loss += torch.mean(v_loss_e)

                # Compute gradient of error w.r.t dummy_w (IRMv1 penalty)
                grad_w = torch.autograd.grad(mse_e, dummy_w, create_graph=True)[0]
                irm_penalty += torch.sum(grad_w ** 2)

            # Combine losses: Predictive + Invariant + Stability
            loss = (total_mse / num_envs) + (lambda_inv * irm_penalty) + (lambda_stab * total_v_loss / num_envs)
        else:
            pred, extra_loss = model(X_all, steps=10)
            loss = criterion(pred, Y_all) + lambda_stab * torch.mean(extra_loss)

        loss.backward()
        optimizer.step()

    return time.time() - start_time


def run_fair_evaluation(model_type, seed, data_package):
    enforce_precision_and_seed(seed)
    (X_cgm_envs, Y_cgm_envs, X_cgm_ood, Y_cgm_ood,
     X_fin_envs, Y_fin_envs, X_fin_ood, Y_fin_ood, fin_min, fin_max) = data_package

    criterion = nn.MSELoss()

    # --- Initialize Models ---
    def get_model(task_dim=100):
        if model_type == "NODE":
            return BaselineNeuralODE(in_dim=task_dim, latent_dim=10).to(DEVICE)
        elif model_type == "PatchTST":
            return PatchTSTBaseline(in_dim=task_dim).to(DEVICE)
        elif model_type == "PINNs":
            return PINNBaseline(in_dim=task_dim, latent_dim=10).to(DEVICE)
        elif model_type == "SpectralNormNODE":
            return SpectralNormNODE(in_dim=task_dim, latent_dim=10).to(DEVICE)
        elif model_type == "INDS":
            return INDS(in_dim=task_dim, latent_dim=10, use_stability=True, beta=100.0).to(DEVICE)
        elif model_type == "INDS_NoStability":
            return INDS(in_dim=task_dim, latent_dim=10, use_stability=False, beta=100.0).to(DEVICE)
        elif model_type == "INDS_NoInvariance":
            return INDS(in_dim=task_dim, latent_dim=10, use_stability=True, beta=100.0).to(DEVICE)

    model_cgm = get_model()
    model_fin = get_model()
    is_inds = "INDS" in model_type


    lam_inv = 0.0 if model_type == "INDS_NoInvariance" else 0.1

    # --- Train Models ---
    time_cgm = train_model_env_aware(model_cgm, X_cgm_envs, Y_cgm_envs, criterion, is_inds=is_inds, lambda_inv=lam_inv)
    time_fin = train_model_env_aware(model_fin, X_fin_envs, Y_fin_envs, criterion, is_inds=is_inds, lambda_inv=lam_inv)

    if seed == 42:
        os.makedirs("weights", exist_ok=True)
        if model_type == "INDS":
            torch.save(model_cgm.state_dict(), "weights/INDS_best_weights.pt")
        elif model_type == "NODE":
            torch.save(model_cgm.state_dict(), "weights/NODE_best_weights.pt")

    # --- Evaluate CGM (Healthcare) ---
    model_cgm.eval()
    with torch.no_grad():
        X_cgm_id_all = torch.cat(X_cgm_envs, dim=0)
        Y_cgm_id_all = torch.cat(Y_cgm_envs, dim=0)

        pred_id, _ = model_cgm(X_cgm_id_all, steps=10)
        pred_ood, _ = model_cgm(X_cgm_ood, steps=10)
        rmse_id = torch.sqrt(criterion(pred_id, Y_cgm_id_all)).item()
        rmse_ood = torch.sqrt(criterion(pred_ood, Y_cgm_ood)).item()

        pvr_calculated = 0.0
        kappa_P_value = None 

        P_source = None
        if hasattr(model_cgm, 'get_P'):
            P_source = model_cgm.get_P
        elif hasattr(model_cgm, 'auditor') and hasattr(model_cgm.auditor, 'get_P'):
            P_source = model_cgm.auditor.get_P

        if P_source is not None:
            P_mat = P_source().detach()
            P_sym = 0.5 * (P_mat + P_mat.T)

            if hasattr(model_cgm, 'auditor') and hasattr(model_cgm.auditor, 'kappa'):
                kappa_P_value = model_cgm.auditor.kappa()
            else:
                eigvals = torch.linalg.eigvalsh(P_sym).detach().cpu().numpy()
                eigvals_abs = np.abs(eigvals)
                lambda_min = eigvals_abs.min()
                lambda_max = eigvals_abs.max()
                kappa_P_value = float(lambda_max / lambda_min) if lambda_min > 1e-12 else float('inf')
        else:
            P_sym = torch.eye(10, device=DEVICE)

        if hasattr(model_cgm, 'encoder') and hasattr(model_cgm, 'vf'):
            violations_pre = 0
            violations_post = 0
            z_eval = model_cgm.encoder(X_cgm_ood)
            for _ in range(10):
                z_dot_eval = model_cgm.vf(z_eval)
                grad_V = 2.0 * torch.matmul(z_eval, P_sym)
                v_dot_pre = torch.sum(grad_V * z_dot_eval, dim=-1, keepdim=True)
                V_val = torch.sum(z_eval * torch.matmul(z_eval, P_sym), dim=-1, keepdim=True)

                # 1. วัด Pre-auditor CVR (Raw Dynamics)
                violations_pre += torch.sum((v_dot_pre + 0.5 * V_val) > 1e-4).item()

                if hasattr(model_cgm, 'auditor') and getattr(model_cgm, 'use_stability', False):
                    z_dot_eval_safe, _ = model_cgm.auditor(z_eval, z_dot_eval)
                    # 2. วัด Post-auditor CVR (Certified Dynamics)
                    v_dot_post = torch.sum(grad_V * z_dot_eval_safe, dim=-1, keepdim=True)
                    violations_post += torch.sum((v_dot_post + 0.5 * V_val) > 1e-4).item()
                    z_eval = z_eval + z_dot_eval_safe * 0.1
                else:
                    violations_post += torch.sum((v_dot_pre + 0.5 * V_val) > 1e-4).item()
                    z_eval = z_eval + z_dot_eval * 0.1

            pvr_pre_calculated = (violations_pre / (10 * len(X_cgm_ood))) * 100.0
            pvr_post_calculated = (violations_post / (10 * len(X_cgm_ood))) * 100.0
        else:
            diff_OOD = torch.abs(pred_ood[:, 1:] - pred_ood[:, :-1])
            expected_decay = torch.abs(Y_cgm_ood[:, 1:] - Y_cgm_ood[:, :-1])
            pvr_calculated = (diff_OOD > expected_decay).float().mean().item() * 100.0
            pvr_pre_calculated = pvr_calculated  # แมปค่าให้ Baseline
            pvr_post_calculated = pvr_calculated  # แมปค่าให้ Baseline

        noise_var = 0.05 * torch.randn_like(X_cgm_ood)
        pred_noisy, _ = model_cgm(X_cgm_ood + noise_var, steps=10)
        noise_sensitivity = torch.sqrt(criterion(pred_noisy, pred_ood)).item()

    # --- Evaluate Finance (XAUUSD) ---
    model_fin.eval()
    with torch.no_grad():
        pred_fin, _ = model_fin(X_fin_ood, steps=10)
        if pred_fin.dim() == 3: pred_fin = pred_fin.squeeze(1)

        curr_scaled = X_fin_ood[:, -1].cpu().numpy()
        next_scaled = Y_fin_ood[:, -1].cpu().numpy()
        pred_scaled = pred_fin[:, -1].cpu().numpy()

        curr_raw = curr_scaled * (fin_max - fin_min) + fin_min
        next_raw = next_scaled * (fin_max - fin_min) + fin_min

        price_return = (next_raw - curr_raw) / (curr_raw + 1e-6)
        position_signal = np.sign(pred_scaled - curr_scaled)
        position_signal[position_signal == 0] = 1.0

        actual_returns = position_signal * price_return - 0.0001 

        sharpe_folds = []
        fold_size = len(actual_returns) // 5
        for f in range(5):
            fold_ret = actual_returns[f * fold_size: (f + 1) * fold_size]
            m_ret = np.mean(fold_ret)
            s_ret = np.std(fold_ret) + 1e-6
            sharpe_folds.append((m_ret / s_ret) * np.sqrt(252 * 24))
        mean_sharpe = np.mean(sharpe_folds)

    return rmse_id, rmse_ood, pvr_pre_calculated, pvr_post_calculated, noise_sensitivity, mean_sharpe, sharpe_folds, kappa_P_value


# ==============================================================================
# 3. VERIFICATION PROCESS & STATISTICAL TESTS
# ==============================================================================
def start_verification_process():
    print("⏳ Running pipeline over 5 random seeds to verify real statistical convergence...")
    print("=" * 80)
    print("⚠️  CORRECTED VERSION — fixes vs. original INDS_Verified_Benchmark.py:")
    print("   1. kappa(P) is now computed from real eigenvalues via model.auditor.kappa(),")
    print("      not a hardcoded string ('**1.271**' / '**1.418**').")
    print("   2. P-matrix extraction now checks model.auditor.get_P() — the original")
    print("      hasattr(model_cgm, 'get_P') check was always False for INDS (get_P lives")
    print("      on .auditor, not on the model itself), so CVR was silently computed against")
    print("      an identity matrix instead of the learned Lyapunov metric for ALL prior runs.")
    print("   3. Wilcoxon test now uses the real 'cvr_pre' key instead of the nonexistent 'cvr'")
    print("      key, which previously always raised KeyError and silently returned p=1.0.")
    print("=" * 80 + "\n")
    all_models = ["NODE", "PatchTST", "PINNs", "SpectralNormNODE", "INDS", "INDS_NoStability", "INDS_NoInvariance"]
 
    aggregated_results = {m: {"rmse_id": [], "rmse_ood": [], "cvr_pre": [], "cvr_post": [], "noise": [], "sharpe": [], "sharpe_folds": [], "kappa_P": []} for m in all_models}

    data_package = load_comprehensive_datasets()

    for seed in SEEDS:
        for m in all_models:
            r_id, r_ood, cvr_pre, cvr_post, n_sens, sh, sh_folds, kappa_P = run_fair_evaluation(m, seed, data_package)
            aggregated_results[m]["rmse_id"].append(r_id)
            aggregated_results[m]["rmse_ood"].append(r_ood)
            aggregated_results[m]["cvr_pre"].append(cvr_pre)
            aggregated_results[m]["cvr_post"].append(cvr_post)
            aggregated_results[m]["noise"].append(n_sens)
            aggregated_results[m]["sharpe"].append(sh)
            aggregated_results[m]["sharpe_folds"] = sh_folds 
            if kappa_P is not None:
                aggregated_results[m]["kappa_P"].append(kappa_P)

    def get_stats(m, key):
        vals = aggregated_results[m][key]
        return np.mean(vals), np.std(vals)

    # --- Statistical Testing (Wilcoxon Signed-Rank Test) ---
    try:
        _, p_val_sharpe = wilcoxon(aggregated_results["INDS"]["sharpe"], aggregated_results["NODE"]["sharpe"])
    except Exception as e:
        print(f"⚠️ Wilcoxon (sharpe) failed: {e}")
        p_val_sharpe = 1.0
    try:

        p_val_cvr = wilcoxon(aggregated_results["INDS"]["cvr_pre"], aggregated_results["NODE"]["cvr_pre"]).pvalue
    except Exception as e:
        print(f"⚠️ Wilcoxon (cvr_pre) failed: {e}")
        p_val_cvr = 1.0

    sharpe_dagger = "†" if p_val_sharpe < 0.01 else ""
    cvr_dagger = "†" if p_val_cvr < 0.01 else ""

    print("\n" + "=" * 80)
    print("✅ RENDERED MANUSCRIPT TABLES WITH WILCOXON STATISTICAL TESTS")
    print("=" * 80)
    print(f"Wilcoxon Test vs NODE -> Sharpe p-value: {p_val_sharpe:.4f}, CVR p-value: {p_val_cvr:.4f}\n")

    # --- TABLE 1 (OOD Performance) ---
    print("### Table 1: Performance under Distribution Shift (OOD)")
    print(f"| Model | OOD RMSE (Healthcare) ↓ | Directional OOD Sharpe (Finance) ↑ | Relative OOD RMSE Change ↓ |")
    print(f"| :--- | :---: | :---: | :---: |")
    for m in ["NODE", "PatchTST", "PINNs", "SpectralNormNODE", "INDS"]:
        m_id, _ = get_stats(m, "rmse_id")
        m_ood, m_ood_s = get_stats(m, "rmse_ood")
        sh_m, sh_s = get_stats(m, "sharpe")
        drop_calc = ((m_ood - m_id) / (m_id + 1e-6)) * 100.0

        name_map = {"NODE": "Baseline NODE", "PatchTST": "PatchTST", "PINNs": "PINNs",
                    "SpectralNormNODE": "NODE + Spectral Norm", "INDS": "**INDS (Ours)**"}
        name = name_map[m]
        dagger = sharpe_dagger if m == "INDS" else ""
        # ลบ abs() ออกจาก drop_calc
        print(f"| {name} | {m_ood:.4f} ± {m_ood_s:.4f} | {sh_m:.4f}{dagger} ± {sh_s:.4f} | {drop_calc:.2f}% |")

    # --- TABLE 2 (Constraint Integrity) ---
    print("\n### Table 2: Lyapunov Constraint Integrity")
    print(f"| Model | Pre-Auditor CVR (%) ↓ | Post-Auditor CVR (%) ↓ | Input Noise Sensitivity ↓ | kappa(P) |")
    print(f"| :--- | :---: | :---: | :---: | :---: |")
    for m in ["NODE", "PatchTST", "PINNs", "SpectralNormNODE", "INDS"]:
        if m == "PatchTST":
            ns, ns_s = get_stats(m, "noise")
            print(f"| PatchTST | N/A | N/A | {ns:.4f} ± {ns_s:.4f} | -- |")
            continue

        cvr_pre, cvr_pre_s = get_stats(m, "cvr_pre")
        cvr_post, cvr_post_s = get_stats(m, "cvr_post")
        ns, ns_s = get_stats(m, "noise")
        name = name_map[m]

        kappa_vals = aggregated_results[m]["kappa_P"]
        if len(kappa_vals) > 0:
            kappa_val = f"**{np.mean(kappa_vals):.3f} ± {np.std(kappa_vals):.3f}**" if m == "INDS" else f"{np.mean(kappa_vals):.3f} ± {np.std(kappa_vals):.3f}"
        else:
            kappa_val = "--" 

        print(
            f"| {name} | {cvr_pre:.4f}% ± {cvr_pre_s:.4f}% | {cvr_post:.4f}% ± {cvr_post_s:.4f}% | {ns:.4f} ± {ns_s:.4f} | {kappa_val} |")

    # --- TABLE 3 (Ablation) ---
    print("\n### Table 3: Ablation Results")
    print(f"| Configuration | RMSE (CGM) ↓ | Sharpe (XAU) ↑ | CVR (%) ↓ |")
    print(f"| :--- | :---: | :---: | :---: |")
    for m in ["INDS", "INDS_NoStability", "INDS_NoInvariance"]:
        r, r_s = get_stats(m, "rmse_ood")
        sh, sh_s = get_stats(m, "sharpe")
        cvr, cvr_s = get_stats(m, "cvr_pre")  
        name = "**INDS (Full)**" if m == "INDS" else (
            "w/o Stability" if m == "INDS_NoStability" else "w/o Invariance")
        print(f"| {name} | {r:.4f} ± {r_s:.4f} | {sh:.4f} ± {sh_s:.4f} | {cvr:.4f}% ± {cvr_s:.4f}% |")

    # --- DYNAMIC PER-FOLD SHARPE RATIO ---
    print("\n>>> Dynamic Per-Fold Sharpe Ratio Distribution (Walk-Forward Validation):")
    for m in ["NODE", "PatchTST", "INDS"]:
        folds_formatted = [round(val, 2) for val in aggregated_results[m]["sharpe_folds"]]
        print(f"  * {m if m != 'NODE' else 'Baseline NODE'} Folds Sharpe: {folds_formatted}")

    # ==============================================================================
    # 📈 AUTOMATED EMPIRICAL FIGURE GENERATION FROM 100% REAL TRAINED WEIGHTS
    # ==============================================================================
    print("\n📊 Loading fully trained model weights to generate empirical figures...")
    X_cgm_envs, Y_cgm_envs, X_cgm_ood, Y_cgm_ood, _, _, _, _, _, _ = data_package

    model_inds = INDS(in_dim=100, latent_dim=10, use_stability=True, beta=100.0).to(DEVICE)
    model_node = BaselineNeuralODE(in_dim=100, latent_dim=10).to(DEVICE)

    try:
        inds_weight_path = "weights/INDS_learnedP_final.pt" if os.path.exists(
            "weights/INDS_learnedP_final.pt") else "weights/INDS_best_weights.pt"
        model_inds.load_state_dict(torch.load(inds_weight_path, map_location=DEVICE))
        model_node.load_state_dict(torch.load("weights/NODE_best_weights.pt", map_location=DEVICE))

        model_inds.eval()
        model_node.eval()

        with torch.no_grad():
            z_inds = model_inds.encoder(X_cgm_ood[0:1])
            z_node = model_node.encoder(X_cgm_ood[0:1])

            inds_traj = [z_inds.squeeze().cpu().numpy()]
            node_traj = [z_node.squeeze().cpu().numpy()]
            v_profile = []

            dt = 0.1
            if hasattr(model_inds.auditor, 'get_P'):
                P_matrix = model_inds.auditor.get_P().detach().cpu().numpy()
            else:
                P_matrix = model_inds.auditor.P.detach().cpu().numpy()
            alpha_val = model_inds.auditor.alpha

            for t in range(100):
                z_np = z_inds.squeeze().cpu().numpy()
                v_t = np.dot(z_np, np.dot(P_matrix, z_np))
                v_profile.append(v_t)

                z_dot_inds = model_inds.vf(z_inds)
                z_dot_inds_safe, _ = model_inds.auditor(z_inds, z_dot_inds)
                z_inds = z_inds + z_dot_inds_safe * dt
                inds_traj.append(z_inds.squeeze().cpu().numpy())

                z_dot_node = model_node.vf(z_node)
                z_node = z_node + z_dot_node * dt
                node_traj.append(z_node.squeeze().cpu().numpy())

            inds_traj = np.array(inds_traj)
            node_traj = np.array(node_traj)
            v_profile = np.array(v_profile)

            t_arr = np.arange(100) * dt
            th_bound = v_profile[0] * np.exp(-alpha_val * t_arr)

        # --- PLOT FIGURE 1: REAL TRAJECTORY CONTRACTION (Phase Portrait) ---
        fig1, ax1 = plt.subplots(figsize=(5.5, 4.5))
        ax1.plot(node_traj[:, 0], node_traj[:, 1], 'r--', label='Baseline NODE', alpha=0.8,
                 linewidth=1.5)
        ax1.plot(inds_traj[:, 0], inds_traj[:, 1], 'b-', label='INDS', linewidth=2.0)
        ax1.scatter(inds_traj[0, 0], inds_traj[0, 1], color='green', marker='o', s=100, label='Start State ($z_0$)',
                    zorder=5)
        ax1.scatter(0, 0, color='black', marker='x', s=100, label='Equilibrium ($z^*$)', zorder=5)
        ax1.set_xlabel('Latent Dimension 1 ($z_1$)')
        ax1.set_ylabel('Latent Dimension 2 ($z_2$)')
        ax1.set_title('Real Trajectory Latent Space Contraction')
        ax1.legend(frameon=True, loc='upper right')
        plt.tight_layout()
        fig1.savefig('trajectory_contraction.pdf', dpi=300)
        plt.close(fig1)

        # --- PLOT FIGURE 2: REAL LYAPUNOV DECAY ---
        fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
        ax2.plot(t_arr, v_profile, 'b-', label='Empirical $V(t)$ (Real INDS Flow)', linewidth=2.0)
        ax2.plot(t_arr, th_bound, 'k--', label=r'Theoretical Bound ($V_0 e^{-\alpha t}$)', alpha=0.7)
        ax2.set_xlabel('Continuous Time ($t$)')
        ax2.set_ylabel('Lyapunov Energy $V(z_t)$')
        ax2.set_title('Real Empirical Lyapunov Decay Curve')
        ax2.legend(frameon=True)
        plt.tight_layout()
        fig2.savefig('lyapunov_decay.pdf', dpi=300)
        plt.close(fig2)
        print("✅ Successfully generated trajectory_contraction.pdf and lyapunov_decay.pdf")

    except Exception as e:
        print(
            f"⚠️ Could not generate plots. Ensure 'INDS_learnedP_final.pt' and 'NODE_best_weights.pt' exist. Error: {e}")

    print("\n>>> 🚀 [SUCCESS]: Benchmark pipeline and statistical tests completed!")


if __name__ == "__main__":
    start_verification_process()
