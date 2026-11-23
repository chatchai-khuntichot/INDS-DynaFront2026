# INDS: Certifiable Contraction Dynamics (Reproducibility Package)

This repository contains the official implementation and empirical reproducibility package for the paper **"Certifiable Contraction Dynamics: Lyapunov-Stable Neural ODEs with Provable OOD Robustness Bounds"**, accepted at the **DynaFront Workshop at NeurIPS 2026**.

This package provides the source code, datasets, pre-trained checkpoints, and execution scripts required to reproduce all empirical claims, tables, and figures presented in the manuscript.

---

## 📂 Directory Structure

```text
INDS_DynaFront@NeurIPS26/
├── README.md
├── requirements.txt
├── data/                                           # Benchmark datasets (CGM & Finance)
│   ├── 559-ws-training.xml                         # OhioT1DM training sequence
│   ├── 559-ws-testing.xml                          # OhioT1DM OOD testing sequence
│   └── XAUUSD_H1_202305282200_202605280000.csv     # XAUUSD hourly close prices
├── weights/                                        # Pre-trained model checkpoints
│   ├── INDS_best_weights.pt                        # Base trained INDS checkpoint
│   └── INDS_learnedP_final.pt                      # Final INDS checkpoint with learned metric P
├── models/                                         # Core model architectures
│   ├── __init__.py
│   ├── base_networks.py                            # Baseline Neural ODE & PatchTST
│   └── inds_core.py                                # INDS & DifferentiablePhysicsAuditor
├── cgm_pipeline.py                                 # Healthcare CGM environment router (Zero Leakage)
├── finance_pipeline.py                             # Financial XAUUSD environment router
├── INDS_Verified_Benchmark_fixed.py                # Main multi-seed evaluation & statistical tests
├── INDS_Common_P_Audit.py                          # Common-metric CVR audit across baselines
├── inds_learnedP_retrain.py                        # Two-phase warm-up training for learnable P
└── extract_panel3_values.py                        # Rademacher generalization bound evaluator

⚙️ Environment SetupWe recommend using Python 3.11 with a virtual environment (venv or conda). Install the required dependencies via:Bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
🚀 How to Reproduce ResultsStep 1: Run the Main Evaluation BenchmarkTo replicate the core tables (OOD RMSE, Directional Sharpe Ratio, Pre/Post-Auditor CVR, Ablation Study) across 5 random seeds (42–46) with Wilcoxon signed-rank tests, and generate the phase portrait and Lyapunov decay figures (trajectory_contraction.pdf, lyapunov_decay.pdf):Bashpython INDS_Verified_Benchmark_fixed.py
Step 2: Run Common-Metric CVR Audit (Appendix G)To audit all baselines (NODE, PINNs, SpectralNormNODE, INDS) under the exact same learned Lyapunov metric $P$ (both single-state and pairwise displacement definitions):Bashpython INDS_Common_P_Audit.py
Outputs: common_P_audit_results.json and console summary tables.Step 3: Evaluate the Theoretical Rademacher BoundTo extract the empirical values for latent diameter $\text{diam}(\mathcal{Z})$, metric condition number $\kappa(P)$, and compute the conditional Rademacher complexity bound $\mathcal{R}_n(\mathcal{H}_\alpha)$ from the pre-trained weights:Bashpython extract_panel3_values.py
Outputs: alpha_rademacher.pdf and panel3_metadata.json.Step 4: (Optional) Re-train Learnable Metric $P$ from ScratchIf you wish to re-run the two-phase warm-up training procedure to learn the anisotropic Cholesky metric $P = L L^\top$ from scratch rather than using weights/INDS_learnedP_final.pt:Bashpython inds_learnedP_retrain.py
Outputs: INDS_learnedP_final.pt, learnedP_curves.pdf, alpha_rademacher_learnedP.pdf, and panel3_metadata_learnedP.json.📝 CitationIf you find this code or our theoretical framework useful in your research, please cite:Code snippet@inproceedings{wattanasuphinyakit2026inds,
  title     = {Certifiable Contraction Dynamics: Lyapunov-Stable Neural ODEs with Provable OOD Robustness Bounds},
  author    = {Wattanasuphinyakit, Sooweechag},
  booktitle = {NeurIPS 2026 Workshop on Dynamics at the Frontier of Optimization, Sampling, and Games (DynaFront)},
  year      = {2026}
}
