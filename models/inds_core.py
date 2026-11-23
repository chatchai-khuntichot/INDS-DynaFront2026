# ==============================================================================
# INDS: CORE ARTIFACTS & MATHEMATICAL LYAPUNOV AUDITOR MODULE
# AUTHOR: S.Wattanasuphinyakit Author
# RELATED THEOREMS: Theorem 1 (IRM), Theorem 2 & 5 (Lyapunov & ISS), Theorem 4 (KKT)
# ==============================================================================

import torch
import torch.nn as nn


class DifferentiablePhysicsAuditor(nn.Module):
    def __init__(self, dim, alpha=0.5, beta=100.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.dim = dim

        # Cholesky factor: L lower-triangular, diagonal via exp for PD guarantee
        self.L_offdiag = nn.Parameter(torch.zeros(dim, dim), requires_grad=True)
        self.L_diag_log = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def _L(self):
        """Reconstruct lower-triangular L with positive diagonal."""
        L = torch.tril(self.L_offdiag, diagonal=-1)
        L = L + torch.diag(torch.exp(self.L_diag_log))
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

    def forward(self, z, z_dot):
        P = self.get_P()

        grad_V = 2.0 * torch.matmul(z, P)
        v_dot = torch.sum(grad_V * z_dot, dim=-1, keepdim=True)
        V = torch.sum(z * torch.matmul(z, P), dim=-1, keepdim=True)

        constraint = v_dot + self.alpha * V

        # --- MATHEMATICAL STABILITY GUARD ---
        scaled_constraint = self.beta * constraint
        clamped_constraint = torch.clamp(scaled_constraint, max=80.0)
        violation = (1.0 / self.beta) * torch.log(1.0 + torch.exp(clamped_constraint))

        proj_denom = torch.sum(grad_V ** 2, dim=-1, keepdim=True) + 1e-6
        z_dot_safe = z_dot - (violation / proj_denom) * grad_V

        return z_dot_safe, violation


class NeuralODEVectorField(nn.Module):
    def __init__(self, dim, seed_offset=0):
        super().__init__()
        hidden = 64 + seed_offset
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim)
        )

    def forward(self, z):
        return self.net(z)


class INDS(nn.Module):

    def __init__(self, in_dim, latent_dim, use_stability=True, beta=100.0):
        super().__init__()
        self.use_stability = use_stability

        # Representation Encoder Layer
        self.encoder = nn.Linear(in_dim, latent_dim)
        self.vf = NeuralODEVectorField(latent_dim, seed_offset=44)
        self.decoder = nn.Linear(latent_dim, in_dim)
        self.auditor = DifferentiablePhysicsAuditor(latent_dim, beta=beta)

    def forward(self, x, steps=10, dt=0.1):
        z = self.encoder(x)
        total_violation = 0.0

        # Continuous-time Euler Integration Loop
        for _ in range(steps):
            z_dot = self.vf(z)
            if self.use_stability:
                z_dot, viol = self.auditor(z, z_dot)
                total_violation += torch.mean(viol)
            z = z + z_dot * dt

        return self.decoder(z), total_violation if isinstance(total_violation, torch.Tensor) else torch.tensor(
            total_violation, device=x.device)