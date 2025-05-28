# QP_ineq_compare.py -------------------------------------------------------
import os
import time
import numpy as np
import torch
import casadi
from src import *

# ----------------- parameters -----------------
BATCH_SIZE = 10000   # adjust as desired
n, m = 3, 2          # must match QP_ineq_gen.py
torch.manual_seed(0)
# ----------------------------------------------

# ----- random SPD Hessians, gradients and constraints --------------------
# H: (B, n, n), make SPD
H = torch.randn((BATCH_SIZE, n, n), device='cuda', dtype=torch.double)
H = H @ H.transpose(-1, -2) + 1e-3 * torch.eye(n, device='cuda', dtype=torch.double)

g = torch.randn((BATCH_SIZE, n, 1), device='cuda', dtype=torch.double)
A = torch.randn((BATCH_SIZE, m, n), device='cuda', dtype=torch.double)
b = torch.randn((BATCH_SIZE, m, 1), device='cuda', dtype=torch.double)

# reshape → (rows, batch) for CusADi inputs
H_in = H.reshape(BATCH_SIZE, n * n).T    # (n*n, B)
g_in = g.reshape(BATCH_SIZE, n).T.unsqueeze(1) if False else g.reshape(BATCH_SIZE, n).T
g_in = g.reshape(BATCH_SIZE, n * 1).T    # (n, B)
A_in = A.reshape(BATCH_SIZE, m * n).T    # (m*n, B)
b_in = b.reshape(BATCH_SIZE, m * 1).T    # (m, B)

# ----- load inequality QP .casadi and wrap with CusADi --------------------
fn_cas = casadi.Function.load(
    os.path.join(CUSADI_FUNCTION_DIR, 'fn_qp_solver_ineq_3_2.casadi')
)
fn_gpu = CusadiFunction(fn_cas, BATCH_SIZE)

# ---------------- GPU batch evaluation -----------------------------------
torch.cuda.synchronize()
t0 = time.time()
fn_gpu.evaluate([H_in, g_in, A_in, b_in])
torch.cuda.synchronize()
t1 = time.time()

# outputs_sparse: [x_opt, lam_opt]
x_gpu   = fn_gpu.outputs_sparse[0]        # (BATCH_SIZE, n)
lam_gpu = fn_gpu.outputs_sparse[1]        # (BATCH_SIZE, m)

# ---------------- CPU reference loop --------------------------------------
x_cpu   = np.zeros((BATCH_SIZE, n))
lam_cpu = np.zeros((BATCH_SIZE, m))
t2 = time.time()
for i in range(BATCH_SIZE):
    H_i = H[i].cpu().numpy()
    g_i = g[i].cpu().numpy()
    A_i = A[i].cpu().numpy()
    b_i = b[i].cpu().numpy()
    out = fn_cas.call([H_i, g_i, A_i, b_i])
    x_cpu[i, :]   = np.array(out[0]).reshape(n)
    lam_cpu[i, :] = np.array(out[1]).reshape(m)
t3 = time.time()

x_cpu_t   = torch.from_numpy(x_cpu).to('cuda', dtype=torch.double)
lam_cpu_t = torch.from_numpy(lam_cpu).to('cuda', dtype=torch.double)

# ---------------- error & timing report -----------------------------------
err_x   = (x_gpu   - x_cpu_t).abs()
err_lam = (lam_gpu - lam_cpu_t).abs()

print(f"QP-ineq compare: n={n}, m={m}, batch={BATCH_SIZE}")
print(f"max |x_error|      = {err_x.max().item():.3e}")
print(f"mean |x_error|     = {err_x.mean().item():.3e}")
print(f"max |lambda_error| = {err_lam.max().item():.3e}")
print(f"mean |lambda_error|= {err_lam.mean().item():.3e}")
print(f"GPU batch time     = {t1 - t0:.4f} s")
print(f"CPU loop time      = {t3 - t2:.4f} s")
