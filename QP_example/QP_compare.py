# qp_compare.py -----------------------------------------------------------
import os, time, numpy as np, torch, casadi
from src import *                       # contains CusadiFunction helper

# ------------------------------------------------------------------------
BATCH_SIZE = 10     # change as desired
n, m = 10, 2           # must match qp_gen.py
torch.manual_seed(1)
# ------------------------------------------------------------------------

# ----- random SPD Hessians, gradients and constraints -------------------
H  = torch.randn((BATCH_SIZE, n, n), device='cuda', dtype=torch.double)
H  = (H.transpose(-1,-2) @ H) + 1e-3*torch.eye(n, device='cuda', dtype=torch.double)
g  = torch.randn((BATCH_SIZE, n, 1), device='cuda', dtype=torch.double)
A  = torch.randn((BATCH_SIZE, m, n), device='cuda', dtype=torch.double)
b  = torch.randn((BATCH_SIZE, m, 1), device='cuda', dtype=torch.double)

# reshape → (rows, batch) for CusADi inputs
H_in = H.reshape(BATCH_SIZE, -1).T      # (n*n , B)
g_in = g.reshape(BATCH_SIZE, -1).T
A_in = A.reshape(BATCH_SIZE, -1).T
b_in = b.reshape(BATCH_SIZE, -1).T
# ------------------------------------------------------------------------

# ----- load .casadi and wrap with CusADi --------------------------------
fn_cas = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR,
                                           'fn_qp_solver_10_2.casadi'))
fn_gpu = CusadiFunction(fn_cas, BATCH_SIZE)

# ---------------- GPU batch evaluation ----------------------------------
torch.cuda.synchronize()
t0 = time.time()
fn_gpu.evaluate([H_in, g_in, A_in, b_in])
torch.cuda.synchronize()
t1 = time.time()
x_gpu = fn_gpu.outputs_sparse[0]          # (BATCH_SIZE, n)

# CPU reference ------------------------------------------------
x_cpu = np.zeros((BATCH_SIZE, n))
t2=time.time()
for i in range(BATCH_SIZE):
    try:
        out = fn_cas.call([ H[i].cpu().numpy(),
                        g[i].cpu().numpy(),
                        A[i].cpu().numpy(),
                        b[i].cpu().numpy() ])
    except RuntimeError as e:
        print("Unsolvable qp: ",e)
    x_cpu[i, :] = np.array(out[0]).reshape(n)
t3=time.time()
x_cpu_t = torch.from_numpy(x_cpu).to('cuda', dtype=torch.double)   # (BATCH_SIZE, n)
print(x_cpu_t)
print(x_gpu)
err = (x_gpu - x_cpu_t).abs()              # now dimensions agree
print(f"n:{n}, m:{m}")
print(f"Solved {BATCH_SIZE} small QPs")
print(f"max |err|  = {err.max().item():.3e}")
print(f"mean |err| = {err.mean().item():.3e}")
print(f"GPU batch  : {(t1-t0):.4f} s")
print(f"CPU loop   : {(t3-t2):.4f} s")
3