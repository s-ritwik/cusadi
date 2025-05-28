# IK_go2_run.py
import os, time
import torch, numpy as np
import casadi
from src import *

# ————————————————
BATCH = 10
# ————————————————
# 1) load CasADi + CusADi

fn_cas = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR,
                                           'fn_IK_go2.casadi'))
fn_gpu = CusadiFunction(fn_cas, BATCH)
# 2) Example: prepare a batch of random foot positions in body frame
#    shape (BATCH,12)
fp = torch.randn(BATCH, 12, device='cuda', dtype=torch.double)*0.1

# 3) run GPU timing
#    CusADi expects inputs as (n_rows, batch), so transpose:
inp = fp.T    # -> (12, BATCH)
torch.cuda.synchronize()
t0 = time.time()
fn_gpu.evaluate([inp])
torch.cuda.synchronize()
t1 = time.time()

# 4) retrieve and reshape output
q_gpu = fn_gpu.outputs_sparse[0]   # (BATCH,12)
print(f"GPU IK for {BATCH} legs took {t1-t0:.4f}s")

# 5) (optional) CPU reference for accuracy
q_cpu = np.zeros((BATCH,12))
t2 = time.time()
for i in range(BATCH):
    out = fn_cas.call([ fp[i].cpu().numpy() ])
    q_cpu[i,:] = np.array(out[0]).reshape(12)
t3 = time.time()
print(f"CPU IK loop took {t3-t2:.4f}s")

# 6) check max error
err = np.abs(q_gpu.cpu().numpy() - q_cpu)
print("max error:", err.max())
