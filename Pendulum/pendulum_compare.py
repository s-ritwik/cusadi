import os
import numpy as np
import torch
from casadi import *
from src import *
import time
# — your existing setup —
BATCH_SIZE = 1000000

x0 = torch.rand((BATCH_SIZE, 2), device='cuda', dtype=torch.double)
g  = 9.81 * torch.ones((BATCH_SIZE, 1), device='cuda', dtype=torch.double)
l  = torch.rand((BATCH_SIZE, 1), device='cuda', dtype=torch.double)
dt = torch.linspace(0.001, 0.1, BATCH_SIZE, device='cuda', dtype=torch.double)
#Starting time note
torch.cuda.synchronize()       
gpu_start=time.time()

fn_casadi_sim_step = casadi.Function.load(
    os.path.join(CUSADI_FUNCTION_DIR, "fn_sim_step.casadi")
)
# wrap in your CusADi function
fn_cusadi_sim = CusadiFunction(fn_casadi_sim_step, BATCH_SIZE)
fn_cusadi_sim.evaluate([x0, g, l, dt])
x_next_cusadi = fn_cusadi_sim.outputs_sparse[0]   # (BATCH_SIZE, 2)
# — now compute “ground truth” via plain CasADi —
# preallocate a NumPy array
x_next_ground = np.zeros((BATCH_SIZE, 2), dtype=np.float64)
#Ending time note
torch.cuda.synchronize()       
gpu_end=time.time()

cpu_start=time.time()

for i in range(BATCH_SIZE):
    # pull one sample off GPU and to CPU numpy
    xi_np = x0[i].cpu().numpy()
    gi_np = g[i].cpu().numpy()
    li_np = l[i].cpu().numpy()
    dti_np = dt[i].cpu().numpy()
    # call the CasADi function; returns a list of arrays
    out = fn_casadi_sim_step.call([xi_np, gi_np, li_np, dti_np])
    # out[0] is a (2,1) or (2,) array
    x_next_ground[i, :] = np.array(out[0]).reshape(2,)
cpu_end=time.time()

# convert back to torch on GPU for direct comparison
x_next_casadi_truth = torch.from_numpy(x_next_ground) \
                         .to(device='cuda', dtype=torch.double)
# — error metrics —
err = x_next_cusadi - x_next_casadi_truth
abs_err = err.abs()
print(f"Solving {BATCH_SIZE} Pendulum dynamics for next [Theta, Omega] parallely")
print(f"Max absolute error:  {abs_err.max().item():.3e}")
print(f"Mean absolute error: {abs_err.mean().item():.3e}")

# optionally RMSE per component or overall:
rmse = torch.sqrt((err**2).mean())
print(f"RMSE:                {rmse.item():.3e}")
print(f"Time for GPU computation: {gpu_end-gpu_start} seconds")
print(f"Time for CPU computation: {cpu_end-cpu_start} seconds")