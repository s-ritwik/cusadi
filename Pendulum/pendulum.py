import torch
from src import *
from casadi import *

BATCH_SIZE = 10000

x0 = torch.rand((BATCH_SIZE, 2), device='cuda', dtype=torch.double)                 # Random initial states
g = 9.81 * torch.ones((BATCH_SIZE, 1), device='cuda', dtype=torch.double)           # Gravity for each env.
l = torch.rand((BATCH_SIZE, 1), device='cuda', dtype=torch.double)                  # Random lengths for each env.
dt = torch.linspace(0.001, 0.1, BATCH_SIZE, device='cuda', dtype=torch.double)      # Varying timestep for each env.

fn_casadi_sim_step = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "fn_sim_step.casadi"))
fn_cusadi_sim_step = CusadiFunction(fn_casadi_sim_step, BATCH_SIZE)
fn_cusadi_sim_step.evaluate([x0, g, l, dt])           # Evaluate fn. with CUDA kernel 
x_next = fn_cusadi_sim_step.outputs_sparse[0]       # Access results.
print("Next states shape:", x_next)
