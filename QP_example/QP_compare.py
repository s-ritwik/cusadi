# ------------------------------------------------------------------------
import os, time, numpy as np, torch, casadi
import pandas as pd
import matplotlib.pyplot as plt
from src import *  # contains CusadiFunction helper

# ------------------------------------------------------------------------
batch_sizes = list(range(1000,10000,100))  # change as desired
n, m = 10, 1        # must match qp_gen.py
torch.manual_seed(1)

# prepare to collect
results = []

fn_cas = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR,
                                           'fn_qp_solver_10_1.casadi'))

for BATCH_SIZE in batch_sizes:
    # ----- random SPD Hessians, gradients and constraints -------------------
    H  = torch.randn((BATCH_SIZE, n, n), device='cuda', dtype=torch.double)
    H  = (H.transpose(-1,-2) @ H) + 1e-3*torch.eye(n, device='cuda', dtype=torch.double)
    g  = torch.randn((BATCH_SIZE, n, 1), device='cuda', dtype=torch.double)
    A  = torch.randn((BATCH_SIZE, m, n), device='cuda', dtype=torch.double)
    b  = torch.randn((BATCH_SIZE, m, 1), device='cuda', dtype=torch.double)
    # reshape → (rows, batch) for CusADi inputs
    H_in = H.reshape(BATCH_SIZE, -1).T
    g_in = g.reshape(BATCH_SIZE, -1).T
    A_in = A.reshape(BATCH_SIZE, -1).T
    b_in = b.reshape(BATCH_SIZE, -1).T

    # wrap for GPU
    fn_gpu = CusadiFunction(fn_cas, BATCH_SIZE)

    # GPU batch evaluation
    torch.cuda.synchronize()
    t0 = time.time()
    fn_gpu.evaluate([H_in, g_in, A_in, b_in])
    torch.cuda.synchronize()
    t1 = time.time()
    x_gpu = fn_gpu.outputs_sparse[0]  # (BATCH_SIZE, n)

    # CPU reference
    x_cpu = np.zeros((BATCH_SIZE, n))
    t2 = time.time()
    for i in range(BATCH_SIZE):
        out = fn_cas.call([
            H[i].cpu().numpy(),
            g[i].cpu().numpy(),
            A[i].cpu().numpy(),
            b[i].cpu().numpy()
        ])
        x_cpu[i, :] = np.array(out[0]).reshape(n)
    t3 = time.time()

    # compute error
    x_cpu_t = torch.from_numpy(x_cpu).to('cuda', dtype=torch.double)
    err = (x_gpu - x_cpu_t).abs()

    # record
    results.append({
        'n':          n,
        'm':          m,
        'batch_size': BATCH_SIZE,
        'solved':     BATCH_SIZE,
        'max_err':    err.max().item(),
        'mean_err':   err.mean().item(),
        'gpu_time':   t1 - t0,
        'cpu_time':   t3 - t2,
    })

    # optional realtime print
    print(f"B={BATCH_SIZE:6d}  max|err|={err.max():.2e}  mean|err|={err.mean():.2e}  GPU={t1-t0:.4f}s  CPU={t3-t2:.4f}s")

# ------------------------------------------------------------------------
# build DataFrame and save CSV
df = pd.DataFrame(results,
    columns=['n','m','batch_size','solved','max_err','mean_err','gpu_time','cpu_time'])
df.to_csv('qp_compare_results.csv', index=False)
print("Saved results to qp_compare_results.csv")

# plot
plt.figure(figsize=(8,5))
plt.plot(df['batch_size'], df['gpu_time'], marker='o', label='GPU batch')
plt.xlabel('Batch size')
plt.ylabel('Time (seconds)')
plt.title(f'QP solve times (n={n}, m={m})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(df['batch_size'], df['cpu_time'], marker='s', label='CPU loop')
plt.xlabel('Batch size')
plt.ylabel('Time (seconds)')
plt.title(f'QP solve times (n={n}, m={m})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()