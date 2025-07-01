#!/usr/bin/env python3
"""
amber_visualize_refs.py
Visualise the reference trajectories produced by amber_casadi_main_cpu.py.

• Assumes the generator saved:
      references/amber_vxs.npy
      references/amber_reference_ts.npy
      references/amber_reference_qs.npy
      references/amber_reference_foot_refs.npy
  (identical naming convention to the Go2 script, just “amber_” instead of “go2_”)
• Only the forward–velocity axis (v_x) is generated, so no v_y or w_z sliders.
• Plots:
      – joint angles versus time for the four actuated joints
      – foot-tip X-position versus time as an extra figure (optional)
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------------------------------------------------
# CLI ─ allow the user to pick which v_x index to display
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Visualise Amber reference trajectories")
parser.add_argument(
    "--idx",
    type=int,
    default=None,
    help="Index into the v_x grid to plot (0 … |v_x|-1). "
         "If omitted, the middle index is used.",
)
parser.add_argument(
    "--animate",
    action="store_true",
    help="Animate joint traces being drawn (Matplotlib FuncAnimation).",
)
args = parser.parse_args()

# ----------------------------------------------------------------------
# Load reference arrays written by amber_casadi_main_cpu.py
# ----------------------------------------------------------------------
refs_dir = Path(__file__).resolve().parent / "references"
v_xs   = np.load(refs_dir / "amber_vxs.npy")                   # (Nv,)
ts     = np.load(refs_dir / "amber_reference_ts.npy")          # (N,)
q_refs = np.load(refs_dir / "amber_reference_qs.npy")          # (Nv, N, 4)
foot_refs = np.load(refs_dir / "amber_reference_foot_refs.npy")  # (Nv, N, 2, 3)

Nv, N, _ = q_refs.shape

# Choose which velocity to plot
idx = args.idx if args.idx is not None else Nv // 2
if idx < 0 or idx >= Nv:
    raise IndexError(f"idx must be in [0, {Nv-1}]")

vx = v_xs[idx]
q  = q_refs[idx]                # (N, 4)
p_foot = foot_refs[idx]         # (N, 2, 3)

joint_labels = ["L-Hip", "L-Knee", "R-Hip", "R-Knee"]

# ----------------------------------------------------------------------
# Plot joint-angle traces
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
lines = [ax.plot([], [], lw=2, label=lbl)[0] for lbl in joint_labels]

ax.set_xlim(ts[0], ts[-1])
ax.set_ylim(q.min()*1.1, q.max()*1.1)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Joint angle [rad]")
ax.set_title(f"Amber joint references  –  vₓ = {vx:.2f} m/s")
ax.legend(loc="upper right")

def init():
    for ln in lines:
        ln.set_data([], [])
    return lines

def update(frame):
    for j, ln in enumerate(lines):
        ln.set_data(ts[:frame], q[:frame, j])
    return lines

if args.animate:
    ani = FuncAnimation(fig, update, frames=N, init_func=init,
                        blit=True, interval=50, repeat=True)
else:  # static plot: draw full traces immediately
    for j, ln in enumerate(lines):
        ln.set_data(ts, q[:, j])

# ----------------------------------------------------------------------
# Secondary plot – foot-tip X-positions (helps sanity-check 1-DOF motion)
# ----------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 3))
for leg in range(2):
    ax2.plot(ts, p_foot[:, leg, 0], label=f"Foot {leg} X")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Foot X [m]")
ax2.set_title("Foot-tip X trajectories (world frame)")
ax2.legend(loc="best")
fig2.tight_layout()

plt.show()
