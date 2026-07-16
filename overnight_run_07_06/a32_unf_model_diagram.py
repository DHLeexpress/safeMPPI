"""Render a compact architecture diagram for the a32_unf baseline arm."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "results", "hp_repr", "pretrained_a32.pt")
FINAL = os.path.join(HERE, "results", "sweep_overnight", "a32_unf", "final.pt")
OUTDIR = os.path.join(HERE, "figures", "model")


def load_metadata():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    sd = ck["state_dict"]
    params = {
        "total": sum(v.numel() for v in sd.values()),
        "encoder": sum(v.numel() for k, v in sd.items() if k.startswith("enc_grid.")),
        "trunk": sum(v.numel() for k, v in sd.items() if k.startswith("trunk.")),
        "head": sum(v.numel() for k, v in sd.items() if k.startswith("head.")),
        "other": sum(
            v.numel()
            for k, v in sd.items()
            if not (k.startswith("enc_grid.") or k.startswith("trunk.") or k.startswith("head."))
        ),
    }
    hist = {}
    if os.path.exists(FINAL):
        fin = torch.load(FINAL, map_location="cpu", weights_only=False)
        hist = fin.get("history_tail", {})
    return cfg, params, hist


def fmt_params(n):
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def add_box(ax, xy, wh, title, lines=(), fc="#FFFFFF", ec="#2E3440", lw=1.5, fs=9.5):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.014,rounding_size=0.025",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.09, title, ha="center", va="top", fontsize=fs + 1.4, weight="bold", color="#111827")
    if lines:
        text = "\n".join(lines)
        ax.text(x + 0.045, y + h - 0.22, text, ha="left", va="top", fontsize=fs, color="#1F2937", linespacing=1.28)
    return box


def arrow(ax, p0, p1, color="#374151", text=None, rad=0.0, lw=1.6):
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    if text:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + 0.035, text, ha="center", va="bottom", fontsize=8.5, color=color)


def main():
    cfg, params, hist = load_metadata()
    os.makedirs(OUTDIR, exist_ok=True)

    h_pred = cfg.get("H_pred", 10)
    d = h_pred * 2
    ctx_dim = cfg.get("ctx_dim", 37)
    t_dim = 32
    trunk_in = d + ctx_dim + t_dim
    repr_dim = cfg.get("repr_dim", 32)
    grid_hw = cfg.get("grid_hw", [32, 32])
    hidden = cfg.get("trunk_hidden", [160, 96])

    sr0, cr0 = 0.34714285714285714, 0.47714285714285704
    sr1, cr1 = hist.get("SR", 0.7328571428571428), hist.get("CR", 0.14)
    gdist1 = hist.get("gdist", 0.5544988800532051)
    n_pos = hist.get("n_pos", 60000)

    fig, ax = plt.subplots(figsize=(17, 9.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    ax.text(
        0.35,
        7.68,
        "a32_unf baseline arm: HP-conditioned flow policy",
        ha="left",
        va="top",
        fontsize=20,
        weight="bold",
        color="#0F172A",
    )
    ax.text(
        0.35,
        7.32,
        "Checkpoint: pretrained_a32.pt -> expanded a32_unf/final.pt | H=10, umax=1.0, repr=32, grid=32x32",
        ha="left",
        va="top",
        fontsize=11.5,
        color="#475569",
    )

    # Inputs and context encoder.
    add_box(
        ax,
        (0.45, 5.7),
        (2.05, 0.95),
        "Scene grid",
        [
            f"grid [B,3,{grid_hw[0]},{grid_hw[1]}]",
            "use channel 2 only",
            "clipped HP field",
        ],
        fc="#E0F2FE",
        ec="#0284C7",
    )
    add_box(
        ax,
        (0.45, 4.45),
        (2.05, 0.82),
        "Raw low state",
        ["low5 [B,5]", "relgoal2 + vel2 + gamma"],
        fc="#DCFCE7",
        ec="#16A34A",
    )
    add_box(
        ax,
        (0.45, 3.42),
        (2.05, 0.68),
        "History",
        ["hist [B,16,2]", "stored, no GRU in a32"],
        fc="#F1F5F9",
        ec="#64748B",
        fs=8.8,
    )
    add_box(
        ax,
        (3.05, 5.18),
        (2.55, 1.62),
        "HP encoder E_g",
        [
            "Conv 1->8, 3x3, SiLU",
            "Conv 8->16, 3x3, SiLU",
            "AdaptiveAvgPool 8x8",
            "Flatten 1024",
            "Linear 1024->32, SiLU",
            f"params {fmt_params(params['encoder'])}",
        ],
        fc="#DBEAFE",
        ec="#2563EB",
        fs=8.8,
    )
    add_box(
        ax,
        (6.1, 5.22),
        (1.75, 0.92),
        "HP token",
        ["E_g(grid)", "[B,32]"],
        fc="#EFF6FF",
        ec="#2563EB",
    )
    add_box(
        ax,
        (6.1, 4.16),
        (1.75, 0.82),
        "Context",
        ["concat", "[B,5] + [B,32]", f"ctx [B,{ctx_dim}]"],
        fc="#ECFDF5",
        ec="#059669",
        fs=8.8,
    )

    # Flow field inputs.
    add_box(
        ax,
        (0.45, 1.96),
        (2.05, 0.92),
        "Noised controls",
        [f"U_tau flat [B,{d}]", f"H={h_pred} x 2 controls"],
        fc="#FEF3C7",
        ec="#D97706",
    )
    add_box(
        ax,
        (3.05, 3.04),
        (2.15, 0.82),
        "Time embedding",
        [f"Fourier(tau) [B,{t_dim}]", "sin/cos features"],
        fc="#FDE68A",
        ec="#B45309",
    )
    add_box(
        ax,
        (6.1, 2.06),
        (1.75, 0.92),
        "Field input",
        [f"[B,{d}] + [B,{ctx_dim}] + [B,{t_dim}]", f"concat [B,{trunk_in}]"],
        fc="#F8FAFC",
        ec="#475569",
        fs=8.6,
    )

    # Trunk and head.
    add_box(
        ax,
        (8.55, 2.42),
        (2.45, 1.58),
        "Flow trunk",
        [
            f"Linear {trunk_in}->{hidden[0]}, SiLU",
            f"Linear {hidden[0]}->{hidden[1]}, SiLU",
            f"Linear {hidden[1]}->{repr_dim}, SiLU",
            f"phi_s repr [B,{repr_dim}]",
            "phi_s -> GP sigma",
            f"params {fmt_params(params['trunk'])}",
        ],
        fc="#F3E8FF",
        ec="#7E22CE",
        fs=8.8,
    )
    add_box(
        ax,
        (11.62, 2.65),
        (1.78, 1.12),
        "Head",
        [f"Linear {repr_dim}->{d}", f"v_theta [B,{d}]", f"params {fmt_params(params['head'])}"],
        fc="#FAE8FF",
        ec="#A21CAF",
        fs=8.8,
    )
    add_box(
        ax,
        (11.62, 1.48),
        (1.78, 0.78),
        "Sampled window",
        [f"reshape [{h_pred},2]", "execute first", "replan"],
        fc="#FFE4E6",
        ec="#E11D48",
        fs=8.7,
    )

    # Footer panels.
    add_box(
        ax,
        (0.45, 0.15),
        (3.4, 1.02),
        "Pretraining data",
        [
            "dr05 off-diagonal SafeMPPI windows",
            "|y-x| >= 0.5, goal fixed at (5,5)",
            "7 gamma values, H=10",
        ],
        fc="#FFFFFF",
        ec="#94A3B8",
        fs=8.9,
    )
    add_box(
        ax,
        (4.08, 0.15),
        (4.8, 1.02),
        "Expansion recipe",
        [
            "unfrozen encoder, enc lr mult=0.3, clip=5",
            "demo replay delta=0.25, LwF eta=0.05, alpha=0",
            "batch=64, inner=4, lr=1e-4, temp=1",
            "beta 1.0->0.5->0.2->0.1; mix 70/30/0->34/33/33",
        ],
        fc="#FFFFFF",
        ec="#94A3B8",
        fs=8.5,
    )
    add_box(
        ax,
        (9.15, 0.15),
        (4.25, 1.02),
        "Origin rollout result",
        [
            f"SR {sr0:.2f}->{sr1:.2f}, CR {cr0:.2f}->{cr1:.2f}",
            f"final gdist {gdist1:.2f}, positives {int(n_pos):,}",
            f"total params {params['total']:,} ({fmt_params(params['total'])})",
        ],
        fc="#FFFFFF",
        ec="#94A3B8",
        fs=8.9,
    )

    # Edges.
    arrow(ax, (2.5, 6.18), (3.05, 6.18), text="ch2 HP")
    arrow(ax, (5.6, 5.95), (6.1, 5.82))
    arrow(ax, (2.5, 4.86), (6.1, 4.58), rad=-0.06)
    arrow(ax, (6.98, 5.22), (6.98, 4.98), color="#059669")
    arrow(ax, (7.85, 4.55), (8.55, 3.42), text="ctx")
    arrow(ax, (2.5, 2.42), (6.1, 2.42), text="x_tau")
    arrow(ax, (4.12, 3.04), (6.1, 2.72), color="#B45309")
    arrow(ax, (7.85, 2.42), (8.55, 3.0))
    arrow(ax, (11.0, 3.21), (11.62, 3.21), text="repr32")
    arrow(ax, (12.51, 2.65), (12.51, 2.26), color="#E11D48")

    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUTDIR, f"a32_unf_model_diagram.{ext}"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(os.path.join(OUTDIR, "a32_unf_model_diagram.png"))
    print(os.path.join(OUTDIR, "a32_unf_model_diagram.svg"))


if __name__ == "__main__":
    main()
