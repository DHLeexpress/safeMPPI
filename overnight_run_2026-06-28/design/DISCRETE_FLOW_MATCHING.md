# Flow matching on discrete spaces — survey + our design decision

You asked whether the grid-coverage policy should use *discrete* flow matching (since the grid is discrete).
Short answer: **no — we keep a CONTINUOUS conditional flow over single-integrator velocity sequences and only
discretize for the coverage *metric*.** Reasoning + the relevant literature below.

## What "discrete flow matching" is, and when it's needed
Continuous flow matching learns an ODE velocity field transporting noise→data for **continuous** variables
(images, our control sequences). For **natively discrete** data (text tokens, molecule graphs, amino acids) you
can't take an ODE in `R^d`; you need a discrete-state process. The main lines:

- **Discrete Flow Matching (Gat, Remez, Shaul, Kreuk, Chen, Synnaeve, Adi, Lipman, NeurIPS 2024)** — a flow paradigm
  over discrete tokens via a continuous-time Markov chain (CTMC) with learned rate/velocity on the probability
  simplex; non-autoregressive discrete generation. https://arxiv.org/abs/2407.15595
- **D3PM — Structured Denoising Diffusion in Discrete State-Spaces (Austin et al., 2021)** — discrete diffusion with
  structured transition matrices (absorbing/uniform/nearest-neighbor). https://arxiv.org/abs/2107.03006
- **SEDD — Score Entropy Discrete Diffusion (Lou, Meng, Ermon, 2023/24)** — learns reverse CTMC jump rates via a
  score-entropy (concrete-score) objective; SOTA discrete-diffusion LM. https://arxiv.org/abs/2310.16834
- **Dirichlet Flow Matching (Stark et al., 2024)** — flow matching on the probability simplex for sequences.
  https://arxiv.org/abs/2402.05841
- **Edit Flows (2025)** — flow matching with edit operations (insert/delete/substitute) for variable-length discrete
  data. https://arxiv.org/abs/2506.09018

These are the right tools **iff the generated object is intrinsically discrete**.

## Why we do NOT need discrete FM here
Our generated object is a **control sequence** `U = (u_0,…,u_{H-1}) ∈ R^{H×2}` of single-integrator velocities —
**continuous**. The 7×7 grid is not the action space; it is only a **coordinate for measuring coverage** (we bin a
continuous trajectory to the cells it passes through and check the monotone-lattice-path it traces). So:

- The **policy** stays a continuous conditional flow `q_θ(U | state)` (cond-OT path, Euler ODE) — same machinery as
  the chessboard Fig-2 reproduction, just **conditioned on the current state** (we learn control, not a
  time-invariant distribution).
- The **discreteness** lives entirely in the **coverage metric / verifier**: `lattice_paths.path_signature` maps a
  continuous rollout to one of the 74 enumerable safe monotone paths (or rejects it). This is a measurement, not a
  modeling, choice — no discrete-FM needed.

Using discrete FM would force the action space onto a token grid (e.g. {right, up} moves), throwing away the
continuous dynamics/safety we actually care about (single integrator, DTCBF, polytope_v2). That's the wrong
abstraction for a *control* problem.

## When we WOULD switch to discrete FM
If we later model the path as a **sequence of discrete decisions** (homotopy choices / move tokens) rather than
continuous controls — e.g. to directly cover a combinatorial path space or generate graph/topology decisions — then
Discrete Flow Matching (Gat et al.) or SEDD would be the right backbone, and the coverage metric would become the
native output space. For the current continuous-control setting it is unnecessary complexity.

Sources: Discrete Flow Matching https://arxiv.org/abs/2407.15595 · D3PM https://arxiv.org/abs/2107.03006 ·
SEDD https://arxiv.org/abs/2310.16834 · Dirichlet FM https://arxiv.org/abs/2402.05841 ·
Edit Flows https://arxiv.org/abs/2506.09018
