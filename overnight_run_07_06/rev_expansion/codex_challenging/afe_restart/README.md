# Planned-window AFE restart

This folder is a clean restart of the giant-obstacle experiment. The legacy
trainer remains archived and is not imported here.

The non-negotiable identity is:

```text
generated U_plan == acquired U_plan == fully verified U_plan == replayed U_plan
executed action == verified U_plan[0]
```

There is no `qbuf`, executed-window reconstruction, easy/frontier split,
progress safety label, gamma curriculum, uncertainty-weighted replay, or fixed
scientific number of Adam steps.

Default temperatures have distinct roles:

- expansion candidate sampling: `1.0`;
- independent model-validity audit: `1.0`;
- low-temperature rollout rendering: `0.5` (diagnostic only).

See [METHOD.md](METHOD.md) for the audited decisions and stage gates.

The former curriculum movie is replaced by an active-expansion movie showing
candidate plans, sigma-only acquisition, deterministic verifier outcomes,
certified execution/fallback, and the exact positive records replayed by the
proximal update. Compute runs on physical GPU 1.
