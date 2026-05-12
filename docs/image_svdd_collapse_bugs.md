# Image-SVDD model-collapse bugs

## Context

While debugging the image-SVDD checkpoint
`/nas/ucb/czempin/data/goal-misgen/trained_svdd/neurips04/svdd_coinrun_image_exp0/trained.joblib`,
we observed that `clf.decision_function` returns the constant
`0.32000002264977` for every input tried:

- 45,880 procgen frames sampled at `random_percent ∈ {0, 50, 100}` via
  `scripts/inspect_image_svdd_scores.py` (per-step *and* per-episode-max).
- Every synthetic input run through `scripts/probe_image_svdd_synthetic.py`
  (zeros, ones, large positive and negative constants, uniform/Gaussian noise,
  single-channel deltas, NaN, ±Inf).

Score range, std, and unique-count (within absolute tolerance 1e-8) collapse
to a single value in every case. This rules out aggregation collapse (per-frame
distribution is already a delta) and sampler bookkeeping issues, and pins the
problem to the trained network itself.

The trained checkpoint reports:

- `clf.c.shape == (32,)`, `||c|| == 0.5657… == √0.32`, every component is ±0.1
  (observed head: `[-0.1 0.1 -0.1 0.1 0.1 -0.1]`).
- `clf.threshold_ == 0.320000022649765`.

`||c||² == 0.32` exactly. The score `||f(x) − c||²` equals `||c||²` iff
`f(x) ≈ 0`, which the synthetic probe confirms is what the network actually
produces for any input. The `2.26e-8` offset is float32 noise on the forward
pass.

Reading `lib/pyod/pyod/models/deep_svdd.py` reveals three architectural bugs
and a compounding regularisation factor that, jointly, make this collapse the
inevitable equilibrium of the training stack as currently written. None of
them depend on the data — retraining the same code on any dataset would
produce the same constant.

## Bug 1 — Conv2d encoder layers have `bias=True`

`InnerDeepSVDD._build_embedder`, lines 146 and 151:

```python
nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1)      # bias defaults to True
nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)             # bias defaults to True
```

The Linear layers further down (`cnn_fc`, `input_layer`, `net_output`, hidden,
decoder) all correctly pass `bias=False`. The two Conv2d layers in the encoder
slipped through. The DeepSVDD theorem requires bias-free layers throughout
the encoder — with biases, the encoder can produce a constant output
independent of the input.

This bug is **YRC-introduced**: upstream `pyod`'s `_build_model` has no Conv2d
layers at all (it's a pure MLP). The CNN `_build_embedder` was added by the
YRC fork.

## Bug 2 — `c` lives in pre-activation space, but the embedding lives in post-activation space

Center init (both `InnerDeepSVDD._init_c` at line 119 and
`DeepSVDD._init_c_from_dataloader` at line 644) registers a forward hook on
the **Linear** module named `net_output`. PyTorch's `forward_hook` captures
the module's output, so this captures the value **before** the trailing
activation added to `fc_part` right after it (lines 187–189):

```python
layers.add_module("net_output", nn.Linear(...))                                       # hook target
layers.add_module(f"hidden_activation_e{len(...)}",
                  get_activation_by_name(self.hidden_activation))                     # ReLU by default
```

But the loss (line 749–755) and `decision_function` (line 768–792) both use
`self.model_(X)`, which returns the **full** forward pass — that is,
post-activation. The center and the embedding live in different spaces.
With ReLU as that activation, the embedding is non-negative, but `c` can have
negative components.

The actual collapse mechanism for this checkpoint is exactly what this bug
plus bug 3 predict:

- The captured `c = [-0.1, +0.1, -0.1, +0.1, +0.1, -0.1, …]` (32-d, all ±0.1) —
  every component was within ε of zero at init, so `_init_c`'s lines 136–137
  (and 687–688) pushed each to ±ε, preserving the (essentially random) sign
  from the init forward pass.
- The post-ReLU embedding can never match any negative `c_i`. The structural
  minimum of `‖f(x) − c‖²` is bounded below by `Σ_{c_i < 0} c_i² > 0` no
  matter what the network learns.
- Combined with bug 1 and the doubled regulariser below, the optimiser
  settles on `f(x) ≈ 0` for every input, which makes
  `dist = ‖0 − c‖² = ‖c‖² = 32 × 0.01 = 0.32` — matching the observed
  `0.32000002264977` to within float32 noise.
- `‖c‖ = 0.5657… = √0.32` is therefore not a coincidence; it's the geometric
  residual of a center that the embedding architecture cannot reach.

This bug is **upstream pyod**, present unchanged from v2.0.2 through v3.4.0.

## Bug 3 — ReLU is the final embedding activation

The default `hidden_activation="relu"` is also applied after `net_output`,
so the SVDD embedding is non-negative. Ruff et al. (2018) call this out
explicitly: a sign-restrictive activation on the embedding output is
structurally incompatible with the SVDD objective. The reference DeepSVDD
implementation uses leaky ReLU intermediates and *no* final activation on
the embedding.

This bug is **upstream pyod**, also present unchanged from v2.0.2 through v3.4.0.

## Bug 4 — Doubled weight regularisation (compounding factor)

`DeepSVDD.fit` line 477 builds the optimiser with
`weight_decay=self.l2_regularizer` (default `0.1`). `_loss` (line 751) **also**
adds an explicit Frobenius penalty on every parameter:

```python
w_d = sum([torch.linalg.norm(w) for w in self.model_.parameters()])
return torch.mean(dist) + w_d
```

So every parameter is being shrunk by both the optimiser's L2 weight decay
*and* an explicit sum-of-Frobenius-norms term in the loss. This stacks
pressure toward `weights ≈ 0`, which (given bug 1's allowed bias absorbing
the data dependence) lands at the constant-near-zero output and matches
what we observe.

Upstream `pyod` has the same dual structure, but multiplies the explicit
Frobenius term by `1e-6`, making it negligible next to the optimiser's
`weight_decay`. The YRC fork dropped that `1e-6` coefficient, so the explicit
term operates at full strength alongside `weight_decay`.

This bug is **upstream pyod in form, YRC-amplified in magnitude**.

## Upstream-vs-YRC attribution summary

| # | Bug | Where it lives | Latest pyod (v3.4.0)? |
|---|-----|----------------|-----------------------|
| 1 | `bias=True` on the two `Conv2d` layers in `_build_embedder` | YRC-introduced | n/a |
| 2 | `_init_c` hook on the `net_output` Linear module captures pre-ReLU | Upstream pyod | Still present |
| 3 | Trailing ReLU added to `_build_fc` after `net_output` | Upstream pyod | Still present |
| 4 | Doubled regularisation (`weight_decay` + explicit `w_d` in `_loss`) | Pattern upstream, magnitude YRC-amplified | Pattern still present; magnitude unchanged upstream |

Upgrading pyod will not fix anything: `deep_svdd.py` is bit-identical from
v2.0.2 through v3.4.0 (`diff` returned no output).

## Minimum-change patch recipe

Producing a checkpoint whose decision scores can actually vary requires
fixing bugs 1, 2, and 3 (any one of which is independently sufficient to
make `||f(x) − c||²` non-constant in principle, but all three are needed to
match the paper). Bug 4 is optional but should be cleaned up.

1. `lib/pyod/pyod/models/deep_svdd.py:146` and `:151` — pass `bias=False` to
   both `nn.Conv2d` calls in `_build_embedder`.
2. `lib/pyod/pyod/models/deep_svdd.py:187-190` — remove the trailing
   activation after `net_output`. The embedding linear shouldn't be
   activation-transformed; that's what makes the captured `c` consistent
   with `self.model_(X)`.
   - Equivalent alternative: move the `register_forward_hook` from
     `net_output` to the activation module that follows it, so `c` is
     captured in the same space as `self.model_(X)`.
   - Equivalent alternative: switch `hidden_activation` to a sign-preserving
     activation (leaky ReLU, tanh, identity) so the embedding can reach
     negative `c_i`.
3. Drop one of the two regularisers in
   `lib/pyod/pyod/models/deep_svdd.py:751` and `:477`. Most reference DeepSVDD
   implementations use only the optimiser's `weight_decay`; keep that and
   remove the explicit `w_d` term, or restore the upstream `1e-6` coefficient.

After those changes, regenerate
`svdd_coinrun_image_exp0/trained.joblib` and re-run
`scripts/probe_image_svdd_synthetic.py` — synthetic inputs should now yield
a *range* of decision scores. (Confirming that the SVDD then actually
*discriminates* on coinrun is a separate question; the patches make the
model capable of varying its output, but DeepSVDD has known training
stability issues even with a correct architecture.)

## Where to land the patches

`lib/pyod/` is vendored in this repo as plain files (not a git submodule;
not a wheel). The patches can land directly in the vendored copy. Bugs 2,
3, and 4 are also genuine upstream bugs in `pyod`; if the user wants to
upstream them, the maintainers should be receptive given that the
behaviour we observed is the structural worst case of the paper's
warnings.
