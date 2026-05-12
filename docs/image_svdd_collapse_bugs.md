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

Reading `lib/pyod/pyod/models/deep_svdd.py` originally revealed two collapse
bugs that this fork now fixes by default, plus one paper-aligned optional
configuration:

- Fixed bug: the image encoder's Conv2d layers used learnable biases.
- Fixed bug: `_loss` added a full-strength explicit `w_d` term on top of the
  optimiser's weight decay.
- Optional preset: `paper-regularized` additionally captures the center from
  `image(φ)` and sets optimiser `weight_decay` to `1e-6`.

The historical `bugfix-v1` and `bugfix-v2` experiment names are no longer
active variants; their fixes are part of normal image-SVDD training. The
historical `bugfix-v3` run is now the `paper-regularized` preset.

The claims below were verified against the original Deep SVDD paper
(Ruff et al., ICML 2018) by an independent reading; references in each
section. The prompt used and the assistant's full answer live in
[`image_svdd_paper_verification_prompt.md`](image_svdd_paper_verification_prompt.md)
and `claude_answer.md` in the same directory.

## Fixed in this fork — Conv2d encoder biases

### What the bug was

`InnerDeepSVDD._build_embedder`, lines 146 and 151 *as inherited from
`modanesh/pyod`*, constructed the two encoder Conv2d layers without
specifying `bias`, leaving PyTorch's default `bias=True` in effect:

```python
# inherited / upstream version (now patched in this fork):
nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1)      # bias defaults to True
nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)             # bias defaults to True
```

The Linear layers further down (`cnn_fc`, `input_layer`, `net_output`, hidden,
decoder) all correctly pass `bias=False`. The two Conv2d layers in the encoder
slipped through.

### Why this is a bug

Proposition 2 (§3.3 of Ruff et al., ICML 2018) shows that if any hidden layer
in `φ(· ; W) : X → F` has a bias term, there exist parameters `W*` such that
`φ(x; W*) = c` for every input `x ∈ X` — a trivial collapse solution that
satisfies the SVDD loss with zero radius and zero data-distance regardless of
the data. The paper's prescription, immediately after Proposition 2:

> "It follows that bias terms should not be used in neural networks with Deep
> SVDD since the network can learn the constant function mapping directly to
> the hypersphere center, leading to hypersphere collapse."

And the experimental section, §4.1, "Deep Baselines and Deep SVDD":

> "For Deep SVDD, we remove the bias terms in all network units to prevent a
> hypersphere collapse as explained in Section 3.3."

The unpatched implementation contradicts both the theoretical requirement and
the paper's own experimental practice on the two Conv2d layers.

### Provenance

This bug was introduced when the CNN encoder was added to `pyod`. The
original `pyod` (`yzhao062/pyod`) has no Conv2d layers — `_build_model` is
a pure MLP. The CNN `_build_embedder` was added in `modanesh/pyod` (the
fork that upstream YRC-Bench `main` consumes as a git submodule pinned at
commit `4a2080874a…`), and the Conv2d-without-`bias=False` was part of
that addition. That same commit is the current `HEAD` of `modanesh/pyod`,
so merging from upstream YRC-Bench will not fix it.

### How we fixed it

`lib/pyod/pyod/models/deep_svdd.py`, in `InnerDeepSVDD._build_embedder`,
both `nn.Conv2d` calls now pass `bias=False` explicitly, with a short
comment pointing back to this document. This is a one-line change per
layer and aligns the encoder with Proposition 2 and the §4.1 experimental
recipe.

### Relationship to upstream

This is a deliberate divergence from upstream YRC-Bench / `modanesh/pyod`.
If a future merge from upstream YRC-Bench bumps the `lib/pyod` submodule
and we re-flatten it, the `bias=False` arguments will be removed unless
the merge is done with conflict awareness. The corresponding upstream fix
belongs in `modanesh/pyod`'s `_build_embedder`; a PR there would
simultaneously fix upstream YRC-Bench. Until that happens, this fork
must keep the patched version of the two lines.

## Optional preset — `paper-regularized`

The `paper-regularized` preset keeps the default bug fixes and additionally
sets:

```bash
-center_init_post_activation true
-l2_regularizer 1e-6
```

This is the behavior of the old `bugfix-v3` smoke run under a descriptive
name. It is not a numbered preset anymore.

### Center init from `image(φ)`

By default, center init (both `InnerDeepSVDD._init_c` and
`DeepSVDD._init_c_from_dataloader`) registers a forward hook on the **Linear**
module named `net_output`. PyTorch's `forward_hook` captures the module's
output, so this captures the value **before** the trailing activation added to
`fc_part` right after it:

```python
layers.add_module("net_output", nn.Linear(...))                                       # default hook target
layers.add_module(f"hidden_activation_e{len(...)}",
                  get_activation_by_name(self.hidden_activation))                     # preset hook target
```

The `paper-regularized` preset moves the hook to the activation that follows
`net_output`, so the center is averaged from `image(φ)`, the same tensor used
by the loss and `decision_function`.

Section 3.1 of Ruff et al. defines `c` as a point in the output space of `φ`:

> "let `φ(· ; W) : X → F` be a neural network … The aim of Deep SVDD then is
> to jointly learn the network parameters `W` together with minimizing the
> volume of a data-enclosing hypersphere in output space `F` that is
> characterized by radius `R > 0` and center `c ∈ F`."

For the init rule, §3.3 says:

> "We found empirically that fixing `c` as the mean of the network
> representations that result from performing an initial forward pass on some
> training data sample to be a good strategy."

The preset implements that interpretation without removing the trailing ReLU.
The ReLU itself is not treated as a bug here; Proposition 3 recommends ReLU-like
activations for Deep SVDD because they avoid collapse through learned biases.

### Optimiser weight decay `1e-6`

The preset also sets the optimiser's `weight_decay` to `1e-6`, matching the
Deep SVDD paper's reported experiments:

> "We use a batch size of 200 and set the weight decay hyperparameter to
> `λ = 10⁻⁶`." (§4.2)
> "We train with a smaller batch size of 64, due to the dataset size and set
> again hyperparameter `λ = 10⁻⁶`." (§4.3)

The default fork behavior keeps the historical PyOD optimiser default
`l2_regularizer=0.1` for now. Use `--variant paper-regularized` in
`scripts/run_svdd_train.py` to opt into the center-init and optimiser
weight-decay settings together.

## Fixed in this fork — explicit full-strength weight term

`DeepSVDD.fit` builds the optimiser with `weight_decay=self.l2_regularizer`
(default `0.1`). The inherited `modanesh/pyod` `_loss` also added an explicit
Frobenius penalty on every parameter at full strength:

```python
w_d = sum([torch.linalg.norm(w) for w in self.model_.parameters()])
return torch.mean(dist) + w_d
```

Every parameter is shrunk by both the optimiser's L2 weight decay *and* an
explicit sum-of-Frobenius-norms term in the loss.

### Why this is a bug

The Deep SVDD objective in equations (3) and (4) has exactly one weight-decay
term with one hyperparameter `λ`:

> Eq. (3): `… + (λ/2) Σ_{ℓ=1}^L ‖W_ℓ‖_F²`
> Eq. (4): `… + (λ/2) Σ_{ℓ=1}^L ‖W_ℓ‖_F²`
> "The last term is a weight decay regularizer on the network parameters `W`
> with hyperparameter `λ > 0`, where `‖·‖_F` denotes the Frobenius norm." (§3.1)

Both reported experiments fix that hyperparameter to `10⁻⁶`:

> "We use a batch size of 200 and set the weight decay hyperparameter to
> `λ = 10⁻⁶`." (§4.2)
> "We train with a smaller batch size of 64, due to the dataset size and set
> again hyperparameter `λ = 10⁻⁶`." (§4.3)

The bug fixed here is the full-strength explicit term layered on top of the
optimiser regulariser. That term pulls weights toward zero directly in the
loss and contributed to the observed constant-near-zero output. The separate
choice of optimiser `weight_decay=0.1` remains the default PyOD behavior; the
`paper-regularized` preset opts into `1e-6`.

The original `yzhao062/pyod` has the same dual structure, but multiplies
the explicit Frobenius term by `1e-6`, making it negligible next to the
optimiser's `weight_decay`. `modanesh/pyod` dropped that `1e-6` coefficient,
so the explicit term operates at full strength. The drop predates this
fork — your vendored copy and upstream YRC-Bench's pinned submodule both
inherit the post-drop version.

### How we fixed it

`DeepSVDD.__init__` now defaults `explicit_wd_coef` to `0.0`, and `_loss`
multiplies the explicit term by that coefficient. Normal image-SVDD training
therefore relies on the optimiser's `weight_decay` only. The flag remains
available, so legacy `modanesh/pyod` behavior can still be reproduced by
passing `-explicit_wd_coef 1.0`.

## Provenance and attribution

Three layers of `pyod` are involved:

- **`yzhao062/pyod`** — the original PyOD library. Pure MLP DeepSVDD, no
  Conv2d encoder. `deep_svdd.py` is bit-identical between v2.0.2 and
  v3.4.0 (`diff` returned no output).
- **`modanesh/pyod`** — a fork that adds the CNN `_build_embedder`, the
  streaming-DataLoader code, and image-handling extensions. Current `HEAD`
  is commit `4a2080874a…` ("removing learnable features + minigrid
  updates"). Upstream YRC-Bench (`modanesh/YRC-Bench`) consumes this as a
  git submodule pinned to that commit.
- **This fork's `lib/pyod/`** — a flattened copy of `modanesh/pyod` (no
  submodule). The collapse fixes and optional preset controls live directly in
  this vendored copy.

| # | Item | Where it lives | Status in this fork | Latest pyod (v3.4.0)? |
|---|------|----------------|---------------------|-----------------------|
| 1 | `bias=True` on the two `Conv2d` layers in `_build_embedder` | Introduced in `modanesh/pyod`; still in its `HEAD`; pinned by upstream YRC-Bench `main` | **Fixed** (`bias=False` added to both Conv2d) | n/a (no CNN in `yzhao062/pyod`) |
| 2 | Full-strength explicit `w_d` term in `_loss` | The dual structure is in `yzhao062/pyod`; the missing `1e-6` coefficient is `modanesh/pyod`-specific | **Fixed by default** (`explicit_wd_coef=0.0`) | Dual structure + `weight_decay=0.1` default still present; the `1e-6` coefficient is intact upstream |
| 3 | `paper-regularized` center init and optimiser weight decay | Supported by this fork's DeepSVDD flags and `scripts/run_svdd_train.py` preset | **Optional preset** | Center hook behavior remains unchanged upstream |

Merging from upstream YRC-Bench `main` will *reintroduce* the Conv2d-bias bug
unless the local `bias=False` lines are preserved during the merge. It can also
restore the old full-strength explicit `w_d` default if the vendored DeepSVDD
constructor is overwritten. Upgrading `yzhao062/pyod` to v3.4.0 does not cover
the CNN encoder, because that code only exists in `modanesh/pyod`.

## Historical experiment names

- `bugfix-v1`: obsolete control run for the Conv2d-bias fix.
- `bugfix-v2`: now the default behavior: Conv2d biases disabled and
  `explicit_wd_coef=0.0`.
- `bugfix-v3`: renamed to `paper-regularized`; use
  `--variant paper-regularized`.

After retraining a checkpoint under either default image-SVDD behavior or the
`paper-regularized` preset, re-run `scripts/probe_image_svdd_synthetic.py` to
confirm whether synthetic inputs produce non-constant decision scores.
