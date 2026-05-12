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

Reading `lib/pyod/pyod/models/deep_svdd.py` reveals two architectural bugs
and an over-regularisation factor that, jointly, make this collapse the
inevitable equilibrium of the training stack as currently written. None of
them depend on the data — retraining the same code on any dataset would
produce the same constant.

The claims below were verified against the original Deep SVDD paper
(Ruff et al., ICML 2018) by an independent reading; references in each
section. The prompt used and the assistant's full answer live in
[`image_svdd_paper_verification_prompt.md`](image_svdd_paper_verification_prompt.md)
and `claude_answer.md` in the same directory.

## Bug 1 — Conv2d encoder layers have `bias=True`

`InnerDeepSVDD._build_embedder`, lines 146 and 151:

```python
nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1)      # bias defaults to True
nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)             # bias defaults to True
```

The Linear layers further down (`cnn_fc`, `input_layer`, `net_output`, hidden,
decoder) all correctly pass `bias=False`. The two Conv2d layers in the encoder
slipped through.

**What the paper says.** Proposition 2 (§3.3) shows that if any hidden layer
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

The implementation contradicts both the theoretical requirement and the
paper's own experimental practice on the two Conv2d layers.

This bug was introduced when the CNN encoder was added to `pyod`. The
original `pyod` (`yzhao062/pyod`) has no Conv2d layers — `_build_model` is
a pure MLP. The CNN `_build_embedder` was added in `modanesh/pyod` (the
fork that upstream YRC-Bench `main` consumes as a git submodule pinned at
commit `4a2080874a…`), and the Conv2d-without-`bias=False` was part of
that addition. That same commit is the current `HEAD` of `modanesh/pyod`,
so merging from upstream YRC-Bench will not fix Bug 1.

## Bug 2 — `c` lives in a different space than the embedding

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

But the loss (lines 749–755) and `decision_function` (lines 768–792) both use
`self.model_(X)`, which returns the **full** forward pass — including the
trailing ReLU. So the center `c` is computed in `ℝ³²` while the embedding
`f(x) = self.model_(X)` is constrained to `[0, ∞)³²`.

**What the paper says.** Section 3.1 defines `c` as a point in the *output
space of `φ`* — the same space `f(x)` lives in:

> "let `φ(· ; W) : X → F` be a neural network … The aim of Deep SVDD then is
> to jointly learn the network parameters `W` together with minimizing the
> volume of a data-enclosing hypersphere in output space `F` that is
> characterized by radius `R > 0` and center `c ∈ F`."

Equation (5) defines the anomaly score as `s(x) = ‖φ(x; W*) − c‖²`, with the
same `φ` used in training. There is no notion in the paper of `c` and
`φ(x)` living in different spaces.

For the init rule specifically, §3.3:

> "We found empirically that fixing `c` as the mean of the network
> representations that result from performing an initial forward pass on some
> training data sample to be a good strategy."

and §4.1:

> "we set the hypersphere center `c` to the mean of the mapped data after
> performing an initial forward pass."

"Mapped data" / "network representations" is the output of `φ`. The
implementation instead averages the *pre-activation* output of the penultimate
linear, which is not `φ(x)`. So `c` ends up in `ℝ³²` while `φ(x) ∈ [0, ∞)³²`,
and any component `c_i < 0` is structurally unreachable. The per-sample loss
acquires a non-zero infimum

```
L_min(x) ≥ Σ_{i: c_i < 0} c_i²
```

that no training can drive to zero, regardless of the data.

The captured `c` for this checkpoint, `[−0.1, +0.1, −0.1, +0.1, +0.1, −0.1, …]`
(32-d, all ±0.1, from the ε-push in lines 136–137 / 687–688), has roughly half
its components in the unreachable half-space. Combined with Bug 1 and Bug 3
below, the optimiser settles on `f(x) ≈ 0` for every input, which gives
`dist = ‖0 − c‖² = ‖c‖² = 32 × 0.01 = 0.32`, matching the observed
`0.32000002264977` to within float32 noise. `‖c‖ = 0.5657… = √0.32` is the
geometric residual of a center the embedding cannot reach.

**Corollary on the trailing ReLU.** It is tempting to also call the trailing
ReLU itself a bug ("the embedding shouldn't be sign-restricted"), but the
paper does not support that framing. Proposition 3 (§3.3) is concerned with
activations that have `sup σ ≠ 0` or `inf σ ≠ 0` — ReLU has `inf σ = 0` and is
outside the proposition's scope. The paragraph after Proposition 3 explicitly
recommends it:

> "unbounded activation functions (or functions only bounded by 0) such as the
> ReLU should be preferred in Deep SVDD to avoid a hypersphere collapse due to
> 'learned' bias terms."

So the final ReLU is paper-endorsed. The problem is not the ReLU; the problem
is that the hook captures `c` from a different layer than the network output,
so `c ∉ image(φ)`. Any of the following resolves Bug 2 equivalently — they
are not three independent fixes:

- Move the `register_forward_hook` from `net_output` to the activation
  module that follows it (or to `fc_part` as a whole), so `c` is captured
  in `image(φ)`.
- Remove the trailing activation, making layer 12's output *be* `φ(x)` so
  the existing hook captures the right tensor.
- Switch `hidden_activation` to a sign-preserving activation (leaky ReLU,
  tanh, identity); `c` is still captured in the wrong layer, but at least
  every component of `c` is reachable by `f(x)`.

This bug is present in the original `yzhao062/pyod` (unchanged from v2.0.2
through v3.4.0) and in `modanesh/pyod`. It is therefore present in your
local vendored copy and in upstream YRC-Bench's pinned `modanesh/pyod`
submodule.

## Bug 3 — Weight regularisation is ~10⁵× the paper's setting

`DeepSVDD.fit` line 477 builds the optimiser with
`weight_decay=self.l2_regularizer` (default `0.1`). `_loss` (line 751) also
adds an explicit Frobenius penalty on every parameter:

```python
w_d = sum([torch.linalg.norm(w) for w in self.model_.parameters()])
return torch.mean(dist) + w_d
```

Every parameter is shrunk by both the optimiser's L2 weight decay *and* an
explicit sum-of-Frobenius-norms term in the loss.

**What the paper says.** The Deep SVDD objective in equations (3) and (4) has
exactly one weight-decay term with one hyperparameter `λ`:

> Eq. (3): `… + (λ/2) Σ_{ℓ=1}^L ‖W_ℓ‖_F²`
> Eq. (4): `… + (λ/2) Σ_{ℓ=1}^L ‖W_ℓ‖_F²`
> "The last term is a weight decay regularizer on the network parameters `W`
> with hyperparameter `λ > 0`, where `‖·‖_F` denotes the Frobenius norm." (§3.1)

And both reported experiments fix that hyperparameter to `10⁻⁶`:

> "We use a batch size of 200 and set the weight decay hyperparameter to
> `λ = 10⁻⁶`." (§4.2)
> "We train with a smaller batch size of 64, due to the dataset size and set
> again hyperparameter `λ = 10⁻⁶`." (§4.3)

The structural deviation ("one regulariser in the paper, two in the
implementation") is real but minor — both terms are L2-on-weights, so the
combination is mathematically equivalent to "one larger λ" with some
weighting. The substantive deviation is **magnitude**: the paper uses
`λ = 10⁻⁶` throughout, while the implementation's `weight_decay` alone is
`0.1 = 10⁵ × 10⁻⁶`, with the explicit Frobenius term layered on top. The
network is being regularised orders of magnitude harder than the paper's
recipe, which by itself biases the optimum toward `weights ≈ 0` and the
constant-near-zero output.

The original `yzhao062/pyod` has the same dual structure, but multiplies
the explicit Frobenius term by `1e-6`, making it negligible next to the
optimiser's `weight_decay`. `modanesh/pyod` dropped that `1e-6` coefficient,
so the explicit term operates at full strength. The drop predates this
fork — your vendored copy and upstream YRC-Bench's pinned submodule both
inherit the post-drop version. Even so, the optimiser's
`weight_decay = 0.1` default (present in `yzhao062/pyod` already) departs
from the paper's `λ = 10⁻⁶` by 5 orders of magnitude, so the dropped
coefficient is not the only thing that needs fixing.

This bug is **upstream-`pyod` in form** (dual-term structure plus
`weight_decay = 0.1` default) and **upstream-`modanesh/pyod`-amplified in
magnitude** (1e-6 coefficient on the explicit term dropped, not by this
fork).

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
  submodule). Differs from `modanesh/pyod` only in `black`/`ruff`-style
  formatting; substantive code at the bug sites is identical.

| # | Bug | Where it lives | Latest pyod (v3.4.0)? |
|---|-----|----------------|-----------------------|
| 1 | `bias=True` on the two `Conv2d` layers in `_build_embedder` | Introduced in `modanesh/pyod`; still in its `HEAD`; pinned by upstream YRC-Bench `main`; inherited verbatim here | n/a (no CNN in `yzhao062/pyod`) |
| 2 | `_init_c` hook on the `net_output` Linear module captures values outside `image(φ)` | Original `yzhao062/pyod`; inherited unchanged by `modanesh/pyod` and this fork | Still present |
| 3 | Weight regularisation ~10⁵× the paper's setting (dual term + `weight_decay=0.1`) | Dual-term structure and `weight_decay=0.1` default are in `yzhao062/pyod`; the `1e-6` coefficient on the explicit term was dropped in `modanesh/pyod`, not here | Dual structure + `weight_decay=0.1` default still present; the `1e-6` coefficient is intact upstream |

Merging from upstream YRC-Bench `main` will not fix any of these — upstream
YRC pins `modanesh/pyod` as a submodule, and the latest commit of that
fork still contains all three bugs at the same lines. Upgrading
`yzhao062/pyod` to v3.4.0 doesn't help either (the relevant file is
unchanged).

## Minimum-change patch recipe

The goal is to bring the trained SVDD architecture into line with what the
paper prescribes, so that decision scores can actually vary across inputs.
This does not by itself guarantee the SVDD will then *discriminate* well on
coinrun — that's a separate question — but it is a precondition.

1. **Bug 1 fix.** `lib/pyod/pyod/models/deep_svdd.py:146` and `:151` — pass
   `bias=False` to both `nn.Conv2d` calls in `_build_embedder`. Aligns with
   Proposition 2 and §4.1.

2. **Bug 2 fix (choose one).** Make `c` and `φ(x)` live in the same space, as
   required by §3.1 and the init rule in §4.1. Any of these are equivalent:
   - Move the `register_forward_hook` from `net_output` to the activation
     module that follows it (`hidden_activation_e{len(...)}`), or to
     `fc_part` itself. *Preferred*: keeps the paper-endorsed ReLU at the
     output (Proposition 3) and uses the existing init code path.
   - Remove the trailing activation in `_build_fc` (lines 187–190), making
     layer 12's output *be* `φ(x)`.
   - Switch `hidden_activation` from `relu` to a sign-preserving activation
     (leaky ReLU, tanh, identity). Not ideal — Proposition 3 recommends ReLU
     specifically — but does fix the reachability issue.

3. **Bug 3 fix.** Two changes, not one:
   - Drop the explicit `w_d` term in `_loss` (line 751), *or* restore the
     upstream `1e-6` coefficient on it.
   - Lower the optimiser's `weight_decay` from `0.1` toward the paper's
     `λ = 10⁻⁶` (§4.2, §4.3). The exact value can be tuned, but `0.1` is
     ~10⁵× too high regardless of what happens with the explicit term.
   The first change alone is insufficient because the optimiser's
   `weight_decay = 0.1` is already several orders of magnitude above the
   paper's setting.

After those changes, regenerate
`svdd_coinrun_image_exp0/trained.joblib` and re-run
`scripts/probe_image_svdd_synthetic.py` — synthetic inputs should now yield
a *range* of decision scores. DeepSVDD has known training stability issues
even with a paper-faithful architecture, so further iteration may still be
required.

## Where to land the patches

`lib/pyod/` is vendored in this repo as plain files (not a git submodule;
not a wheel). The immediate patches can land directly in the vendored copy.

If you also want the fixes available outside this fork, the natural upstream
target differs by bug:

- **Bug 1** was introduced in `modanesh/pyod`. The fix belongs there, and
  would simultaneously fix upstream YRC-Bench (which pins that fork).
- **Bug 2** is in the original `yzhao062/pyod` as well as in
  `modanesh/pyod`. A PR to `yzhao062/pyod` would reach the broader pyod
  user base; the paper references in this document make the case directly.
- **Bug 3**: the dual-term structure and the `weight_decay = 0.1` default
  are in `yzhao062/pyod`; the dropped `1e-6` coefficient on the explicit
  term is `modanesh`-specific. The `weight_decay` default is arguably
  configuration-not-bug (callers can set their own `l2_regularizer` /
  `weight_decay`), so the cleanest upstream change is to restore the
  `1e-6` coefficient in `modanesh/pyod`. For this fork's purposes, the
  more pragmatic fix is to pin both values to paper-faithful settings at
  training time rather than rely on library defaults.
