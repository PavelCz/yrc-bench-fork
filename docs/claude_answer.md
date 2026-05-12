I've read the relevant sections of the paper (especially §3.1, §3.3 with Propositions 1–3, and §4.1). Here's my verdict on each claim, with quotes located.

## Claim 1 — Biases in the encoder

**Verdict: Inconsistent with the paper.** The claim is correctly stated.

This is directly addressed by Proposition 2 (§3.3), which is stated for "any hidden layer in network φ(· ; W) : X → F having a bias term" and concludes that this admits a trivial collapse solution where "φ(x; W∗) = c for every x ∈ X." The paper then states the prescription explicitly:

> "It follows that bias terms should not be used in neural networks with Deep SVDD since the network can learn the constant function mapping directly to the hypersphere center, leading to hypersphere collapse." (§3.3, immediately after Proposition 2)

And the experimental section confirms they acted on this:

> "For Deep SVDD, we remove the bias terms in all network units to prevent a hypersphere collapse as explained in Section 3.3." (§4.1, "Deep Baselines and Deep SVDD")

The implementation's use of bias=True (PyTorch default) on the two Conv2d layers contradicts both the theoretical prescription and the paper's own experimental setup. Your reasoning ("biases give the network a way to produce a constant output independent of the input") is exactly the reasoning given in the proof of Proposition 2.

## Claim 2 — `c` captured in a different space than `f(x)`

**Verdict: Inconsistent with the paper.** The claim is correctly stated.

The paper defines `c` as a point in the output space F, with F being the codomain of φ:

> "let φ(· ; W) : X → F be a neural network … The aim of Deep SVDD then is to jointly learn the network parameters W together with minimizing the volume of a data-enclosing hypersphere in output space F that is characterized by radius R > 0 and center c ∈ F" (§3.1)

And the anomaly score in eq. (5) is `s(x) = ‖φ(x; W∗) − c‖²` — same φ as in the loss. There is no notion in the paper of `c` and `φ(x)` living in different spaces.

For the initialisation rule specifically, the paper says:

> "we set the hypersphere center c to the mean of the mapped data after performing an initial forward pass." (§4.1)

and earlier:

> "We found empirically that fixing c as the mean of the network representations that result from performing an initial forward pass on some training data sample to be a good strategy." (§3.3)

"Mapped data" / "network representations" means the output of φ. The implementation instead records the pre-ReLU activations of layer 12 and averages those, which is not the network's output. So `c` is being defined in ℝ³² while `f(x)` is constrained to [0,∞)³², and your structural-floor argument (`Σ_{c_i < 0} c_i² > 0` is unreachable by any `f(x)`) is correct as a consequence. The paper's formulation makes the loss zero achievable in principle; this implementation's loss has a nonzero infimum that grows with however many components of `c` are negative.

## Claim 3 — Final ReLU on the embedding

**Verdict: Not clearly addressed by the paper, and your claim as stated is partly wrong.** I want to push back here.

You wrote: *"Deep SVDD's theoretical analysis … specifically avoid[s] sign-restrictive activations (and bounded activations more generally) on the embedding output."* The theoretical analysis says the opposite of the first half. Proposition 3 is concerned with activations that have a **non-zero** bound:

> "Consider a network unit having a monotonic activation function σ(·) that has an upper (or lower) bound with sup_z σ(z) ≠ 0 (or inf_z σ(z) ≠ 0)." (Proposition 3, §3.3)

ReLU has inf σ = 0, so it does not satisfy the proposition's condition. The discussion immediately following Proposition 3 makes this explicit and recommends ReLU:

> "Therefore, unbounded activation functions (or functions only bounded by 0) such as the ReLU should be preferred in Deep SVDD to avoid a hypersphere collapse due to 'learned' bias terms." (§3.3, after Proposition 3)

So the paper's theoretical analysis explicitly *endorses* ReLU as the preferred activation. It does not single out sign-restriction as a problem.

What the paper does not address: whether the very last operation of the network should be an activation at all, and specifically whether a ReLU on the final embedding layer is fine. The architecture descriptions in §4.1 and §4.3 end with phrasing like "a final dense layer of 32 units" without saying whether anything follows. The paper just doesn't speak to it.

So the right way to state the concern is narrower than your claim: the paper requires `c ∈ F` (the image of φ), and if you put a ReLU at the very end then F ⊆ [0,∞)^p, which means `c` must also be in [0,∞)^p for the formulation to make sense. That's really the same point as Claim 2, not an independent objection. The paper is silent on the final-ReLU choice per se, and you should not cite Proposition 3 against it — Proposition 3 actually points the other way.

(I'm honoring your instruction not to extrapolate from the reference implementation, which uses Leaky ReLU throughout per §4.1.)

## Claim 4 — Two weight regularisers stacked

**Verdict: Inconsistent with the paper as written, but in a quantitative rather than structural sense.**

The paper's objectives have exactly one weight-decay term with one hyperparameter:

> Eq. (3): `... + (λ/2) Σ_{ℓ=1}^L ‖W_ℓ‖_F²`
> Eq. (4): `... + (λ/2) Σ_{ℓ=1}^L ‖W_ℓ‖_F²`
> "The last term is a weight decay regularizer on the network parameters W with hyperparameter λ > 0, where ‖·‖_F denotes the Frobenius norm." (§3.1)

And the experiments fix this single hyperparameter:

> "We use a batch size of 200 and set the weight decay hyperparameter to λ = 10⁻⁶." (§4.2)
> "We train with a smaller batch size of 64, due to the dataset size and set again hyperparameter λ = 10⁻⁶." (§4.3)

So your claim that the paper specifies one such regulariser is correct. The implementation has two L2-style penalties (explicit Σ_W ‖W‖_F in the loss plus Adam's weight_decay=0.1), and there's no provision in the paper for that.

A skeptical note on how to frame this: the two regularisers in the implementation are *the same kind* of regulariser (both L2 on weights), so structurally this is "effectively a much larger λ" rather than a different functional form. The bigger departure from the paper here is arguably the magnitude — the paper uses λ = 10⁻⁶ throughout its experiments, and the implementation's weight_decay alone is 10⁵× that, with the explicit Frobenius term on top. The "two stacked regularisers" framing is technically true; the "operating at 10⁵× the paper's regularisation strength" framing is the more substantive deviation. The paper doesn't explicitly forbid stacking, but it specifies exactly one term in the objective and one λ.

---

**Summary:** Claims 1, 2, and 4 are correctly diagnosed against the paper. Claim 3 is the one I'd revise — the paper's theoretical analysis does not condemn sign-restrictive activations; if anything, it endorses ReLU. The real problem in that area collapses back into Claim 2 (the `c` ∉ image(φ) issue).