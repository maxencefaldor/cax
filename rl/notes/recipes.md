# Recipes

Implementation-level facts from the full text of the papers the experiments copy, so that a run can be checked against its source without reopening the PDF.
Symbols follow the papers.

## SHAC (Xu et al., 2022)

- Actor loss over a window of h steps and N rollouts: minus the mean over rollouts of the discounted reward sum plus gamma^h times the target critic at the last state.
- Windows are chained: the next window starts from the last state of the previous one, with the gradient cut on the carried state.
- Critic target is TD(lambda) over the window with lambda 0.95, computed from the target critic, treated as a constant; critic loss is the squared error to it.
- Target critic is a Polyak copy, alpha 0.995 in general.
- Sixteen critic iterations per actor update, each over four minibatches of the window's N times h states.
- Learning rates 2e-3 actor and 5e-4 critic with linear decay, Adam betas (0.7, 0.95), gamma 0.99, h 32, N 64, gradient-norm clip 1.0 (from AHAC's table).
- Observations normalized by a running mean and standard deviation; ELU and LayerNorm in every hidden layer.
- Notebook 63 differs in: gamma 0.999, per-tensor gradient normalization then a global clip at 0.5, four critic iterations on the full batch, alpha 0.95, a softplus-bounded critic, and a 200-step bootstrap ramp.

## AHAC (Georgiev et al., 2024)

- The constraint is on the norm of the contact-force Jacobian at each step of the window, not on the force; threshold C 500 for every task, after scaling rows by max(acceleration, 1).
- Lagrangian on the constrained return with one multiplier per step; the horizon H is itself updated by the sum of the multipliers, so it shrinks only while the constraint is violated.
  H starts at 32 and converges to about 29 on Ant, the gait period.
- Truncation is per batch: one H shared by all parallel rollouts; per-rollout truncation was unvectorizable and hurt through variance across lengths.
- Critic is a pair with the minimum taken, no target copy, trained until the loss stops falling, at most 64 iterations, minibatches of 8.
- For an NCA the analogue of the contact-force Jacobian is the per-step state Jacobian norm of the rule, which E2 measures anyway.

## Gradients are not all you need (Metz et al., 2021)

- The gradient through an unroll is a sum over steps of products of per-step Jacobians; if the largest eigenvalue magnitude of a per-step Jacobian is typically above one the norm grows exponentially with the unroll, below one it vanishes, near one it is well behaved.
- Their measurement, on Brax Ant in float64: fix a random direction in parameter space, plot the loss along that line for several unroll lengths at fixed randomness, then averaged over noise seeds, then the empirical variance of the gradient over noise seeds at four fixed parameter points against unroll length on a log axis.
- Their figure 5 has four panels: the eigenvalue spectrum of one per-step Jacobian in the complex plane, the largest eigenvalue magnitude of each per-step Jacobian against the step, the same for the cumulative product against the step, and the gradient norm against the unroll length.
- Remedies tested: truncated backpropagation (only a narrow band of truncation lengths learns on a 400-step Ant), clipping, a learned value bootstrap after a few steps, and ES, whose variance does not depend on the landscape's frequency content.
- E2 reproduces the four panels for the gecko at checkpoints along training, with and without dropout.

## Attractors in NCA (Kvalsund and Stovold, 2026)

- System: the Distill gecko with the original code, update probability set to one, pool training with circle-mask damage, 72x72x16 state, three seeds.
- Lyapunov exponents by the finite-difference method with renormalization every step, epsilon 1e-4, 4,000-step burn-in, 10,000 steps, Gram-Schmidt for the spectrum.
- Seed one: largest exponent 0.0, a limit cycle with a Fourier peak at frequency 0.183 (period about 5.5 steps).
  Seed two: largest exponents -0.005, a torus with four incommensurate frequencies.
  Seed three: -0.009 with three frequencies.
  No fixed points, no positive exponents.
- Along training, seed three: no attractor at epoch 1,000, an oscillatory attractor with a largest exponent near zero at epoch 2,000, -0.009 at the end.
- Large perturbations recover in 1,500 steps but can settle in a secondary mode.

## DreamerV3 (Hafner et al., 2023)

- symlog(x) = sign(x) ln(|x| + 1) and its inverse symexp; regression targets are symlog-transformed.
- Two-hot critic head: 255 bins at symexp(linspace(-20, 20)), the target's mass split between the two neighbouring bins in proportion to distance, cross-entropy loss, readout as the expectation over bins, output layer zero-initialized.
- Lambda-return with lambda 0.95, gamma 0.997, horizon 15, the last value bootstrapped from the critic.
- Critic regularized by a second cross-entropy toward the output of an EMA copy of itself with decay 0.98, scale 1; no hard target.
- Return normalization: S is an EMA with decay 0.99 of the range between the 5th and 95th percentiles of the lambda-returns in the batch; the advantage is divided by max(1, S).
- Actor is REINFORCE on the normalized advantage plus entropy at 3e-4 in v3; v2 found pure analytic gradients best for continuous actions.
- For E3: the two-hot head and percentile normalization replace the softplus bound and the fixed reward scale of 63.

## Alpha-order gradients (Suh et al., 2022)

- Zeroth-order estimator: the score-function gradient over parameter noise with the zero-noise rollout as baseline; first-order estimator: the analytic gradient; both averaged over N independent rollouts, with empirical variances sigma_0^2 and sigma_1^2.
- The mixture is alpha times first-order plus (1 - alpha) times zeroth-order, with alpha = sigma_0^2 / (sigma_0^2 + sigma_1^2) as the variance-minimizing choice.
- The bias guard: B is the norm of the disagreement between the two mean estimators and epsilon is a Bernstein confidence radius on the zeroth-order mean; alpha is reduced to (gamma - epsilon) / B when alpha B exceeds gamma - epsilon, and set to zero when epsilon exceeds gamma.
- The first-order estimator's low variance is not evidence of correctness; disagreement with the zeroth-order estimator beyond its confidence radius is the bias detector.

## Per-step inverse-variance weighting (Onoda et al., 2026)

- For each step, rollout and action dimension, compute the reparameterization gradient and the likelihood-ratio gradient with respect to the policy's mean and standard deviation at that step.
- Variances are taken across rollouts at a fixed step and action dimension; the weight is v_0 / (v_0 + v_1) and the fused per-step gradient is pushed through a vector-Jacobian product of the policy to reach the parameters.
- No EMA, no epsilon, no clipping; critic fitted by squared error to advantage plus value, advantages by GAE.
- For an NCA the per-step, per-cell quantities exist naturally, so the same weighting can be computed per cell.
