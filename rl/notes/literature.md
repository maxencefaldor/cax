# Literature

The survey behind the programme, as of 2026-09-08.
Three questions: what has been done with reinforcement learning on cellular automata, what the short-horizon analytic-gradient literature knows that transfers, and which NCA problems are stuck on backpropagation through time.
Each entry carries the claim that matters for us, not the abstract.

## Analytic policy gradients through differentiable systems

This is what notebook 63 is an instance of.

- **SHAC** (Xu et al., ICLR 2022, [2204.07137](https://arxiv.org/abs/2204.07137)).
  Actor gradient through a 32-step window of the simulator, terminated by a critic; windows chained across the episode with the state carried and the gradient cut; critic on TD(lambda) targets with lambda 0.95 from a Polyak target copy.
  The critic is described as a smooth surrogate for the long-horizon landscape.
  Code public.
- **AHAC** (Georgiev et al., ICML 2024, [2405.17784](https://arxiv.org/abs/2405.17784)).
  Diagnoses SHAC's failures as gradient error from stiff dynamics inside the window, and ends the window early when a stiffness signal (contact force or Jacobian norm) crosses a threshold, with the horizon tuned online by a Lagrangian.
  The direct precedent for an adaptive horizon in an NCA driven by a measured Jacobian norm.
  Code public.
- **SAPO** (Xing et al., ICLR 2025, [2412.12089](https://arxiv.org/abs/2412.12089)).
  SHAC with a stochastic actor and an entropy term; entropy smooths the analytic-gradient landscape and cuts seed variance.
  Code public.
- **Suh et al.** (ICML 2022, [2202.00817](https://arxiv.org/abs/2202.00817)) and **Onoda et al.** (ICLR 2026, [2604.18161](https://arxiv.org/abs/2604.18161)).
  First-order estimators are empirically biased under discontinuity and high-variance under chaos, zeroth-order ones are unbiased; the alpha-order mixture picks the weight from a variance-bias diagnostic.
  The 2026 revisit finds that per-step inverse-variance weighting of the two (the PIPPS lineage) beats switching.
- **Metz et al., Gradients are not all you need** ([2111.05803](https://arxiv.org/abs/2111.05803)).
  One failure across RNNs, physics and learned optimizers: if the per-step Jacobian's leading eigenvalue exceeds one, the gradient norm grows exponentially with the horizon and its variance grows faster than its mean.
  The remedy list is truncation, black-box estimators, mixing both, or a less chaotic system.
  Their gradient-norm-versus-horizon curve is the measurement of E2.
- **PIPPS** (Parmas et al., ICML 2018) and **Metz et al. 2019** ([1810.10180](https://arxiv.org/abs/1810.10180)).
  Combine reparameterization and likelihood-ratio (or ES) gradients per step, or per parameter, with inverse-variance weights.
  The ES branch is unusually cheap for an NCA, whose rule has about 10^4 parameters.
- **PES** (Vicol et al., ICML 2021, [2112.13835](https://arxiv.org/abs/2112.13835)).
  ES over truncated unrolls with the perturbation history carried across truncations, which is unbiased for the full-horizon objective from short windows.
  In evosax.
- **Dreamer v1 to v3** (Hafner et al., [2301.04104](https://arxiv.org/abs/2301.04104)).
  The actor is trained by backpropagating lambda-returns through 15 imagined steps; v2 finds pure analytic gradients best for continuous actions; v3 adds symlog inputs, a two-hot critic head, percentile return normalization, and critic regularization toward its own EMA weights instead of a hard target.
- **TD-MPC2** (Hansen et al., ICLR 2024).
  One-step gradient of a five-head Q ensemble through the encoder; the extreme short-horizon point.
- **SVG(k)** (Heess et al., 2015) names the family: k steps through the model, then the critic; SVG(infinity) is BPTT, SVG(0) is the deterministic policy gradient.
- **MAGE** (D'Oro and Jaskowski, NeurIPS 2020, [2004.14309](https://arxiv.org/abs/2004.14309)).
  Train the critic so that the gradient of the TD error is small, since the actor only consumes the critic's gradient.
  **Lipschitz-regularized critics** (2024, [2404.13879](https://arxiv.org/abs/2404.13879)) and **PAVE** (2026, [2601.22970](https://arxiv.org/abs/2601.22970)) show that actor sensitivity is set by the critic's gradient field.
  All untested outside MuJoCo.
- **GI-PPO** (Son et al., NeurIPS 2023, [2312.08710](https://arxiv.org/abs/2312.08710)) and **AGPO** (Gao et al., ICML 2024).
  Hybrids of analytic and likelihood-ratio gradients with a trust signal from the analytic gradient's variance.
- **List et al.** ([2402.12971](https://arxiv.org/abs/2402.12971)) and **Mikhaeil et al.** (NeurIPS 2022, [2110.07238](https://arxiv.org/abs/2110.07238)).
  On chaotic PDEs and RNNs, gradients through long unrolls always diverge, tied to the Lyapunov spectrum; unrolling without temporal gradients recovers most of the benefit, and sparse teacher forcing at a period set by the largest exponent is the fix.
- **Truncation corrections**: ARTBP (Tallec and Ollivier, 2017) makes truncated BPTT unbiased with random window lengths; burn-in steps before the window cut truncation error (Schiller et al., 2026, [2602.10911](https://arxiv.org/abs/2602.10911)).
- **Without BPTT**: synthetic gradients (Jaderberg et al., 2017) are what a critic at the truncation boundary already is; RTRL is exact and cheap only for diagonal recurrence (Zucchet et al., NeurIPS 2023; Irie et al., ICLR 2024); SnAp and e-prop keep only the self path and would drop the spatial propagation an NCA depends on; forward gradients scale with parameter count (Ren et al., ICLR 2023).
  None is a replacement for a dense convolutional recurrence over 128 steps.
- **Spatial credit with a shared policy**: COMA (Foerster et al., 2018) and difference rewards give per-agent counterfactual credit with a score-function gradient.
  With a differentiable automaton, backpropagation through the tied rule already gives exact per-cell credit.
  A critic that outputs a per-cell value map is unpublished.

## The NCA frontier, 2023 to 2026

What is stuck, in the authors' own words.

- **Growing NCA** (Mordvintsev et al., Distill 2020).
  Full BPTT over 64 to 96 steps, final-step loss, sample pool, damage, per-variable gradient normalization added against "sudden jumps of the loss value in the later stages of the training".
  Without the pool the rule "only learns growth dynamics within the training window".
- **Stability and geometry of attractors in NCA** (Kvalsund and Stovold, 2026, [2604.12720](https://arxiv.org/abs/2604.12720)).
  On the deterministic gecko (update probability one), the finite-difference Lyapunov spectrum over 10,000 steps has no positive exponent: seed one is a limit cycle of period about 5.5 steps with a largest exponent of 0.0, the other two seeds are tori with largest exponents of -0.005 and -0.009.
  The attractor forms by epoch 2,000 of training and large perturbations can land in a secondary mode.
  So the trained gecko is not chaotic; whether the rule is chaotic earlier in training, or with cell dropout on, is what E2 measures.
  The gecko's learned attractor is periodic or quasi-periodic rather than a fixed point, forms early in training, and large perturbations push it into secondary modes.
  The evidence that the chaos regime of Metz et al. is real for NCAs.
- **Identity increases stability of NCA** (Stovold, ALIFE 2025, [2508.06389](https://arxiv.org/abs/2508.06389)).
  Organisms grown side by side over 1,000 steps merge or grow tumours; the 1,000-step rollout is the stability benchmark.
- **Non-equilibrium memories with NCA** (Pajouheshgar et al., PRL 2026, [2508.15726](https://arxiv.org/abs/2508.15726)).
  Trains NCAs under noise to retain information for thermodynamically long times; the objective is a lifetime, trained on short noisy windows and verified at long horizons.
- **Differentiable logic CA** (Miotti et al., ALIFE 2025, [2506.04912](https://arxiv.org/abs/2506.04912)).
  Learns Game of Life exactly through a softmax relaxation; "scaling this approach to larger and more complex tasks remains challenging, particularly due to significant numerical instabilities during training".
  The soft-versus-hard gate gap is the discrete-state problem in its purest form.
- **NCA for ARC** (Xu and Miikkulainen, ALIFE 2025, [2506.15746](https://arxiv.org/abs/2506.15746); Guichard et al., [2505.08778](https://arxiv.org/abs/2505.08778)).
  Loss at the final step "can lead to a problem with gradient flow", "small errors tend to compound over time", results are single-trial.
- **From cells to pixels** (Pajouheshgar et al., SIGGRAPH 2026, [2506.22899](https://arxiv.org/abs/2506.22899)) and the **TMLR review** (Spitznagel and Keuper, 2026, [2604.24990](https://arxiv.org/abs/2604.24990)).
  Training memory grows with steps times grid: 14.3 GB per 256x256 sample; "regeneration effects only emerge reliably when sample pooling is enabled".
- **Med-NCA, M3D-NCA, OctreeNCA, LNCA** (Kalkhof, Mukhopadhyay et al., 2023 to 2025; Menta et al., ALIFE 2024).
  Every one is a workaround for VRAM under BPTT: patches, octrees, latent spaces.
- **Neural particle automata** (Kim et al., SIGGRAPH 2026, [2601.16096](https://arxiv.org/abs/2601.16096)).
  Gradients through positions stopped in perception "for stability"; "the learning dynamics are sensitive to hyperparameters".
- **Growing 3D artefacts** (Sudhakaran et al., ALIFE 2021, [2103.08737](https://arxiv.org/abs/2103.08737)).
  The only voxel benchmark, dormant since 2021.
- **Sensorimotor Lenia** (Hamon et al., Science Advances 2025, [2402.10236](https://arxiv.org/abs/2402.10236)).
  Gradient alone failed on "long horizons" and "local minima"; curriculum plus diversity search were required.
- **ASAL** (Kumar et al., 2024, [2412.17799](https://arxiv.org/abs/2412.17799)).
  Runs both regimes side by side: CMA-ES with a CLIP score for Lenia, Boids and Particle Life, truncated BPTT for NCA.
  The natural head-to-head.
- **BraiNCA** (Pio-Lopez, Hartl, Levin, 2026, [2604.01932](https://arxiv.org/abs/2604.01932)).
  Morphogenesis by BPTT, Lunar Lander by REINFORCE, since the control reward has no gradient.
- **HyperNCA, Biomaker CA, Empowered NCA** (Najarro et al. 2022; Randazzo and Mordvintsev 2023; Grasso and Bongard).
  All evolve the NCA because the objective is sparse, non-differentiable, or defined over lifetimes longer than any BPTT window.
- **Surveys**: Hartl, Levin and Pio-Lopez (Physics of Life Reviews 2026, [2509.11131](https://arxiv.org/abs/2509.11131)) list scalability, convergence theory, and discrete states as open; the TMLR review lists memory and the pool as load-bearing.

## Reinforcement learning on cellular automata

The direct prior work, and it is thin.

- **Variengien et al.** (2021, [2106.15240](https://arxiv.org/abs/2106.15240)).
  An NCA with input and output cells trained by deep Q-learning to balance a cart-pole; stable over thousands of steps, regenerates after damage.
  The first NCA trained by RL, as a controller.
- **Guichard** (MSc thesis, TU Delft 2024).
  NCA controllers on cart-pole and Lunar Lander by Double DQN and by neuroevolution, with criticality pre-training; DQN "failed entirely" on Lunar Lander where neuroevolution got modest success.
  The only RL-versus-evolution head-to-head for an NCA, on two small control tasks.
- **BraiNCA** (Pio-Lopez, Hartl, Levin, 2026, [2604.01932](https://arxiv.org/abs/2604.01932)).
  Lunar Lander by REINFORCE with entropy, morphogenesis by BPTT; the only NCA trained with a policy-gradient estimator, and again as a policy for an external environment, not as a rule shaped by reward.
- **HyperNCA** (Najarro et al., 2022, [2204.11674](https://arxiv.org/abs/2204.11674)).
  An NCA of a few hundred parameters grows a policy network's weights, optimized by CMA-ES on the return.
- **CARL** (Cvjetko et al., ALIFE 2026, [2608.26116](https://arxiv.org/abs/2608.26116)).
  A goal-conditioned Double DQN with a dense per-cell Q-field intervenes on Lenia by adding or removing mass; finds solitons, steers them, generalizes across rules.
  RL on a CA with a fixed rule, from the outside; the per-cell Q-field is the nearest published relative of E3's value map.
- **Bagnoli et al.** (2026, [2604.10066](https://arxiv.org/abs/2604.10066)).
  Mobile RL agents flipping cells to steer density: works on the passive rule, fails on majority rules and Life.
  A negative result on local control of active CA.
- **QD-trained NCA generators** (Earle et al., GECCO 2022; Zhang et al., NeurIPS 2023, [2310.18622](https://arxiv.org/abs/2310.18622)).
  CMA-ME and CMA-MAE against non-differentiable playability objectives; small-grid training scales to large grids.
- **Empowered NCA** (Grasso and Bongard, GECCO 2022 and 2023).
  Each cell treated as an agent, empowerment as a secondary evolutionary objective; short time lags help most.
- **Shared-policy distributed agents**: Pathak et al. (NeurIPS 2019) self-assembling limbs with one PPO policy; Tang and Ha's sensory neurons (NeurIPS 2021) by ES; mean-field MARL (Yang et al., ICML 2018) with an Ising lattice where every spin is an agent, the canonical cell-as-agent testbed.
- **Environments**: CARLE (Davis, 2021) and Tape (Pan, 2026) put an agent inside a CA; neither learns the rule.

Gaps the survey states outright: no controlled comparison of policy gradient, BPTT and ES on one NCA growth task against rollout length; no formulation of each cell as an RL agent with a shared stochastic policy; no discrete CA rule learned with a likelihood-ratio estimator rather than a relaxation; no RL that learns a Lenia or Flow Lenia rule rather than steering a fixed one.

## What the survey says about the programme

- The useful analytic window is set by the system's largest Lyapunov exponent, not by the task; the trained gecko sits at zero, so the case for RL on the gecko may rest on cost alone, and E2 must measure the exponent during training and under dropout before the chaos argument is used.
- A per-cell value map is unpublished, and spatial credit is otherwise free through the tied rule; E3 stands.
- With 10^4 parameters, the ES and PES estimators are cheap enough that a per-window inverse-variance mixture with the analytic gradient is a real option, which the robotics literature never gets to exploit; this becomes E2's second half.
- The stated bottlenecks of the field are memory with steps times grid, instability of gradients through recurrence, and objectives that are discrete or non-local in time; E4, E5 and E6 are aimed at each.
- Nobody has reported a truncated-BPTT-with-critic trainer on an NCA, and nobody has compared it to ES on the same substrate with the same evaluation; the ASAL split is the closest.
