# Experiments

The programme, as a ranked list of questions.
Each experiment states the question, the setup, the measurement, and the rule that settles it, so that a run either answers it or was not the experiment.
The first two are the foundation and every later one depends on them.

## Why reinforcement learning for a differentiable system

Backpropagation through the developmental rollout is the trainer of every NCA in the library, and it works.
Its cost has three parts, and each is a reason to look elsewhere:

1. Memory and time grow linearly with the developmental length, so a thousand-step morphogenesis, a 3D voxel grid, or a large canvas is out of reach, and `remat` only trades one for the other.
1. The gradient through a long rollout of a nonlinear recurrent system is a product of Jacobians, so its norm grows or vanishes exponentially with the horizon when the dynamics are chaotic, and the empirical gradient stops being a useful estimator well before it stops being computable (Metz et al., 2021).
1. Whatever cannot be differentiated cannot be trained: a discrete state, a hard constraint, a black-box environment the automaton lives in, a score from a frozen model or a human.

An actor-critic replaces the tail of the rollout by a learned value, so cost and gradient norm are set by the horizon rather than by the developmental length, which addresses the first two.
A stochastic policy with a likelihood-ratio gradient addresses the third.
Both are reinforcement learning, and the automaton is the unusual case where every cell is an agent with the same parameters, the world is the automaton itself, and the reward is dense.
The programme is to find out where each pays, against the strongest backpropagation baseline, on tasks where the difference is not cosmetic.

## E1. The horizon curve

**Question.**
Does the quality of the trained automaton depend on the horizon once a critic carries the credit, and where is the crossover with backpropagation?

**Setup.**
The gecko of 63, horizons 1, 2, 4, 8, 16, 32, 64, 128, two seeds each, same training budget in simulated steps.
Two controls: the same horizons with the bootstrap turned off (truncated backpropagation with no critic), and 40 itself.
Equal-footing evaluation for every arm: 32 fresh seeds rolled 256 steps, MSE recorded at every step, reported as the mean over steps 128 to 256 and as the curve.
A third arm once E2 has run: the adaptive horizon of AHAC, where the window ends early when the per-step Jacobian norm of the rollout crosses a threshold and the critic bootstraps from there.

**Measurement.**
Final MSE, wall clock, peak memory, all against the horizon.

**Verdict.**
RL earns its place if the curve is flat down to a horizon well below 128 while the no-critic control degrades, since that is the claim of 63.
If the curve is not flat, the critic is not carrying the credit and E3 comes before anything else.
If the no-critic control is flat too, the task does not need long credit and the gecko is not the benchmark for this programme.

## E2. Chaos and the gradient

**Question.**
Is the NCA rollout in the regime where the backpropagated gradient stops being an estimator, and at what horizon?

**Setup.**
Take checkpoints of 40 along training and, for each, measure the norm of the gradient of the final-step loss with respect to the parameters as a function of the horizon, and the spread of that gradient across the dropout noise (variance over 64 masks at fixed parameters).
This is the measurement of Metz et al. (2021), and the prior is against chaos: Kvalsund and Stovold (2026) find no positive Lyapunov exponent on the trained deterministic gecko, whose attractor is a limit cycle or a torus.
What they did not measure is the exponent along training, before the attractor forms at about epoch 2,000, or with cell dropout on, which is when and how backpropagation is actually run.
Fit the growth rate: this is the finite-time Lyapunov exponent of the trained rule as seen by backpropagation.
Repeat for the RL-trained rule of 63 and for an ES-trained rule of 62, since the three trainers may find rules of different chaoticity.
The second half of the experiment uses the same checkpoints to compare estimators of the same gradient at each horizon: the analytic gradient, an antithetic ES estimator over the 8,768 parameters, and their per-window inverse-variance mixture (PIPPS; Onoda et al., 2026), all against a long-horizon average of the analytic gradient as the reference direction.
The rule has few enough parameters that the ES branch costs a batch of forward rollouts, which is the case the robotics literature never gets to exploit.

**Measurement.**
Gradient norm and signal-to-noise ratio versus horizon, per checkpoint; cosine to the reference direction per estimator and horizon.

**Verdict.**
If the signal-to-noise ratio of the 128-step gradient is below one at any point in training, backpropagation is winning on the gecko by clipping and luck, and the RL estimator has a case on principle.
If it stays above one, the honest conclusion is that the gecko is not chaotic and the argument for RL rests on cost and non-differentiability alone.
If the mixture tracks the reference where the analytic gradient alone does not, it becomes the actor's estimator in every later experiment.
Either way, this is the measurement that says which tasks to pick next.

## E3. Local value functions

**Question.**
Can the critic be local, so that credit is assigned cell by cell, and does locality make it a better critic?

**Setup.**
Three critics for the same actor and horizon: the global CNN of 63, a fully convolutional critic with no pooling that outputs one value per cell whose target is the per-cell lambda-return of the per-cell reward, and a critic that is itself a cellular automaton reading a window of the state.
The per-cell reward is the negative squared RGBA error of that cell, whose mean over the grid is the reward of 63, so the two objectives agree in expectation.

**Measurement.**
E1 at horizon 8 and 32 for each critic, and the critic's explained variance on held-out rollouts.

**Verdict.**
A per-cell critic that matches or beats the global one at short horizon settles the design of every later experiment: the value function of a self-organizing system should itself be self-organizing.
This is the experiment with the most scientific content: a value function carried by the cells is a distributed prediction of the future of the tissue, and nothing in the NCA literature has one.
The nearest relative is the per-cell Q-field of CARL (Cvjetko et al., 2026), which steers a fixed Lenia from outside; here the value belongs to the rule being learned.
Critic details to take from the analytic-gradient literature: the two-hot head and percentile return normalization of DreamerV3, an EMA-regularized or min-of-two critic in place of the hard target copy, and a penalty on the critic's state Jacobian since the actor consumes it over a whole grid (MAGE, PAVE).

## E4. Developmental length is free

**Question.**
With the horizon fixed at the value E1 found, can the automaton be trained for developmental lengths that backpropagation cannot afford?

**Setup.**
Three scalings of the gecko task: 1,024 and 4,096 developmental steps with an objective that asks for the shape to be held over the whole second half, a 128x128 and a 256x256 canvas with a target scaled accordingly, and a 3D voxel target (a 32x32x32 emoji extrusion, or a Minecraft-style structure).
The pool supplies states at every age, so no rollout is ever longer than the horizon.
Backpropagation at full length with `remat` is the baseline until it runs out of memory, and truncated backpropagation without a critic is the second control.

**Measurement.**
MSE over the second half of the developmental length on fresh seeds, peak memory, wall clock to a fixed MSE.

**Verdict.**
This is the demonstration the method exists for: a curve of quality against developmental length that is flat for RL and cut off for backpropagation.
Without it, the programme has a method and no reason.

## E5. Homeostasis as infinite-horizon control

**Question.**
Does the discounted infinite-horizon objective produce regeneration and long-term stability without the damage tricks of the sample pool?

**Setup.**
Training as in 63 but with damage as environment stochasticity: with probability p per rollout, a random disk of the state is zeroed before the rollout begins, and the reward is unchanged.
Compare with 40 with the same damage in its pool (the Distill regeneration recipe) and with 40 without damage.
Evaluate on 4,096-step rollouts with damage applied every 512 steps.

**Measurement.**
MSE over time on the long rollouts, time to recover after each damage, fraction of runs that explode or die.

**Verdict.**
RL's claim is that a critic trained on all ages gives a value gradient toward the target from any state, not only from the seed and the trajectories the pool happened to hold.
If the 40 recipe matches it, regeneration is a pool question and not a credit question.

## E6. Non-differentiable rewards

**Question.**
Can a stochastic cell policy with a likelihood-ratio gradient train an NCA on a reward that has no gradient, and what does it cost against a differentiable proxy?

**Setup.**
Three rewards, on the gecko canvas and grid: a frozen classifier's confidence in the target class (differentiable, but treated as a black box, so the two estimators can be compared on the same task), a structural score with no gradient (number of connected components equals one, alpha mass within a band, a symmetry score), and a discrete-state rule where each cell's next state is sampled from a categorical head and the reward is exact match to a target pattern, the Game of Life target of 50 among them.
The policy is the NCA of 40 with a Gaussian head on the update (or a categorical head for the discrete case), trained by PPO with per-cell actions, a shared per-cell advantage from the E3 critic, and the pool as the replay of ages.

**Measurement.**
Reward on fresh seeds, sample efficiency in simulated steps, and for the classifier reward, the gap between the likelihood-ratio gradient and the analytic gradient at equal steps.

**Verdict.**
The discrete-state result is the one that matters: a rule of bits learned without a relaxation is something the difflogic construction cannot do beyond a single step, and it decides whether RL opens a class of automata rather than a trainer.
No published work learns a discrete CA rule with a likelihood-ratio estimator; the field uses relaxation (difflogic CA, ARC NCA) or evolution (ASAL, QD generators).
Each cell as an agent with a shared stochastic policy is the mean-field MARL setting (Yang et al., 2018) applied to a lattice rule, which is also unpublished.

## E7. The automaton in a world

**Question.**
Can an NCA trained by RL act as a body in an environment whose dynamics are not differentiable?

**Setup.**
A grid world with the NCA as the creature: nutrients that must be reached, walls, a light gradient to climb.
The world steps by hand-written rules with no gradient, the NCA's alpha channel is the body, movement is the body growing on one side and dying on the other, the reward is nutrient gathered.
Sensorimotor Lenia (Hamon et al., 2024) is the differentiable precedent; here the world is a black box, which is the case that matters for artificial life.
The RL-trained NCAs that exist (Variengien et al., 2021; BraiNCA, 2026) are controllers reading input cells and writing output cells; CARL (2026) steers a fixed Lenia from outside; Bagnoli et al. (2026) find local RL agents cannot control an active rule.
None trains the rule of a body that moves.

**Measurement.**
Nutrient collected per episode, body integrity over the episode, transfer to unseen worlds.

**Verdict.**
A creature that navigates is the artificial-life result of the programme, and the only experiment where RL is not competing with backpropagation but replacing something that had no trainer.

## E8. Sparse terminal reward

**Question.**
When only the final state is scored, does the critic's credit assignment beat backpropagation through the whole rollout?

**Setup.**
Self-classifying MNIST (44) with reward equal to accuracy at the last step only, and 1D-ARC (47) with reward equal to exact match.
Backpropagation through the full rollout is the baseline and is expected to be strong, since both are short.
Lengthen both: 200 steps for MNIST with the digit redrawn mid-rollout, and ARC tasks of length 96 with a 256-step budget.

**Measurement.**
Accuracy on held-out digits and tasks against training budget.

**Verdict.**
This is the experiment most likely to fail, and worth running for that reason: if RL never beats backpropagation when the objective is terminal and the horizon is short, the boundary of its usefulness is drawn.

## Order and budget

E1 and E2 are cheap (an afternoon on one GPU each) and settle whether the gecko is the right task.
E3 is the intellectual centre and depends on E1.
E4 is the demonstration and depends on E1 and E3.
E5 through E8 are independent of one another and each is a paper section.
Every run reports the equal-footing evaluation of E1, and every claim about quality is judged on the rendered rollouts, not the MSE alone.

## Standing risks

- **The critic as a leaky objective.**
  The actor differentiates through the critic, so the actor can find states the critic overvalues; softplus bounds and the target copy limit this but do not remove it.
  Track the gap between the critic's prediction and the realized return on the actor's own states.
- **The pool as a curriculum.**
  With a critic, the pool is a replay buffer of developmental ages, and its composition decides what the critic sees; 16 resets per batch was tuned for horizon 64 and will not transfer to horizon 8.
- **Cell dropout as policy noise.**
  Dropout makes the system stochastic, which the analytic gradient averages over but the likelihood-ratio gradient could exploit; E6 should try dropout as the only source of stochasticity before adding a Gaussian head.
- **The measurement.**
  MSE against one emoji rewards a blob at the right place; the eye decides whether there are legs and eyes, and the rendered 256-step rollout is part of every result.
