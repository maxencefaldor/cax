# Critique of the BFF-in-CAX brief (written before building)

Date: 2026-09-16.
Sources checked: Agüera y Arcas et al. 2024 (arXiv:2406.19108v2), Knierim et al. 2026 (arXiv:2607.01483v1), cubff at commit f212e84, Wegner's review.
Everything below was verified against those, not recalled from memory.

## 1. There are three different "BFF"s, and the brief conflates them

| Machine | IP start | head0 / head1 start | ops | where |
| --- | --- | --- | --- | --- |
| `bff_noheads` | 0 | 0 / 0 | 10 ASCII ops + null | **Section 2 of the 2024 paper**; the paper says explicitly "to run the BFF variant from Section 2, pass `--lang bff_noheads`" |
| `bff` | 2 | tape[0] % 128 / tape[1] % 128 | same | cubff default in the Python tooling; the 2024 paper only mentions head offsets for the long-tape setting |
| `bff_selfmove` | 2 | from tape | 7 ops on bytes 0-6, one copy `,` that auto-advances head1, no `.`, `{`, `}` | **main variant of the 2026 paper** |

The headline "40% of 1000 runs within 16k epochs" is a `bff_noheads` number.
Phase 1 must implement head initialisation as a flag and reproduce with heads at zero.
The brief's Phase 2 must say which of the three it is being "principled" relative to; the 2026 paper's own variant already dropped one copy op and both head1 moves, so "symmetric heads" is not a new direction, it is the direction the authors already took, and it works.

## 2. Semantics the brief asked me to verify (all from `bff.inc.h` and `common_language.h`)

- Step cap: 8192 (2^13) per pair execution.
  Every step counts, including no-op bytes.
- Loop test: raw byte at head0 equal to 0.
  Any nonzero byte, instruction or not, is "true".
- `[` with zero at head0: scan forward with nesting; unmatched: halt.
  `]` with nonzero at head0: scan backward with nesting; unmatched: halt.
  A bracket whose jump is not taken never scans and never halts.
  Falling off the end (pc reaches 128) halts.
- Heads are masked mod 128 at the start of every step (equivalent to masking on move).
- Mutation: each of the 128 bytes of the concatenated pair is replaced by a uniform random byte with probability 1/4096 = 0.0244% before execution.
  The paper's "0.024%" is this.
- Pairing: a full Fisher-Yates permutation of all 2^17 programs each epoch; pair i is (perm[2i], perm[2i+1]) with perm[2i] first.
  Concatenation order is therefore already random.
  Every program interacts exactly once per epoch.
- The brief's "first half executes first" is only approximately true: the IP starts at 0
  (or 2), and control reaches the second half by running off the end of the first half or
  by a bracket jump.
- Complexity: not zlib.
  brotli quality 2, window 24, on the whole 8 MB soup; reported as "high-order entropy" = H0 (byte Shannon entropy of the soup) minus brotli bits per byte.
  Fig. 6 of the paper calls a run transitioned when this reaches 1; the cubff scripts `time_to_sr.py` and `runit.py` use 3.
  I will report both thresholds.
- Compute: cubff on one CPU core does an epoch of 2^17 programs in 0.13 s on a random soup (measured here, 256 epochs in 33 s, log matches the shipped reference byte for byte).
  That is ~2 µs per interaction, because almost every random tape halts within ~100 steps.

## 3. What is wrong or missing in the brief

**"Emergence probability scales like 256^(-length)"** is the wrong model.
The 2026 paper measures programs tested until the first replicator: 2.9e7 for uniform bytes, 1.7e6 under the byte distribution the soup drifts to, 9.4e4 under a 50/50 op/no-op mix with byte 64 promoted.
The base is not 256, the soup changes it by two orders of magnitude, and what matters is the *count* of replicators at each length under the current byte distribution, not the minimal length.
The right metric for Phase 2 instruction-set comparisons is the 2026 paper's: sample tapes from a stated distribution, run the replication detector, count programs tested per replicator found.
That needs no soup at all and is the cheapest experiment that could change someone's mind about a variant.
I will port cubff's `CheckSelfRep` (13 chains of 5 executions against random partners, per-byte agreement in >3 chains, score = min over halves, threshold 48) as the definition of "replicator".

**Appearance and takeover are different statistics and the brief mixes them.**
The paper's 40% is takeover (compression drop).
The 2026 paper's central result is that appearance is found at least as fast by random mutation and that pairing is what spreads replicators.
So the language sets the appearance hazard; the interaction structure sets takeover.
Phase 2 (instruction set) should be judged on appearance; Phase 3 (geometry) on takeover and on what happens after.
This also means the brief's "Phase 2 must reproduce the 40% before Phase 3 is trusted" is too strong: a principled variant with a different appearance rate is a finding, not a validation failure.

**Compute plan is missing.**
A branch-free scan that pays 8192 steps for every tape is ~100x the work cubff does, because cubff stops each tape when it halts.
1000 runs x 16k epochs x 2^17 programs is out of reach on this machine with either code.
The plan: (a) bit-exact equivalence with cubff on identical inputs, by replaying cubff's SplitMix64 initialisation, permutation and mutation host-side; (b) transitions at reduced soup size in CAX, several seeds, reported in interactions not epochs; (c) cubff itself, which builds and runs here, as the full-scale reference for the transition statistics.
Compaction of halted tapes is not a nicety: without it Phases 2 and 3 are not runnable either.

**Totality removes halting, and halting is doing work.**
In BFF a random tape mostly runs off the end after ~130 steps.
A total, cyclic machine runs every tape for the full budget.
That is an 60x compute change and a semantic one: "does nothing" becomes "writes nothing" rather than "halts".
The brief should treat the step budget as the definition of death and measure how many random tapes write anything.

**Direction-flip brackets lose conditional skipping, not just nesting.**
`[` can no longer jump over a block, so there is no "if", only "bounce".
Instruction semantics themselves are not mirrored, only the IP direction, so a body `.>}` executed forward then backward does copy, h0++, h1++, h1++, h0++, copy: two copies and double moves per bounce.
Whether that makes minimal replicators shorter or longer is exactly what the appearance-rate experiment answers.
It should be run before any 2D work assumes the answer.

**"A single-head machine needs ~30-byte replicators" is unverified.**
`bff_selfmove` has one copy op and finds replicators easily.
Copy-with-two-address-registers is the whole trick that turned BF into a replicator substrate; the number of copy ops is secondary.

**The byte-to-opcode map is a parameter with two effects the brief does not separate:** instruction density (10/256 vs 7/256) and the identity of the zero byte.
In `bff_selfmove` byte 0 is `>`, so the loop-test "false" value is an instruction.
That choice alone changes what random tapes do.
Make the map a parameter and measure.

**Wegner's critique** (the language is engineered so copying is common; replicators are attractors so takeover is inevitable) is right about takeover and about engineering, and the brief already agrees: it treats "what fraction of random strings do something" and "how short the shortest replicator is" as the design levers.
The honest framing is that this programme studies *which substrate properties set the appearance rate and what happens after takeover*, not whether life emerges.
Write that in the docs.

**Post-takeover plateau.**
The 2024 paper does not claim sustained complexity growth in BFF; it shows generations of replicators in Z80 and competition in 2D.
The brief's premise that structure is the lever is plausible and is what Phase 3 tests, but the 2026 paper's result that pairing is a weak search operator is also an argument that geometric interaction (Phase 3) might be a *worse* spreader, which would show up as slower takeover.
Measure takeover time in Phase 3 against the paired soup.

## 4. What I would do differently

1. Phase 1 = the interpreter as a pure function `run(tapes, num_steps) -> tapes`, bit-exact against cubff on cubff's own random inputs, with the soup dynamics (permutation, mutation) as a separate `ComplexSystem` epoch step.
   Head initialisation and the byte-to-opcode map as parameters so `bff`, `bff_noheads`, `bff_perm` are configurations.
1. Port the replication detector to JAX now, not in Phase 2.
   It is the measurement instrument for everything after.
1. Add the 2026 paper's appearance-rate experiment (programs tested per replicator, under uniform and under the CUST distribution) as the first Phase 2 experiment.
   It is cheap and it is the metric on which instruction-set variants should be compared.
1. Report transition time in interactions, not epochs, so soup sizes are comparable.
1. Phase 3 write conflicts: encode (random per-step priority, value) into one integer and scatter with max.
   Deterministic, order-independent, and it does not let scatter's unspecified duplicate handling decide.
