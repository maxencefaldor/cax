# Open questions

- **Does time-to-transition scale with interactions across soup sizes?**
  The paper's drift in the first ~1000 epochs is per epoch, takeover is faster in small soups, and the appearance hazard is per interaction.
  Data: cubff at N = 8192 (4 seeds running) vs the paper's N = 2^17 distribution.
  If the per-interaction hazard differs, reduced-size reproductions need a correction before any Phase 2 comparison at reduced size.
- **Seeded takeover rate vs soup size.**
  cubff at N = 8192 gives ~15% (66 runs so far); the paper reports 22% at N = 2^17.
  Full-size cubff runs in progress.
  If the rate is size-dependent, it is a takeover-phase effect and belongs to the Phase 3 discussion.
- **What does the survivor / jump profile look like on an evolved soup?**
  The fast-path buffers are tuned on random soups.
  Post-transition soups keep most tapes alive and jumping; the fallback path will fire and epochs will cost ~20× more.
  Measure on a transitioned checkpoint and decide whether a second compaction stage is worth it.
- **Phase 2 metric.**
  Programs tested per replicator found (2026 paper), under uniform and under the CUST distribution, using the ported detector.
  Costs 65 executions per program; feasible for ~1e5–1e6 programs per variant on CPU with the fast path.
  This is the experiment to run *before* choosing between direction-flip and cyclic brackets.
- **Totality changes the compute profile.**
  A total machine keeps every tape alive for the whole budget: the survivor compaction stops helping.
  Phase 2 variants need their own speed measurement, and "does nothing" needs redefining as "writes nothing".
- **Which zero?**
  In `bff_selfmove` byte 0 is an instruction.
  Whether the loop-test value should be an instruction, inert, or the most common byte is a design choice with measurable consequences on random-tape behaviour.
  Make the map a parameter and measure.
- **Phase 3 conflicts.**
  Encode (per-step random priority, value) and scatter-max.
  Decide whether priority is per thread per step or per cell per step; the former is cheaper.
