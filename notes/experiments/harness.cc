// Oracle harness: read 128-byte tapes from stdin, run cubff's Bff::Evaluate on each
// for `steps` steps, write back each final tape followed by the uint32 op count.
// Build twice: with -DBFF_HEADS (language "bff") and without ("bff_noheads").
#include "bff.inc.h"
namespace { const char *Bff::name() { return "harness"; } }
int main(int argc, char **argv) {
  size_t steps = argc > 1 ? atol(argv[1]) : 8192;
  uint8_t tape[2 * kSingleTapeSize];
  while (fread(tape, 1, sizeof(tape), stdin) == sizeof(tape)) {
    uint32_t ops = Bff::Evaluate(tape, steps, false);
    fwrite(tape, 1, sizeof(tape), stdout);
    fwrite(&ops, sizeof(ops), 1, stdout);
  }
  return 0;
}
