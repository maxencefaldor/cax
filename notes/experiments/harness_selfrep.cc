// Oracle for CheckSelfRep: stdin = 64-byte programs; argv[1] = seed; stdout per program:
// the 13 noise tapes (13*64 bytes, exactly as CheckSelfRep generates them) then uint32 score.
#include "bff.inc.h"
namespace { const char *Bff::name() { return "harness"; } }
int main(int argc, char **argv) {
  size_t seed = argc > 1 ? atol(argv[1]) : 0;
  std::vector<uint8_t> progs;
  uint8_t buf[kSingleTapeSize];
  while (fread(buf, 1, sizeof(buf), stdin) == sizeof(buf)) progs.insert(progs.end(), buf, buf + sizeof(buf));
  size_t n = progs.size() / kSingleTapeSize;
  std::vector<size_t> result(n + 1);
  for (size_t index = 0; index < n; index++) {
    IndexThreadLocal() = index;
    CheckSelfRep<Bff>(progs.data(), seed, n, result.data(), false);
    // regenerate the noise exactly as CheckSelfRep does
    uint64_t local_seed = SplitMix64(n * seed + index);
    for (size_t i = 0; i < 13; i++) {
      uint8_t noise[kSingleTapeSize];
      for (int j = 0; j < kSingleTapeSize; j++)
        noise[j] = SplitMix64(local_seed ^ SplitMix64((i + 1) * kSingleTapeSize + j)) % 256;
      fwrite(noise, 1, kSingleTapeSize, stdout);
    }
    uint32_t score = result[index];
    fwrite(&score, sizeof(score), 1, stdout);
  }
  return 0;
}
