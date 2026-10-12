// r4d_ar_oneshot_2rank_wht6.hip: translation unit for r4d_ar_oneshot_2rank_wht6, the all-reduce whose wire
// payload is Walsh-Hadamard rotated and quantised to 6 bits. The device code and the launcher live
// in r4d_ar_oneshot_2rank_wht6.h so the same source also compiles standalone (no Python, no r4d); this file gives
// them the library's C ABI and reports the packing constants the caller needs to size its scratch.
#include "r4d.h"
#include "r4d_ar_oneshot_2rank_wht6.h"

void r4d_ar_oneshot_2rank_wht6(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                               long seq_ctrs, long loc_pack, long slot_stride_bytes,
                               long scale_off_bytes, long inp, long out, long n_elem, long dtype,
                               long stream, long nblocks, long nthreads, long drain, long acq) {
  r4d_ar_wht6_launch(peer_scratch, my_scratch, peer_flags, my_flags, seq_ctrs, loc_pack,
                     slot_stride_bytes, scale_off_bytes, inp, out, n_elem, dtype, stream, nblocks,
                     nthreads, drain, acq);
}

// Block cap of both all-reduce kernels (the flag array is sized from it).
int r4d_ar_max_blocks(void) { return R4D_AR_MAX_BLOCKS; }

// The rotated-6-bit wire format, which the caller needs to size its packed scratch.
void r4d_ar_wht6_dims(int* group, int* bits, int* chunk_elems) {
  if (group)       *group       = GROUP;
  if (bits)        *bits        = BITS;
  if (chunk_elems) *chunk_elems = CHUNK_ELEMS;
}
