# 1 h bitwise regression receipts

Each file compares every word of a 1 h forecast's `wrfout` (one file per
domain) written by the Metal backend at one revision with the one written at
the revision before it (`tools/metal/wrfout_bitwise.py`): r12 vs r8 (the
reference), r13 vs r12, r14 vs r13, run after the translator, runtime and
array-layer changes of 2026-11-18.  `differing_words` is 0 in every file:
142 775 319 words per run (46 287 130 in d01, 96 488 189 in d02).
