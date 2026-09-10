# CUDA

`ssv_cuda.cu` implements HMMER 3.4's public signed-byte SSV filter. The current
adapter accepts the optimized profile and digital sequences already held by
Astra; it does not load biological files or replace Astra's search pipeline.

`SequenceBatch` packs and uploads targets once, then reuses its CUDA allocations
across profiles. `cpu_candidates(profile, F1)` applies HMMER's exact F1 gate and
returns indexes that direct SSV cannot safely reject. Fallbacks, overflows,
empty targets, and invalid parameters are always retained. `filter_ssv`
remains available for score diagnostics. The corresponding `*_many` methods
compact HMMER's striped scores into GPU-native `[model][residue]` rows and
evaluate a profile batch in one two-dimensional CUDA launch. The single-profile
path keeps HMMER's striped layout. The fused batched gate reuses exact
host-computed length terms, evaluates the binary32 cutoff on-device, and copies
one candidate bit per pair instead of a full result. If a cutoff cannot be
proven safe, the whole profile row is retained for the CPU pipeline.

`load_pressed_profiles` verifies that each protein HMM and pressed optimized
profile agree before either can enter a provenance-bound `CandidateBatch`.
Candidate rows own immutable target and CSR state. Searches require a standard
canonical-background PyHMMER `Pipeline` with matching F1, exclusively owned by
the calling worker until the search returns.
