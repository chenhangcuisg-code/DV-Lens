# Implementation notes

## Estimator

The dense estimator fits module-output to final-block-output Jacobians using
batched reverse-mode products. For each reference sequence it applies output
cotangents at positions `skip_first` through `sequence_length - 2`, averages
the returned gradients at those positions, and then averages across usable
prompts. This is the implemented causal sequence-aggregated VJP estimator;
it does not materialize every position-to-position Jacobian block.

The dense path estimates the full hidden-by-module-output matrix of that
estimator. The sketch path applies fixed output-metric cotangents first and
retains a projected matrix. Both target the final decoder block output.
The readout applies the final norm's learned scale, but does not differentiate
through the final norm's input-dependent normalization, biases, or logit
soft-capping. "Dense" describes the stored Jacobian representation, not an
exact nonlinear or finite-edit response of the language model.

## Geometry and scoring

The output metric is `H = Gamma W_U.T W_U Gamma`. Dense coordinates use its
eigendecomposition and omit an orthogonal left factor. All coordinates in
one matrix are divided by the same maximum square-root eigenvalue. This
preserves their relative Euclidean distances. The sketch uses frozen Gaussian
rows with variance `1 / sketch_dim` over these coordinates.

Columns are **not individually normalized** before clustering. This is the
implemented magnitude-sensitive geometry. Per-column unit normalization
would define a different partition. k-means uses random distinct data points
for initialization, empty-cluster reseeding, seed 0 and 12 iterations by
default. The effective cluster count is `min(k, number_of_columns // 2)`;
matrices with fewer than four columns cannot produce this score.

DV-Complexity is raw-weight relative SSE under fixed output-space labels.
The denominator is the uncentered raw-weight squared norm. The Joint score
is the unweighted mean across matrices. Output-coordinate SSE and the hybrid
FFN-only path are separate diagnostics and should not be substituted for
the Joint score.

## Module coverage

Projection discovery recognizes common linear names such as `q_proj`,
`k_proj`, `v_proj`, `o_proj`, `down_proj`, and corresponding architecture
aliases. FFN denotes the residual-writing down projection. Fused QKV
projections remain a single native `qkv` matrix in this release; the code
does not impute independent Q/K/V scores. Non-Linear projection classes,
including GPT-2-style Conv1D, require an adapter and are not covered merely
because the layout resolver recognizes that model family.

The final decoder layer is excluded from target selection. Module-set
validation requires at least one supported attention projection and one
FFN down projection; it is not a certificate that every architecture has
five separate supported matrices. Inspect `matrix_records` for actual
coverage. Model loading uses the pinned Jacobian Lens adapter. General
architecture coverage is not asserted by the CPU smoke test.

## Caches and reproducibility

The cache manifest includes the model identifier/path, selected layers,
corpus SHA-256, prompt count, token-position settings, precision, and fitting
settings. Intermediate sums and final tensors are saved atomically. Complete
fits are reused and in-progress fits can resume.

The manifest does not hash checkpoint weights or tokenizer files and does
not pin a remote model revision. Use an immutable local checkpoint snapshot
for reproducibility. Start with a new cache root after changing checkpoint
contents, tokenizer files, adapter behavior, or the implementation. Do not
copy tensors between manifests. The selected-column exporter verifies the
model identifier/path, but cannot prove that files at that path are unchanged.
Treat tensor caches as trusted local artifacts.

Fit output is append-only JSONL. Precision, device kernels, sketch dimension,
reference corpus, and selected modules can affect the resulting partitions.
Record the environment and actual `matrix_records` when comparing scores.

## Release validation

The numerical suite compares dense output distances with explicit vocabulary
projection, checks the raw-weight score against a hand-computed partition,
and compares dense and joint-sketch VJPs with an analytic linear downstream
map. The offline Hugging Face smoke test creates a tiny random Llama and
checks both fit CLIs and selected-column export through the pinned adapter.
These checks establish implementation behavior on small CPU cases. They
do not establish numerical equivalence of sketches to dense fits on full
models, full architecture support, or reproduction of benchmark results.
