# Release scope

This public repository contains the main DV-Lens method and the minimum
supporting material needed to run and inspect it.

## Included

- Native projection discovery for supported `torch.nn.Linear` modules.
- Dense module-output downstream Jacobians.
- Output-metric sketches with joint layer-group VJPs.
- Signed vocabulary readout for selected columns of a dense fit.
- Seeded output-space grouping and raw-weight DV-Complexity.
- Portable Python entry points, examples, numerical checks, dependency
  declarations, license, and implementation documentation.

## Excluded

- Intervention experiments, benchmark/evaluation suites, cohorts, and results.
- Paper sources, figure/table pipelines, and research working notes.
- Reference corpora, weights, tokenizers, caches, and raw run artifacts.
- Server-specific paths, schedulers, launch queues, and credentials.
- Development snapshots, backup files, and private Git history.

Only files in the [release inventory](../RELEASE_MANIFEST.tsv) are published.
The inventory lists each payload file's size and SHA-256; it excludes itself.
The source archive is kept separate. Adding an excluded directory to a
development workspace does not expand the public release scope.
