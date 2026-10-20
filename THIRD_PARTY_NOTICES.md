# Third-party notices

DV-Lens uses [Jacobian Lens](https://github.com/anthropics/jacobian-lens),
copyright 2026 Anthropic PBC, under Apache-2.0. The dependency is installed
from commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`; its source is not
vendored here. `resolve_model_parts` uses its Hugging Face layout adapter.
The dependency's license and notices remain in its installed distribution.

PyTorch, NumPy, Transformers, and Hugging Face Hub are external dependencies
and retain their own licenses. Model checkpoints, tokenizers, and reference
corpora are supplied by users and retain their original licenses.

`code/kmeans.py` is the tensor k-means routine used by the DV-Lens development
implementation, extracted without the unrelated compression utilities.
