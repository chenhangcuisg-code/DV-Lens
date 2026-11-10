# DV-Lens / BARQ figure and experiment provenance

Last checked: 2026-11-10 (Asia/Singapore).

This record maps the current paper figures to the local plotting/layout workspaces, Overleaf projects, and the shared server workspace. The public method repositories intentionally exclude private experiment dumps, model checkpoints, raw logs, and scheduler launchers.

## Public repositories

- DV-Lens: https://github.com/chenhangcuisg-code/DV-Lens
- BARQ: https://github.com/chenhangcuisg-code/BARQ
- DV-Lens paper: https://arxiv.org/abs/2610.04489
- BARQ paper: https://arxiv.org/abs/2610.04490

## Overleaf source of truth

| Project | MCP local mirror | Remote project | Main source | Figure assets |
|---|---|---|---|---|
| DV-Lens | `C:\Users\CCH\overleaf_dvl` | `DV-Lens-cvpr2027` | `manuscript.tex` | `figures/`, `paper_figures/` |
| BARQ | `C:\Users\CCH\overleaf_barq` | `BARQ-cvpr2027` | `main.tex` | `figures/` |

The mirrors were pulled from the remote projects on 2026-11-10 and checked against the sync manifests: DV-Lens 168 managed files and BARQ 119 managed files. Build/authentication/metadata files are local-only and excluded from upload.

## Local figure-generation and layout workspaces

### DV-Lens

- Main audit workspace: `C:\Users\CCH\Documents\ChatGPT\next\cvpr_conversion_20261103\`
- Current manuscript workspace: `online_work\dvl_cvpr\`
- Figure assets: `online_work\dvl_cvpr\figures\`
- LaTeX figure assembly: `online_work\dvl_cvpr\paper_figures\`
- Recorded figure/layout scripts include `dvl_fig3b_restructure.py`, `apply_dvl_fig3b_layout.py`, `apply_dvl_fig3b_reviewfix.py`, `apply_table3_figure.py`, and `black_text_and_table3.py`.
- These are private audit/layout scripts. They are separate from the public DV-Lens method implementation.

### BARQ

- Main audit workspace: `C:\Users\CCH\Documents\ChatGPT\next\cvpr_conversion_20261103\`
- Current manuscript workspace: `online_work\barq_cvpr\`
- Figure source snapshot: `visual_style_20261104\baseline\barq\figures\`
- Primary plotting script: `visual_style_20261104\baseline\barq\figures\make_figures.py`
- Shared plotting style: `visual_style_20261104\baseline\barq\figures\sci_style.py`
- Current assets copied into Overleaf: `online_work\barq_cvpr\figures\`
- Public implementation mirror: `C:\Users\CCH\BARQ_repo\`, with quantization code under `code\` and portable commands under `scripts\`.

## Server workspace

Shared development root:

`chenhang-cur:/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/LA/chenhangcui/`

The exact server-side DV-Lens/BARQ result subdirectory is intentionally left as **to be filled from a successful server listing** rather than guessed. The local audit paths and Overleaf mappings above are verified. The public BARQ README records Python 3.10, PyTorch 2.4.1 + CUDA 12.1, Transformers 4.53.2, and datasets 2.20.0; the DV-Lens README records Python 3.10+ and the CPU validation environment.

## Figure-to-source rule

When a figure changes, update the plotting/layout script or LaTeX assembly source in the local audit workspace first, regenerate the asset, record the changed path and hash, then copy only the verified asset/source into the matching Overleaf mirror. Do not treat a PDF in Overleaf as the plotting source. Do not upload private audit workspaces or old backup directories.

## Synchronization note

The Overleaf MCP mirrors the remote project and keeps `.overleaf-mcp-state.json`. Push requires a recent pull baseline, checks for remote changes, skips stale local files, and synchronizes recursively. This prevents old plot assets or files from another project being uploaded.
