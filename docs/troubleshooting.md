# Troubleshooting

## GitHub Release Download Failure

Symptom: the downloader cannot reach GitHub or times out.

Likely cause: network access, proxy settings, or regional restrictions.

Solution: retry with a working network, download the exact Release asset manually, place it at the manifest `download_path`, and rerun the downloader for verification/installation.

## SHA256 Mismatch

Symptom: the downloader reports a hash mismatch.

Likely cause: incomplete or corrupted download.

Solution: remove the affected artifact file or `.part` file and rerun `python scripts/download_release_artifact.py <artifact>`.

## `.tar.gz` Extraction Problems

Symptom: `mini_case` downloads but does not install.

Likely cause: incomplete archive, locked file, or unexpected extraction interruption.

Solution: verify the archive hash with the downloader, ensure `assets/demo_data/` is writable, and rerun the command.

## Windows Locked File

Symptom: a file cannot be overwritten or deleted on Windows.

Likely cause: another process, viewer, Python interpreter, or antivirus scan still holds the file handle.

Solution: close programs using the file and retry. Avoid broad cleanup commands.

## Conda Temporary-File Error

Symptom: `conda run` reports temporary file access errors.

Likely cause: concurrent Conda invocations or Windows temp-file locking.

Solution: run one Conda command at a time and use a clean project-local temp directory when necessary.

## CUDA Unavailable

Symptom: PyTorch reports `False` for `torch.cuda.is_available()`.

Likely cause: CPU-only PyTorch build or driver/CUDA mismatch.

Solution: install a PyTorch build matching your CUDA driver, or run small demos on CPU.

## CUDA Out Of Memory

Symptom: neural training or inference exits with a CUDA OOM error.

Likely cause: GPU memory is insufficient for the chosen batch size or other processes are using memory.

Solution: reduce batch size, close other GPU processes, or run small checks on CPU.

## Missing Checkpoint

Symptom: Demo 3 cannot find a model checkpoint.

Likely cause: Release checkpoint not downloaded and no Demo 2 checkpoint exists.

Solution: run `python scripts/download_release_artifact.py pcr_time` or `python scripts/download_release_artifact.py pcr_spatial`, or provide `--model-path`.

## Incompatible Checkpoint Keys

Symptom: loading a checkpoint raises key mismatch errors.

Likely cause: checkpoint architecture does not match the repository model definition.

Solution: use the Release checkpoint for the selected temporal/spatial version or a checkpoint produced by the current code.

## Missing HDF5 Files

Symptom: a demo reports missing `era5_t2m_YYYY.h5`, `lst_YYYY.h5`, `t_base_advanced_YYYY.h5`, or `cb_t2m_YYYY.h5`.

Likely cause: incomplete data installation or running a downstream demo before upstream products exist.

Solution: download `mini_case`, run the required upstream demo, or use a complete `--data-root`.

## Too Few CatBoost Samples

Symptom: CatBoost training fails or produces unstable metrics.

Likely cause: the tracked `smoke_case` is intentionally tiny.

Solution: use `--dataset mini_case` for public workflow validation.

## Windows Path Escaping

Symptom: backslashes in paths are interpreted unexpectedly.

Likely cause: shell escaping.

Solution: quote paths and prefer repository-relative paths in commands.

## DataLoader Multiprocessing On Windows

Symptom: worker process errors or hangs.

Likely cause: Windows multiprocessing constraints.

Solution: use `--num-workers 0`, as the demos do.

## Existing Output Directory

Symptom: outputs already exist from an earlier run.

Likely cause: demos write deterministic output roots under `outputs/demos/`.

Solution: inspect the existing `summary.json`. Remove only the specific demo output directory if you intentionally want a clean rerun.

## Benchmark Values Differ From Manuscript

Symptom: Demo 6 peak memory or effective FLOPs differ from another run.

Likely cause: peak memory depends on hardware, driver, CUDA, PyTorch, precision, and system conditions. Effective FLOPs also depend on the configured diffusion number of function evaluations.

Solution: compare generated `complexity_results.csv` files and record the exact command-line options used for the run.
