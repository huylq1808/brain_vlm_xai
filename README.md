# Brain MRI VLM baseline

This baseline takes an **MRI study and draft Findings**, uses frozen MedGemma1.5 to inspect the images, checks a supported T2 signal attribute, and returns revised Findings with decisions and evidence IDs.

```text
Aligned NIfTI volumes + supplied Findings
 → render 12 MRI PNGs + sequence/slice metadata
 → MedGemma1.5 structured observation
 → bounded T2 comparison → keep / correct / flag
 → revised Findings + decisions + unresolved + presenter
```

The notebook does not generate a new draft. An image observation can be shared across draft variants when the image inputs and neutral target match. Editing is limited to the primary finding's T2 signal direction; text outside this scope is preserved and recorded as unassessed. Successful execution or schema validation does not establish clinical correctness.

## Getting started

- [Colab notebook](notebooks/Brain_MRI_Baseline_Colab.ipynb).
- [Source package](releases/baseline_01/brain_mri_baseline_clean_v1.zip) and [package metadata and checksum](releases/baseline_01/package.json).
- [Sample catalog](samples/catalog.json): sample names, draft variants, file sizes, and SHA256 checksums.

The repository contains code, tests, and operating instructions. MRI volumes, sample ZIP payloads, model weights, access tokens, and raw runs are not distributed through Git. Prepare the sample data from sources you are authorized to use. The code, model weights, and datasets are subject to their respective licenses and terms.

## Input

- Four 3D volumes in `.nii` or `.nii.gz` format: T1, post-contrast T1 (`t1c`), T2, and FLAIR.
- The sequences must be aligned, with matching grids and affine transforms after canonicalization. The code checks these conditions but does not perform registration or DICOM conversion.
- Draft Findings are supplied as text. `source` is an authored report; `controlled_*` variants contain artificial changes; `generated08`, where available, is a cached generated draft, not a new generation in the current run.
- Evaluation labels are not used in inference prompts or image selection.

## Running on Google Colab

1. Upload the notebook to Colab and select a GPU runtime. The current profile targets GPUs with approximately 12 GiB of VRAM or more; actual memory requirements depend on the runtime and inputs.
2. Upload the source ZIP and one sample ZIP to this Google Drive folder:

   `MyDrive/research/brain_vlm_xai/experiment/`

3. Reuse the model weights at:

   `MyDrive/research/brain_vlm_xai/assessor_models/medgemma-1.5-4b-it`

   Model: `google/medgemma-1.5-4b-it`, revision `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`. Set `DOWNLOAD_MISSING=True` to download missing files. Access to the gated model must be authorized. Store your token in Colab Secrets as `HF_TOKEN`; do not put it in the notebook source.
4. In code cell 1, set `MODE="run"`, `SAMPLE_NAME="sample_001"`, `DRAFT_VARIANTS=["source","controlled_low"]`, and `RESUME_RUN=""`. Run code cells 1 through 6.

| Code cell | Action | Loads the model? |
|---|---|---|
| 1 | Mount Drive and configure the run | No |
| 2 | Verify checksums, extract ZIPs, and prepare the worker environment | No |
| 3 | Render MRI PNGs, extract claims, and prepare the prompt and job | No |
| 4 | Load MedGemma and run inference for pending jobs | Yes, when needed |
| 5 | Apply acceptance checks, compare observations, and edit exact text spans | No |
| 6 | Display the saved results in an HTML presenter | No |

Source ZIP SHA256: `1ec15cbbf2cf7f367acb1ff9d1e8bf7283f9bd10a5db6a16457eecbb4cd8834e`. Use the notebook and source ZIP from the same version. Create a new run when the source, configuration, or inputs change. Code cell 4 reuses validated cached calls when available; there is no synthetic model fallback or repeated automatic retry.

## Samples

Use sample ZIPs whose checksums match the catalog. Their default local location is `outputs/samples/baseline_demo_v1/`. Upload the selected ZIP to the Drive folder listed above.

| Sample | Draft variants |
|---|---|
| sample_001 | source, controlled_low, controlled_intermediate, generated08 |
| sample_002 | source, controlled_intermediate, controlled_high, generated08 |
| sample_003 | source, controlled_low, generated08 |
| sample_004 | source, controlled_low, controlled_high |

These are four meningioma (MEN) development cases with pathology, not healthy controls or a clinical benchmark. When changing samples, select draft variants listed for that sample and set `RESUME_RUN=""`.

## After a runtime disconnect

- **Read saved results:** Set `MODE="read"` and point `READ_RUN` to the saved run folder on Drive. Run code cells 1 through 6. This requires the source ZIP and raw run, but no sample ZIP, GPU, or weights. Code cells 3–5 skip inference and editing.
- **Resume the same run:** Set `MODE="run"` and `RESUME_RUN`, keeping the source, sample, draft variants, and configuration unchanged. Run code cells 1 through 6; completed calls are reused.
- Interrupted or failed calls retain their records and logs for inspection. They are not automatically retried until a preferred answer appears.

## Following the outputs

| Artifact | Contents |
|---|---|
| config.json, identity.json, source_snapshot.zip | Configuration and source/input identity |
| images/, evidence.json | Actual model-input PNGs; evidence IDs, sequences, slices, and normalization |
| requests/, jobs.json, prepared.json | Drafts, claims, editable spans, neutral task, prompt, and schema |
| model_load.json, model_config.json | Model revision, weight inventory, runtime profile, and resources |
| observations/ | Raw responses, stop/schema/contract status, actual tensor metadata, and resources |
| gates/ | Raw and effective observations, with reasons for accepting or deferring |
| studies/ | Assessment, policy, result, decisions.csv, revised_findings.txt, unresolved.json |
| summary.json, presentation.html | Execution summary and HTML presenter |

Code cell 6 saves its review outside the raw run folder. Inspect the actual images first, then the draft and claim, raw observation, acceptance checks, and text changes. A valid evidence ID does not establish correct lesion or reference-tissue identification. Flagged or unassessed text has not been confirmed correct.

RAM/DRAM refers to system memory; VRAM refers to GPU memory. Weight size is not the same as peak inference memory. Consult the model-load and observation resource records for each run. Sampled memory peaks are lower bounds.

For a review of a run, provide the complete run folder as a ZIP, including PNGs, observations, model metadata, and study outputs. The HTML file alone is insufficient to recheck the results.

## Local tools

Run local commands from the repository root. Install the preparation dependencies and set `PYTHONPATH` to `src`.

PowerShell:

```powershell
python -m pip install -r src/requirements.txt
$env:PYTHONPATH = (Resolve-Path src).Path
python -m brain_vlm present RUN --output NEW_REVIEW_FOLDER
```

Bash:

```bash
python -m pip install -r src/requirements.txt
export PYTHONPATH="$PWD/src"
python -m brain_vlm present RUN --output NEW_REVIEW_FOLDER
```

Replace `RUN` with a saved run folder and `NEW_REVIEW_FOLDER` with a separate output directory. The reader does not load model weights.

Offline evaluation uses labels only after inference:

```text
python -m brain_vlm evaluate RUN --output NEW_EVALUATION_FOLDER --labels LABELS.json
```

Reader and evaluation outputs must be outside the raw run folder. Agreement with a reference does not by itself establish clinical accuracy.

## Code

```text
README.md
notebooks/Brain_MRI_Baseline_Colab.ipynb
src/brain_vlm/
  baseline/pretrained.py       prepare / infer / finalize
  baseline/claims.py           bounded claim mapping
  baseline/vlm/loader.py       MedGemma loading and input tracing
  baseline/vlm/models.py       snapshot registry / encoder export
  prima/                      reusable PRIMA utilities
  refinement/                 model adapter, contracts, gates, editor, resources
  viewer.py                   saved-run reader
  runtime.py                  worker setup / logs
src/tests/
src/requirements.txt
tools/build_colab_package.py
samples/catalog.json
releases/baseline_01/          source-only ZIP and package metadata
```

The baseline notebook does not load PRIMA. Optional PRIMA encoder extraction is available in a separate environment with compatible dependencies; do not use the MedGemma worker for this step:

```text
python -m brain_vlm extract-prima --model-root MODEL_ROOT --prima-root PRIMA_SOURCE --compact-root COMPACT_ROOT --no-download --record EXPORT_RECORD.json
```

`MODEL_ROOT` contains `test/trained_models`; `COMPACT_ROOT` defaults to `MODEL_ROOT/prima_encoders_v1` when omitted. A valid existing bundle is reused. Remove `--no-download` to allow missing files to be downloaded. Save the export record outside the weights directory. This is encoder extraction, not student-model training.

## Building and testing

Build the source ZIP and update its checksum in the notebook, README, and package metadata:

```text
python tools/build_colab_package.py
```

Building does not load models or read MRI data. Identical source bytes produce the same ZIP checksum.

Regression tests require the preparation dependencies, PyTorch, and `PYTHONPATH` pointing to `src`:

```text
python -m unittest discover -s src/tests -v
```

Some integration and reader tests require local samples or raw runs and are skipped when those fixtures are unavailable. Software tests do not measure clinical correctness. The notebook is stored without cell outputs or embedded patient images.
