# Brain MRI VLM baseline

Baseline nhận **MRI study + draft Findings**, dùng frozen MedGemma1.5 để quan sát ảnh, so sánh thuộc tính T2 trong phạm vi hỗ trợ, rồi xuất Findings đã chỉnh cùng quyết định và evidence IDs.

```text
Aligned NIfTI volumes + supplied Findings
 → render 12 MRI PNGs + sequence/slice metadata
 → MedGemma1.5 structured observation
 → bounded T2 comparison → keep / correct / flag
 → revised Findings + decisions + unresolved + presenter
```

Notebook không sinh draft mới. Một observation có thể dùng chung cho nhiều draft variants khi inputs/neutral target giống nhau. Baseline chỉ sửa primary-finding T2 direction; phần ngoài phạm vi được giữ và ghi unassessed. Chạy thành công hoặc đúng schema không chứng minh kết quả đúng y khoa.

## Các file để chạy

- [Notebook duy nhất](notebooks/Brain_MRI_Baseline_Colab.ipynb).
- [Source package](releases/baseline_01/brain_mri_baseline_clean_v1.zip) và [hash/metadata](releases/baseline_01/package.json).
- [Sample catalog](samples/catalog.json): tên sample, draft variants, kích thước và SHA256.

Repository chứa code, tests và hướng dẫn. MRI/sample ZIP payloads, model weights, tokens và raw runs không được phân phối trong Git. Người chạy chuẩn bị sample từ nguồn được phép sử dụng. Code và model/dataset có điều kiện sử dụng riêng của từng nguồn.

## Input

- Bốn volume `.nii`/`.nii.gz`: T1, T1c, T2 và FLAIR, mỗi volume3D.
- Các sequence phải aligned, cùng grid/affine sau canonicalization. Code kiểm tra nhưng không tự registration hoặc chuyển DICOM.
- Draft là văn bản Findings. `source` là authored report; `controlled_*` chứa thay đổi nhân tạo; `generated08` nếu có là draft đã cache, không sinh mới trong lần chạy này.
- Evaluation labels không được đưa vào prompt hoặc chọn ảnh cho inference.

## Chạy trên Google Colab

1. Upload notebook lên Colab, chọn GPU có khoảng12GiB VRAM trở lên cho profile hiện tại. Tài nguyên thực còn phụ thuộc runtime/input.
2. Upload source ZIP và một sample ZIP vào Drive:

   `MyDrive/research/brain_vlm_xai/experiment/`

3. Reuse weights ở:

   `MyDrive/research/brain_vlm_xai/assessor_models/medgemma-1.5-4b-it`

   Model: `google/medgemma-1.5-4b-it`, revision `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`. `DOWNLOAD_MISSING=True` tải phần thiếu. Model gated cần quyền truy cập; đặt token trong Colab Secrets `HF_TOKEN`, không ghi token vào code.
4. Cell1 đặt `MODE="run"`, `SAMPLE_NAME="sample_001"`, `DRAFT_VARIANTS=["source","controlled_low"]`, `RESUME_RUN=""`. Chạy code cells1→6.

| Cell | Việc thực hiện | Load model? |
|---|---|---|
| 1 | Mount Drive, cấu hình run | Không |
| 2 | Kiểm hash/extract ZIP, chuẩn bị worker environment | Không |
| 3 | Render MRI PNGs, tách claims, tạo prompt/job | Không |
| 4 | Load MedGemma và inference cho pending job | Có |
| 5 | Gate/compare và exact-span editing | Không |
| 6 | Đọc kết quả thành HTML presenter | Không |

Source ZIP SHA256: `1ec15cbbf2cf7f367acb1ff9d1e8bf7283f9bd10a5db6a16457eecbb4cd8834e`. Dùng notebook và ZIP cùng phiên bản. Source/config/input đổi thì tạo run mới; không resume một run bằng source khác. Cell4 reuse cache đã kiểm hợp lệ; không có synthetic model fallback hoặc tự retry nhiều lần.

## Samples

Chuẩn bị sample ZIP có hash khớp catalog. Vị trí local mặc định: `outputs/samples/baseline_demo_v1/`. Sau đó upload ZIP cần dùng vào Drive như trên.

| Sample | Draft variants |
|---|---|
| sample_001 | source, controlled_low, controlled_intermediate, generated08 |
| sample_002 | source, controlled_intermediate, controlled_high, generated08 |
| sample_003 | source, controlled_low, generated08 |
| sample_004 | source, controlled_low, controlled_high |

Đây là bốn MEN development cases có bệnh, không phải healthy controls hoặc clinical benchmark. Đổi sample cần chọn variants thực sự có trong catalog và đặt `RESUME_RUN=""`.

## Sau disconnect

- **Đọc lại:** `MODE="read"`, đặt `READ_RUN` là folder run đã lưu trên Drive, chạy1→6. Cần source ZIP và raw run, không cần sample/GPU/weights. Cells3–5 bỏ qua inference và editing.
- **Tiếp tục cùng run:** `MODE="run"`, đặt `RESUME_RUN`, giữ nguyên source/sample/variants/config. Chạy1→6; completed calls được reuse.
- Interrupted/failed calls có records/logs để kiểm tra, không tự retry đến khi có output mong muốn.

## Đọc output theo luồng

| Artifact | Nội dung |
|---|---|
| config.json, identity.json, source_snapshot.zip | Config và source/input identity |
| images/, evidence.json | PNG thực vào model; evidence ID, sequence, slice, normalization |
| requests/, jobs.json, prepared.json | Draft/claims/editable spans, neutral task, prompt/schema |
| model_load.json, model_config.json | Model revision, weight inventory, runtime profile/resources |
| observations/ | Raw response, stop/schema/contract status, actual tensor metadata/resources |
| gates/ | Raw/effective observation và lý do accept/defer |
| studies/ | Assessment, policy, result, decisions.csv, revised_findings.txt, unresolved.json |
| summary.json, presentation.html | Tổng hợp execution và presenter |

Cell6 lưu review ngoài raw run. Xem ảnh thực trước, rồi draft/claim, raw observation, gate và phần text đã đổi. Evidence ID đúng chưa chứng minh lesion/reference tissue đúng. Flag/unassessed không có nghĩa câu đã được xác nhận đúng.

RAM/DRAM là system memory; VRAM là GPU memory. Weight bytes không bằng inference peak; xem model-load/observation resource records của từng run. Sampled memory peaks là lower bounds.

Để gửi kết quả, ZIP toàn folder run gồm PNGs, observations, model metadata và study outputs; chỉ HTML không đủ tái kiểm tra.

## Chạy công cụ local

Cài preparation dependencies trong `src/requirements.txt` và đặt `PYTHONPATH` trỏ tới `src`. Trong PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m brain_vlm present RUN --output NEW_REVIEW_FOLDER
```

Offline evaluation, labels chỉ vào sau inference:

```text
python -m brain_vlm evaluate RUN --output NEW_EVALUATION_FOLDER --labels LABELS.json
```

Reader/evaluation output phải nằm ngoài raw run. Reference-relative metrics không tự là clinical accuracy.

## Code

```text
README.md
notebooks/Brain_MRI_Baseline_Colab.ipynb
src/brain_vlm/
  baseline/pretrained.py       prepare / infer / finalize
  baseline/claims.py           bounded claim mapping
  baseline/vlm/loader.py       MedGemma loading và input tracing
  baseline/vlm/models.py       snapshot registry / encoder export
  prima/                      reusable PRIMA utilities
  refinement/                 model adapter, contracts, gates, editor, resources
  viewer.py                   saved-run reader
  runtime.py                  worker setup / logs
src/tests/
src/requirements.txt
tools/build_colab_package.py
samples/catalog.json
releases/baseline_01/          source-only ZIP và package metadata
```

PRIMA không được load trong notebook baseline này. Optional encoder extraction vẫn có trong PRIMA environment riêng với dependencies tương thích, không dùng chung MedGemma worker:

```text
python -m brain_vlm extract-prima --model-root MODEL_ROOT --prima-root PRIMA_SOURCE --compact-root COMPACT_ROOT --no-download --record EXPORT_RECORD.json
```

`MODEL_ROOT` chứa `test/trained_models`; `COMPACT_ROOT` mặc định là `MODEL_ROOT/prima_encoders_v1`. Reuse bundle hợp lệ; bỏ `--no-download` để tải phần thiếu. Export record đặt ngoài weights folder. Đây là encoder extraction, không student training.

## Build và kiểm tra

Đóng gói source, cập nhật hash pin trong notebook/README và package metadata:

```text
python tools/build_colab_package.py
```

Không load model hoặc đọc MRI khi build. Cùng source bytes cho cùng ZIP hash.

Regression tests cần preparation dependencies, PyTorch và `PYTHONPATH=src`:

```text
python -m unittest discover -s src/tests -v
```

Một số integration/reader tests cần samples/raw runs local và sẽ skip nếu thiếu. Software tests không đánh giá độ chính xác y khoa. Notebook được lưu không có cell outputs hoặc embedded patient images.
