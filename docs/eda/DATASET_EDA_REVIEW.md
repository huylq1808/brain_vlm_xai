# Dataset EDA review: six datasets

**Updated:** 2026-10-06. Reviewed source/code and saved text outputs of all six EDA notebooks, the local RadGenome manifests/QC, current files and official release pages. This revision does not rerun EDA or download datasets. MET remains excluded locally.

This file owns the six completed EDA records and storage inventory. [Additional image–text sources](IMAGE_TEXT_DATASET_RESEARCH_2026-10-06.md) owns the candidate catalog; [protocol 11](../../research_proposal/11_DATASET_TRAINING_EVALUATION_PROTOCOL.md) owns how data enter M0–M10 and training/evaluation. A saved Colab/Kaggle result is distinguished from files available on this workstation.

**Concept:** [Introduction to Intracranial Haemorrhage](https://youtu.be/Kb_wzb7-rvE?si=i5SrPgfKGWdzl5q4)

## Comparison at a glance

| Dataset | Year | Clinical focus | Modality / unit | Size observed in EDA | Labels and lesion masks | Paired text |
|---|---|---|---|---|---|---|
| CT-ICH v1.3.1 | 2020 release | Traumatic intracranial hemorrhage (ICH) | Non-contrast CT / 3D volume and slices | 75 volumes; 2,814 labeled slices | Five hemorrhage subtypes, fracture; **ICH masks** | No |
| RSNA-IHD | 2019 challenge | Acute ICH | Head CT DICOM / image-slice | 4,516,842 label rows; ~752,807 images implied | Five subtypes + `any`; **no masks in reviewed data** | No |
| SLAKE | 2021 | Medical visual question answering (VQA), multiple body regions | CT, MRI, X-ray / 2D image and QA pair | 14,028 QA pairs; 642 images | Answers; semantic masks/boxes in sampled image folders | **Questions and answers**, English/Chinese |
| MR-RATE | 2026 release | Broad brain **and spine** findings | Multi-sequence 3D MRI / series and study | 705,254 series; 98,334 studies | 37 report-derived pathology labels; **no manual lesion masks** | **Radiology reports** |
| ATLAS R3.0 / ISLES'26 | 2026 challenge release | Ischemic stroke lesion segmentation | T1w 3D MRI / subject | 1,453 T1w–mask pairs | **Stroke lesion masks** | No reports or QA |
| RadGenome-Brain MRI | 2025 online; 2026 journal issue | Glioma, meningioma, ischemic stroke, WMH; MET excluded locally | Multi-sequence 3D MRI / sequence and case | 770 local cases; 2,460 volumes; **2,150 volume–report pairs** | Source lesion masks; anatomy pseudo-labels if generated | Case findings/impressions; GPT-4-decomposed modality findings |

## Storage and access — checked 2026-10-06

**GB/TB are decimal; GiB is binary (2³⁰ bytes).** Published repository size, compressed download, extracted files and eligible model subset are different quantities. These are disk sizes, not model RAM/VRAM requirements. Local counts below exclude code/EDA caches unless stated.

| Dataset | Published/hosted size and access | Files currently in this workspace | What the EDA actually established |
|---|---|---|---|
| CT-ICH v1.3.1 | [PhysioNet](https://physionet.org/content/ct-ich/1.3.1/) currently marks **Restricted Access**; remote byte total not visible in this check | ZIP **600.35 MB** (600,348,969 B); ZIP members **2.951 GB / 2.748 GiB** uncompressed. Folder **3.551 GB / 3.307 GiB** including ZIP + extracted content | Local 75 images + 75 masks; header audit of all pairs. Folder overhead does not add patients |
| RSNA-IHD | [Official Kaggle Data Explorer](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data?select=rsna-intracranial-hemorrhage-detection): **458.97 GB**, 874,037 listed files. This is hosted data, not a verified compressed ZIP size; competition terms apply | No RSNA image directory now | CSV EDA **and sampled DICOM reading/visualization on Kaggle**; not full-image QC or verified patient grouping |
| SLAKE | [Author-linked HF release](https://huggingface.co/datasets/BoKelvin/SLAKE/tree/main): **217 MB** total, `imgs.zip` **212 MB**; CC BY 4.0 | No image cache now; extracted byte total not retained/verified | Colab downloaded the image ZIP; 165/165 selected brain image references resolved. Sample masks/boxes inspected |
| MR-RATE native | [HF](https://huggingface.co/datasets/Forithmus/MR-RATE): **8.11 TB** total files (card rounds to 8.1 TB), brain **and spine**; gated with specific terms | No MR-RATE images now; size of earlier 20-study sample was not recorded | Full metadata/report/label EDA plus **20 sampled study ZIPs and 20 series visual/header checks in saved Colab output**. Sample was not brain/train-filtered |
| ATLAS R3.0 raw / ISLES'26 | [Release/challenge](https://isles-26.grand-challenge.org/dataset/); follow its access and raw-data rules | `ATLAS_R3.0_raw.tar.gz`: **6.088 GB / 5.670 GiB** (6,088,023,020 B). No extracted cohort now; uncompressed total unknown | Saved EDA extracted 4,370 files, including 2,906 NIfTI and 1,453 CSV; 1,453 image–mask matches. Sample image QC rather than exhaustive voxel audit |
| RadGenome and source MRI | [HF annotation release](https://huggingface.co/datasets/JiayuLei/RadGenome-Brain_MRI/tree/main) supplies JSON; original MRI/masks have separate source terms | Source payload **49.924 GB / 46.495 GiB** (49,924,028,755 B); raw including annotation/support files **49.926 GB / 46.497 GiB**. Six ZIPs plus WMH files; annotation/support difference about **1.65 MB** | 770 report-linked non-MET cases; all 2,460 selected image volumes read. Full source payload contains many more cases/representations than this subset |

MR-RATE's **17.6 TB coreg**, **12.3 TB atlas** and **415 GB anatomical segmentation** repositories are separate derivatives of the cohort, not included in native 8.11 TB and not extra patients. Start with selected native brain studies, not a full multi-repository download. Sizes come from [the official card](https://huggingface.co/datasets/Forithmus/MR-RATE); access conditions must be retained with the source manifest.

The RadGenome portable development ZIP is about **63 MiB**, containing 10 cases / 26 images / 10 masks / paired Findings. It is not the full 46.5 GiB source cohort. Source masks and reports are annotation/scoring inputs, not visual encoder inputs.

## 1. CT-ICH v1.3.1

- **What it is:** Non-contrast CT of traumatic ICH; five subtypes (IVH, IPH, SAH, EDH, SDH), plus a fracture label.
- **Data:** 75 CT NIfTI volumes, 75 matching-name mask volumes, and 2,814 slice-label rows. The original cohort has 82 patient records; seven have no released CT volume.
- **Key EDA:** 36/75 imaged patients have at least one subtype after slice-to-patient aggregation. Positive slice counts: EDH 173, IPH 73, SDH 56, IVH 24, SAH 18. **25 slices are multi-label.**
- **Use / caution:** Suitable for hemorrhage classification and ICH localization assessment. The reviewed masks do not establish a subtype-to-voxel mapping. The demographics CSV contains an extra header and footer rows; use the cleaned 82-patient table. The original notebook checked one image/mask example; the later header audit is described below.
- **Additional local QC:** A 2026-09-29 read-only audit checked full qform/sform geometry header fields for all 75 image/mask pairs and found no mismatch; visual overlays remain. The slice CSV has a UTF-8 BOM and uses `SliceNumber=1..D`, so convert to zero-based tensor indices explicitly. Sampled mask files store int16 extremes with NIfTI scaling to approximately `0/255`; load scaled values before binarizing. A raw `>0` threshold on stored values is unsafe. All 2,814 CSV rows belong to the 75 imaged patients; 36 patients have a positive ICH subtype.
- **Links:** [Notebook](../../notebooks/01_CT-ICH_EDA.ipynb) · [Official PhysioNet release, 2020](https://physionet.org/content/ct-ich/1.3.1/)

## 2. RSNA Intracranial Hemorrhage Detection

- **What it is:** Acute ICH classification on head CT, with five subtype targets and an `any` target.
- **Data:** Stage-2 training CSV has 4,516,842 label rows and no missing labels. The official format uses six rows per DICOM image, implying ~752,807 images; unique image IDs were not counted separately in this notebook.
- **Key EDA:** Positive image-label rows: `any` 107,933; subdural 47,166; intraparenchymal 36,118; subarachnoid 35,675; intraventricular 26,205; epidural 3,145. Epidural is rare relative to other subtypes.
- **Use / caution:** Strong slice/image classification resource; no lesion masks or paired reports in the reviewed data. The notebook's derived `PatientID` column is actually an **image ID** extracted from the competition label key; do not use it for patient counts or leakage checks.
- **Observed image EDA:** Saved Kaggle cells list train/test DICOM examples, visualize 20 images and subtype examples, and inspect 512×512 int16 pixels/headers. DICOM UID warnings were observed; audit metadata before study reconstruction. These checks do not cover all hosted images.
- **Links:** [Notebook](../../notebooks/02_rsna-ihd.ipynb) · [RSNA 2019 challenge](https://www.rsna.org/artificial-intelligence/ai-image-challenge/RSNA-Intracranial-Hemorrhage-Detection-Challenge-2019) · [Official Kaggle data format](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data)

## 3. SLAKE

- **What it is:** Bilingual medical VQA across several body regions and conditions; not a brain-only or single-disease dataset.
- **Data:** 14,028 QA pairs over 642 unique 2D images (CT, MRI, X-ray). English and Chinese questions; open and closed answers. Sampled image folders contain semantic masks and bounding-box files.
- **Key EDA:** The notebook's brain-related subset (`Brain`, `Brain_Face`, `Brain_Tissue`) has **3,148 QA pairs / 165 images**: 2,464 MRI rows and 684 CT rows. Some sampled masks identify edema or tumor classes.
- **Use / caution:** Useful for image–question–answer grounding. Semantic masks are not universal lesion masks. Image IDs overlap across the supplied QA splits: 130 train–validation, 142 train–test, 26 validation–test; make image-disjoint splits before image-level evaluation.
- **Observed image EDA:** The saved Colab run resolved all 165 selected brain image paths, inspected five mask/box examples and visualized selected overlays. This does not establish region truth for all QA. The overlap counts above concern the full supplied QA release, not an audited brain-only split.
- **Links:** [Notebook](../../notebooks/03_SLAKE_EDA.ipynb) · [Official SLAKE page, 2021](https://www.med-vqa.com/slake/) · [Paper](https://arxiv.org/abs/2102.09542)

## 4. MR-RATE

- **What it is:** Broad brain **and spine** MRI with reports; includes T1, T2, FLAIR, SWI, MRA, and other series. Public dataset release: **2026**; its paper is listed as forthcoming.
- **Data:** 83,425 patients, 98,334 studies, 705,254 series. Available tables contain 98,200 report rows and 97,896 pathology-label rows, so coverage is not complete for either table.
- **Key EDA:** 37 study-level pathology categories; patient-level train/validation/test study counts are 88,985 / 3,781 / 5,568, with **zero patients shared across splits** in the notebook check. `findings` is missing in 0.3% of available reports; `impression` in 8.8%.
- **Use / caution:** Strong image–report resource. Pathology labels come from a report-classification pipeline; available anatomical segmentations are model-generated, **not manually drawn lesion masks**. Filter brain versus spine for brain-specific experiments. Parsed age reaches 140 years and needs outlier review.
- **Brain-filter gate:** The [official card](https://huggingface.co/datasets/Forithmus/MR-RATE) still lists series-level brain/spine Body Part Labels as forthcoming. Do not assume that field already exists. Build a provenance-tracked filter from available metadata/series descriptions, confirm sampled anatomy visually and exclude ambiguous/mixed-scope pairs pending review. A disease label or the presence of a brain mask alone is not a verified body-part classifier.
- **Observed image EDA:** Cells sampled 20 studies across the supplied splits without a brain/train filter; saved output shows ZIP downloads, 20 series visualizations and no header errors in that sample. A fresh-runtime extraction path still needs checking. Recover sampled IDs and original splits before reuse; this is not an established brain-only training cohort. Median Findings length is 143 words, with an extreme 14,262-word record requiring review; clinical information is 52% missing.
- **Links:** [Notebook](../../notebooks/04_MR-RATE.ipynb) · [Official dataset card](https://huggingface.co/datasets/Forithmus/MR-RATE) · [Dataset guide](https://github.com/forithmus/MR-RATE/blob/main/data-preprocessing/docs/dataset_guide.md)

## 5. ATLAS R3.0 raw training data for ISLES'26

- **What it is:** Native-space, skull-stripped T1w brain MRI for ischemic stroke **lesion segmentation**. ISLES'26 requires the raw/native-space release.
- **Data:** 1,453 subjects, each with a T1w NIfTI image and lesion mask. The release combines 955 earlier ATLAS cases, 169 SOOP cases, and 329 new cases.
- **Key EDA:** All 1,453 T1w files match a mask by subject ID. The combined metadata has **1,451 rows** because two SOOP CSVs contain no data rows. `CHRONICITY` is 79.6% missing; `DAYS_POST_STROKE` is 23.4% missing. An unseeded 200-case sample has 127 RAS / 73 LAS orientations, 0.25–6.00 mm through-plane spacing, no empty masks, and median lesion volume **5.1 mL**.
- **Use / caution:** Direct MRI lesion ground truth, but native-space heterogeneity requires geometry/alignment checks. The 200-case statistics are sample results. A NIfTI `pixdim` warning needs investigation before trusting all physical-volume estimates. Do not infer the meaning of `CHRONICITY = 1` without its codebook.
- **Local readiness:** The saved notebook used an extracted cohort; only the raw archive is currently present. Re-extract before repeating those image checks. All subject pairs were inventoried, while orientation/mask statistics concern 200 sampled cases and intensity checks concern 30; overlays were sampled.
- **Links:** [Notebook](../../notebooks/05_ISLES_26_EDA.ipynb) · [ISLES'26 dataset page](https://isles-26.grand-challenge.org/dataset/) · [ATLAS release page](https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html)

## 6. RadGenome-Brain MRI

**Simple inventory:** [Paper vs public release vs local data](RADGENOME.md). Read this before the detailed notebook.

| Measure | Paper / expected public scope | Public annotation downloaded | Local matched subset, excluding MET |
|---|---:|---:|---:|
| Cases | 1,007 | 1,007 case findings + 1,007 impressions | **770 cases**, all with findings/impressions |
| MRI volumes | 3,408 image IDs | 3,408 IDs in the split JSON; MRI downloaded separately | **2,460 volumes** |
| MRI–modality-report pairs | Paper claims 3,408 | **3,098 modality findings** | **2,150 pairs** |
| MRI sequences | 6 | ADC, DWI, T1CE, T1WI, T2FLAIR, T2WI | All 6 represented |
| Disease groups | 5 | GLI, MEN, MET, ISLES22, WMH | 4; MET excluded |
| Primary lesion masks | Ground-truth anomaly segmentation described | Obtained from source datasets | **770 case-level masks**, reused across sequences |

Source for paper claims: [AutoRG-Brain, section 4 and Table 4](https://arxiv.org/html/2407.16684v3). The paper's private SSPH cohort is separate from RadGenome.

### Complete, missing and extra

| Category | Count and explanation |
|---|---|
| Complete within the non-MET scope | All **770 cases / 2,460 MRI volumes / 770 primary masks** matched by ID; all case findings and impressions available |
| Missing locally | MET: **237 cases / 948 MRI volumes**; its 948 modality reports are downloaded, but no local MRI–report pairs exist |
| Absent from the public modality JSON | **310 reports**: 250 ISLES DWI + 60 WMH T1WI; this explains the gap between 3,408 claimed pairs and 3,098 released modality findings |
| Full source cohorts downloaded | **3,031 source case IDs**, including train/validation/test; **12,214 MRI representations**, excluding MEN correction versions; **46.50 GiB** of raw source files |
| Extra source cases outside RadGenome | **2,261**: GLI 1,240 + MEN 911 + WMH test 110; these have no assigned RadGenome reports |
| Extra files for existing cases | ISLES FLAIR; WMH original/3DT1/preprocessing and extra-observer masks; MEN FIX-V4 for 47 existing cases, not 47 new cases |

**Availability calculation:** 1,007 − 237 MET = **770 cases**. 3,098 public modality reports − 948 MET = **2,150 local pairs**. Do not add findings, impressions and modality findings together as independent patients or pairs.

### Current model pools and remaining decisions

- **Single-sequence input → modality finding:** 2,150 candidate pairs; **2,143** pass the current geometry filter.
- **Multiple sequences → case finding/impression:** 770 candidate cases; **763** pass that filter.
- **QC:** All 2,460 selected images read successfully. Fourteen sequences are geometry-flagged across seven cases; three ISLES masks are empty; MEN spatial units are unspecified.
- **Before evaluation:** Resolve three cross-split MEN patient-ID proxies, verify GLI release mapping (paper: BraTS2021; local archive: BraTS2023), and review image–text agreement. ID coverage alone does not establish identical source releases or clinical target validity.
- **Mentor discussion:** Choose case-level or sequence-level report generation; establish patient grouping; evaluate separately by source/modality; define mask-grounding endpoints. Modality findings are GPT-4 decompositions of case text.

**Links:** [Simple data summary](RADGENOME.md) · [Notebook](../../notebooks/06_RadGenome_Brain_MRI_EDA.ipynb) · [Detailed EDA results](RADGENOME.md) · [Data setup](../../archive/legacy_docs/RADGENOME_DOWNLOAD_GUIDE.md) · [Original annotations](https://huggingface.co/datasets/JiayuLei/RadGenome-Brain_MRI/tree/main) · [Author repository](https://github.com/ljy19970415/AutoRG-Brain)

## Dataset decisions for the active MRI VLM + XAI proposal

| Source | Keep/use for | How it enters the pipeline | Why this role |
|---|---|---|---|
| **RadGenome + its source masks** | Current A/B/C development; later scoped generation, grounding and reviewed verification | M0/M1 MRI inputs; reports for M2/M3 targets; masks + reviewed claim links for M5/M6; reviewed verdicts for M7 | Already local and combines MRI/text/spatial labels. QC, clinical pairing and source exposure still gate research use |
| **MR-RATE brain native subset** | Next intake; clinical image–report diversity; optional adaptation/generation | M0/M1 complete study → matched Findings for S1/S3; selected reviewed claims for S4 | Complements selected lesion cohorts with routine clinical studies; no expert lesion/verdict truth follows from report-derived labels |
| **ATLAS R3.0** | Optional separate T1 stroke spatial validation/adaptation | Image + stroke mask → M5 spatial objective, or held-out spatial test if untouched | Useful masks; no authentic text, no diffusion-sequence truth. Earlier ATLAS exposure must be audited |
| **SLAKE brain subset** | Optional 2D VQA/visual reliance experiment | Separate 2D image + QA model, image-disjoint splits; selected semantic-region diagnostic | QA exists, but does not provide whole-study 3D report supervision |
| **CT-ICH / RSNA-IHD** | Preserve completed EDA; defer active training to a CT extension | CT-ICH mask localization; RSNA classification, after DICOM/patient audit | They lack paired reports in reviewed data and require a CT-specific model/input protocol |
| **BIND; BrainMD alternative** | Conditional external clinical MRI image–text evaluation | Entire compatible source/site holdout after approved access and EDA | Tests transfer beyond the development sources; access/pairing/exposure are unconfirmed |
| **ROCOv2 → MedPix; VQA-RAD/VQA-Med** | Optional small 2D image–text/QA branch | 2D pretrained VLM; caption/retrieval/QA objectives and separate scores | Supports supplementary VLM analysis without claiming volumetric grounding |
| **BTReport / PDGM-VQA / UniBrain derivative** | Conditional weak-label or reproduction diagnostics | Match source/version/patient IDs before adding text/QA/2D representation | They can reuse BraTS/RadGenome patients; cannot become independent external cohorts by changing annotations |

**Combine only within a defined objective.** RadGenome/MR-RATE can contribute eligible MRI report targets with explicit study/sequence scope and source-balanced sampling. RadGenome and BraTS/ISLES/WMH are linked annotations of the same cases: use a shared patient/source manifest, not duplicated patients across splits. Tumor, infarct and WMH masks retain target-specific semantics. ATLAS contributes spatial targets only; missing text receives no language loss. All sequences/crops/reports/visits of a patient stay in the same partition.

Reference reports may construct development claims; they never become context for automatic report generation in the main evaluation. Lesion masks may supervise localization and score predictions; main inference uses predicted evidence. Report-derived labels, synthetic text and expert-reviewed verdicts remain distinct. A true claim can be unassessable to the system when its predicted ROI is wrong; record image truth separately from evidence assessability.

## Preparation that remains

1. Run prepared A/B on Colab, review ADC preprocessing and matched image/text diagnostics; then C on one reviewed claim. These are inference diagnostics, not training stages or a benchmark.
2. Resolve RadGenome GLI release mapping, three MEN cross-split proxies, flagged geometry and image–text agreement. The current 763 cases / 2,143 geometry-eligible pairs are candidate pools, not clinically validated counts.
3. Reconcile the historical 20 MR-RATE study IDs/splits; form a new **brain-only train/development** intake of approximately 20–50 studies for engineering QC. Measure actual subset bytes before downloading more. This budget is not a statistical sample-size justification.
4. Freeze patient/source partitions before training or claiming held-out performance. The 10 PRIMA cases are development. Metadata inventory alone does not imply test tuning; cases used to choose methods/prompts/thresholds must be marked development.
5. Build reviewed claim–region–sequence–verdict labels. Reports plus masks alone do not constitute a verification benchmark. If clinical review is unavailable, restrict claims to reference agreement and valid spatial endpoints.

No new dataset download or training was launched by this revision. Candidate details, access/year/bytes and rejected sources are in [the image–text review](IMAGE_TEXT_DATASET_RESEARCH_2026-10-06.md). Models are in [the landscape](../research/MODEL_LANDSCAPE_2026.md); the executable next-chat state is in [the handoff](../../research_proposal/10_IMPLEMENTATION_HANDOFF.md).
