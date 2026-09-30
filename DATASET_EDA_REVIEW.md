# Dataset EDA review: five datasets

**Concept:** [Introduction to Intracranial Haemorrhage](https://youtu.be/Kb_wzb7-rvE?si=i5SrPgfKGWdzl5q4)

## Comparison at a glance

| Dataset | Year | Clinical focus | Modality / unit | Size observed in EDA | Labels and lesion masks | Paired text |
|---|---|---|---|---|---|---|
| CT-ICH v1.3.1 | 2020 release | Traumatic intracranial hemorrhage (ICH) | Non-contrast CT / 3D volume and slices | 75 volumes; 2,814 labeled slices | Five hemorrhage subtypes, fracture; **ICH masks** | No |
| RSNA-IHD | 2019 challenge | Acute ICH | Head CT DICOM / image-slice | 4,516,842 label rows; ~752,807 images implied | Five subtypes + `any`; **no masks in reviewed data** | No |
| SLAKE | 2021 | Medical visual question answering (VQA), multiple body regions | CT, MRI, X-ray / 2D image and QA pair | 14,028 QA pairs; 642 images | Answers; semantic masks/boxes in sampled image folders | **Questions and answers**, English/Chinese |
| MR-RATE | 2026 release | Broad brain **and spine** findings | Multi-sequence 3D MRI / series and study | 705,254 series; 98,334 studies | 37 report-derived pathology labels; **no manual lesion masks** | **Radiology reports** |
| ATLAS R3.0 / ISLES'26 | 2026 challenge release | Ischemic stroke lesion segmentation | T1w 3D MRI / subject | 1,453 T1w–mask pairs | **Stroke lesion masks** | No reports or QA |

## 1. CT-ICH v1.3.1

- **What it is:** Non-contrast CT of traumatic ICH; five subtypes (IVH, IPH, SAH, EDH, SDH), plus a fracture label.
- **Data:** 75 CT NIfTI volumes, 75 matching-name mask volumes, and 2,814 slice-label rows. The original cohort has 82 patient records; seven have no released CT volume.
- **Key EDA:** 36/75 imaged patients have at least one subtype after slice-to-patient aggregation. Positive slice counts: EDH 173, IPH 73, SDH 56, IVH 24, SAH 18. **25 slices are multi-label.**
- **Use / caution:** Suitable for hemorrhage classification and ICH localization assessment. The reviewed masks do not establish a subtype-to-voxel mapping. The demographics CSV contains an extra header and footer rows; use the cleaned 82-patient table. The original notebook checked one image/mask example; the later header audit is described below.
- **Additional local QC:** A 2026-09-29 read-only audit checked full qform/sform geometry header fields for all 75 image/mask pairs and found no mismatch; visual overlays remain. The slice CSV has a UTF-8 BOM and uses `SliceNumber=1..D`, so convert to zero-based tensor indices explicitly. Sampled mask files store int16 extremes with NIfTI scaling to approximately `0/255`; load scaled values before binarizing. A raw `>0` threshold on stored values is unsafe. All 2,814 CSV rows belong to the 75 imaged patients; 36 patients have a positive ICH subtype.
- **Links:** [Notebook](notebooks/01_CT-ICH_EDA.ipynb) · [Official PhysioNet release, 2020](https://physionet.org/content/ct-ich/1.3.1/)

## 2. RSNA Intracranial Hemorrhage Detection

- **What it is:** Acute ICH classification on head CT, with five subtype targets and an `any` target.
- **Data:** Stage-2 training CSV has 4,516,842 label rows and no missing labels. The official format uses six rows per DICOM image, implying ~752,807 images; unique image IDs were not counted separately in this notebook.
- **Key EDA:** Positive image-label rows: `any` 107,933; subdural 47,166; intraparenchymal 36,118; subarachnoid 35,675; intraventricular 26,205; epidural 3,145. Epidural is rare relative to other subtypes.
- **Use / caution:** Strong slice/image classification resource; no lesion masks or paired reports in the reviewed data. The notebook's derived `PatientID` column is actually an **image ID** extracted from the competition label key; do not use it for patient counts or leakage checks.
- **Links:** [Notebook](notebooks/02_rsna-ihd.ipynb) · [RSNA 2019 challenge](https://www.rsna.org/artificial-intelligence/ai-image-challenge/RSNA-Intracranial-Hemorrhage-Detection-Challenge-2019) · [Official Kaggle data format](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data)

## 3. SLAKE

- **What it is:** Bilingual medical VQA across several body regions and conditions; not a brain-only or single-disease dataset.
- **Data:** 14,028 QA pairs over 642 unique 2D images (CT, MRI, X-ray). English and Chinese questions; open and closed answers. Sampled image folders contain semantic masks and bounding-box files.
- **Key EDA:** The notebook's brain-related subset (`Brain`, `Brain_Face`, `Brain_Tissue`) has **3,148 QA pairs / 165 images**: 2,464 MRI rows and 684 CT rows. Some sampled masks identify edema or tumor classes.
- **Use / caution:** Useful for image–question–answer grounding. Semantic masks are not universal lesion masks. Image IDs overlap across the supplied QA splits: 130 train–validation, 142 train–test, 26 validation–test; make image-disjoint splits before image-level evaluation.
- **Links:** [Notebook](notebooks/03_SLAKE_EDA.ipynb) · [Official SLAKE page, 2021](https://www.med-vqa.com/slake/) · [Paper](https://arxiv.org/abs/2102.09542)

## 4. MR-RATE

- **What it is:** Broad brain **and spine** MRI with reports; includes T1, T2, FLAIR, SWI, MRA, and other series. Public dataset release: **2026**; its paper is listed as forthcoming.
- **Data:** 83,425 patients, 98,334 studies, 705,254 series. Available tables contain 98,200 report rows and 97,896 pathology-label rows, so coverage is not complete for either table.
- **Key EDA:** 37 study-level pathology categories; patient-level train/validation/test study counts are 88,985 / 3,781 / 5,568, with **zero patients shared across splits** in the notebook check. `findings` is missing in 0.3% of available reports; `impression` in 8.8%.
- **Use / caution:** Strong image–report resource. Pathology labels come from a report-classification pipeline; available anatomical segmentations are model-generated, **not manually drawn lesion masks**. Filter brain versus spine for brain-specific experiments. Parsed age reaches 140 years and needs outlier review.
- **Links:** [Notebook](notebooks/04_MR-RATE.ipynb) · [Official dataset card](https://huggingface.co/datasets/Forithmus/MR-RATE) · [Dataset guide](https://github.com/forithmus/MR-RATE/blob/main/data-preprocessing/docs/dataset_guide.md)

## 5. ATLAS R3.0 raw training data for ISLES'26

- **What it is:** Native-space, skull-stripped T1w brain MRI for ischemic stroke **lesion segmentation**. ISLES'26 requires the raw/native-space release.
- **Data:** 1,453 subjects, each with a T1w NIfTI image and lesion mask. The release combines 955 earlier ATLAS cases, 169 SOOP cases, and 329 new cases.
- **Key EDA:** All 1,453 T1w files match a mask by subject ID. The combined metadata has **1,451 rows** because two SOOP CSVs contain no data rows. `CHRONICITY` is 79.6% missing; `DAYS_POST_STROKE` is 23.4% missing. An unseeded 200-case sample has 127 RAS / 73 LAS orientations, 0.25–6.00 mm through-plane spacing, no empty masks, and median lesion volume **5.1 mL**.
- **Use / caution:** Direct MRI lesion ground truth, but native-space heterogeneity requires geometry/alignment checks. The 200-case statistics are sample results. A NIfTI `pixdim` warning needs investigation before trusting all physical-volume estimates. Do not infer the meaning of `CHRONICITY = 1` without its codebook.
- **Links:** [Notebook](notebooks/05_ISLES_26_EDA.ipynb) · [ISLES'26 dataset page](https://isles-26.grand-challenge.org/dataset/) · [ATLAS release page](https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html)

## Three points to emphasize in the review

1. **Task fit:** CT-ICH and RSNA address hemorrhage; ATLAS addresses ischemic stroke; SLAKE addresses VQA; MR-RATE addresses broad MRI/report learning.
2. **Spatial evidence:** CT-ICH and ATLAS have lesion masks. SLAKE has selected semantic annotations. RSNA has classification labels. MR-RATE has report-derived pathology labels and model-generated anatomical masks.
3. **Evaluation unit and leakage:** Distinguish slices, QA pairs, series, studies, and patients. SLAKE has image overlap across its supplied QA splits; MR-RATE's checked splits are patient-disjoint; RSNA needs a DICOM-level patient split audit.

## Processing and loading declaration for a combined catalog

The common index has one row per **source-native case** plus child image/series/QA/annotation records. It stores source release and provenance, patient/study/series identity when verified, label level, image and mask geometry, text availability, split, task eligibility, QC state, and preprocessing version. A model loader requests a task and emits only eligible examples; absent labels are masked out of that task's loss. CT, MRI, and 2D QA have separate decoding and normalization paths. Any shared representation is tested after the source-specific baselines.

| Branch | Unit entering model | Required preprocessing/QC | First research role |
|---|---|---|---|
| RSNA-IHD | Verified NCCT series/exam with ordered DICOM slices; optional sampled 2.5D view | Reconstruct DICOM hierarchy and patient split from metadata; apply valid pixel rescale/window rules; audit orientation, spacing, duplicates, missing slices; **images not local yet** | Later large CT classification training and internal test |
| CT-ICH | NIfTI volume, aligned ICH mask, slice labels | Check all 75 image-mask affine/shape pairs and overlays; inspect released intensity convention; retain inverse coordinate map | Small mask-grounded CT pilot or independent test with uncertainty |
| SLAKE | One 2D image plus its grouped QA/semantic annotations | Image-disjoint split; brain filter; annotation-type flags | Separate VQA branch, not CT/ICH report supervision |
| MR-RATE | Brain MRI study with sequence-specific series and verified report link | Brain/spine filtering, official patient split, sequence identity, availability flags | Later MRI image-report branch |
| ATLAS R3.0 / ISLES'26 | Native-space T1w volume and stroke lesion mask | Affine/orientation/spacing checks; nearest-neighbor mask resampling; challenge raw-data rule | Separate stroke segmentation branch |

Do **not** concatenate labels across these branches. The shared hemorrhage subtype ontology applies only where verified; stroke and tumor masks describe different targets. Report source counts after QC and filtering, and evaluate source by source. The [PhysioNet CT-ICH release](https://physionet.org/content/ct-ich/1.3.1/) documents the limited released volume count and its conversion history; the [ISLES'26 organizer](https://isles-26.grand-challenge.org/dataset/) states that all 1,453 public cases are training data and requires raw native-space MRI.

*Year note:* ATLAS R3.0 is a 2026 challenge release with its own manuscript still forthcoming; the frequently cited **2022** ATLAS paper describes the earlier v2.0 release.

## Additional image–text datasets to consider

**Reading rule:** These six datasets were researched from their official papers and repositories; they were **not** analyzed in the five notebooks above. Size units differ: CT scans or slices, MRI scans, clinical studies, 2D images, reports, and QA pairs are not interchangeable. **Fit** refers to a brain-focused multimodal healthcare system.

| Name | Year | Task | Modality / scale | Size of dataset | DOI | Public availability | Access | Fit |
|---|---|---|---|---|---|---|---|---|
| [RadGenome-Brain MRI](https://github.com/ljy19970415/AutoRG-Brain) | 2025 online; 2026 journal issue | Grounded MRI report generation; abnormality localization | 3D, six MRI sequences; infarction, WMH, glioma, meningioma, metastasis | 3,408 multimodal scans with reports and anomaly masks | [10.1109/JBHI.2025.3646647](https://doi.org/10.1109/JBHI.2025.3646647) | **Partial original release; secondary preprocessed images available** | [Original Hugging Face](https://huggingface.co/datasets/JiayuLei/RadGenome-Brain_MRI/tree/main) hosts annotations, **not MRI volumes**. [UniBrain's derivative release](https://huggingface.co/datasets/Astrostellar/RadGenome-Brain_MRI_parquet) hosts 25.3 GB of preprocessed data; verify image coverage, subject IDs, and source terms | **High** for MRI report generation with lesion grounding; check whether the derivative 2D preprocessing meets the task |
| [CTRG-Brain-263K](https://github.com/tangyuhao2016/CTRG) | 2024 | Brain CT report generation | CT, sequential 2D scans grouped into cases; reports | 263,670 CT scans across 10,009 cases | [10.1016/j.eswa.2023.121442](https://doi.org/10.1016/j.eswa.2023.121442) | **Publicly linked**, but reuse terms unclear | Full dataset linked by authors through Baidu Netdisk (code `nfyn`); check download and license before use | **High** for CT-to-report work if accessible; language and preprocessing need review |
| [BIND v1.0](https://bdsp.io/content/n1vba1x5qt62frfjem65/1.0/) | 2025 dataset; 2026 paper | Image–report learning and broad neurological phenotyping | MRI, CT, PET, SPECT; clinical brain imaging with reports | ~1.8 million scans, ~39,000 people; 84,960 brain-related reports processed for metadata | [10.60508/mby8-3a26](https://doi.org/10.60508/mby8-3a26) (dataset); [10.1038/s41597-026-07421-x](https://doi.org/10.1038/s41597-026-07421-x) (paper) | **Controlled access** for approved research | [BDSP](https://bdsp.io/content/n1vba1x5qt62frfjem65/1.0/) requires AWS account, data use agreement, CITI training, and approval | **High** for large clinical multimodal research; substantial access and compute effort |
| [VQA-RAD](https://www.nature.com/articles/sdata2018251) | 2018 | Clinician-authored visual question answering | 2D radiology images across head, chest, abdomen; multiple modalities | 315 images; 3,515 questions | [10.1038/sdata.2018.251](https://doi.org/10.1038/sdata.2018.251) | **Open** | [OSF archive](https://osf.io/89kps/) with images and QA files | **Medium**: useful small VQA benchmark; filter head images for brain work |
| [ImageCLEF VQA-Med 2019](https://github.com/abachaa/VQA-Med-2019) | 2019 | Visual question answering on modality, plane, organ, abnormality | 2D radiology images; mixed body regions and imaging modalities | 4,200 images: 3,200 train, 500 validation, 500 test; 15,292 QA pairs/questions total | [10.5281/zenodo.10499039](https://doi.org/10.5281/zenodo.10499039) (data); [overview paper](https://ceur-ws.org/Vol-2380/paper_272.pdf) has no DOI listed | **Open** | [Zenodo ZIPs](https://zenodo.org/records/10499039); [organizer repository](https://github.com/abachaa/VQA-Med-2019) states CC BY 4.0 | **Medium**: accessible VQA baseline; filter skull/brain cases and review automatically generated QA |
| [ROCOv2](https://www.nature.com/articles/s41597-024-03496-6) | 2024 | Image captioning, image–text retrieval, concept detection | 2D radiology figures, seven modalities; mixed anatomy | 79,789 image–caption pairs; 6.4 GB download | [10.5281/zenodo.10821435](https://doi.org/10.5281/zenodo.10821435) (data); [10.1038/s41597-024-03496-6](https://doi.org/10.1038/s41597-024-03496-6) (paper) | **Open** | [Zenodo](https://zenodo.org/records/10821435) images, captions, and per-image `license_information.csv` | **Medium** for image–text pretraining; select brain images and check individual image licenses |

**Selection:** For an immediately downloadable VQA benchmark, start with VQA-RAD or VQA-Med 2019. For brain-specific report generation, prioritize RadGenome-Brain MRI or CTRG-Brain after checking image coverage and access. BIND offers the broadest clinical scale once access is approved. ROCOv2 is useful for general medical image–text pretraining, but its brain subset must be identified.

*BIND count note:* The 2025 [dataset page](https://bdsp.io/content/n1vba1x5qt62frfjem65/1.0/) and 2026 [paper](https://www.nature.com/articles/s41597-026-07421-x) report slightly different exact scan and participant counts. Approximate values above avoid mixing versions; use one cited version for any formal quantitative comparison.

## Model landscape

For recent foundation and task-specific models, direct dataset evidence, release status, and ranked testing priorities, see [Brain imaging model landscape — September 2026](MODEL_LANDSCAPE_2026.md). Rankings are separated by task because segmentation, classification, VQA, and report generation use different endpoints.
