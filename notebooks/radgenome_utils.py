"""Conservative indexing/QC for the original RadGenome annotation release.

Source files stay untouched. No implicit registration, ID renaming or report filling.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import re
import gzip
import zipfile
from functools import lru_cache
import numpy as np
import pandas as pd

SOURCES = ('BraTS_GLI', 'BraTS_MEN', 'BraTS_MET', 'ISLES22', 'WMH')
MODALITIES = {'t1n': 'T1WI', 't1c': 'T1CE', 't2w': 'T2WI', 't2f': 'T2FLAIR',
              'adc': 'ADC', 'dwi': 'DWI', 'T1': 'T1WI', 'FLAIR': 'T2FLAIR'}

MAX_VOLUME_BYTES = 256 * 1024**2  # decompressed bytes per volume; no disk cache

@lru_cache(maxsize=8)
def archive_handle(path):
    return zipfile.ZipFile(path)

def split_archive_ref(ref):
    return str(ref)[6:].split('!', 1) if str(ref).startswith('zip://') else None

def ref_exists(ref):
    if not ref or pd.isna(ref):
        return False
    archive = split_archive_ref(ref)
    if archive:
        try:
            return archive_handle(archive[0]).getinfo(archive[1]).is_dir() is False
        except (OSError, KeyError, zipfile.BadZipFile):
            return False
    return Path(ref).is_file()

class HeaderView:
    """Geometry-only view: does not allocate a voxel array."""
    def __init__(self, header):
        self.header = header
        self.shape = header.get_data_shape()
        self.affine = header.get_best_affine()
    def get_data_dtype(self): return self.header.get_data_dtype()
    def get_qform(self, coded=False): return self.header.get_qform(coded=coded)
    def get_sform(self, coded=False): return self.header.get_sform(coded=coded)

def load_image(ref, header_only=False):
    """Read ZIP NIfTI in RAM, or local NIfTI lazily. Never extract to disk."""
    import nibabel as nib
    archive = split_archive_ref(ref)
    if not archive:
        image = nib.load(ref)
        estimate = int(np.prod(image.shape)) * image.get_data_dtype().itemsize
        if not header_only and estimate > MAX_VOLUME_BYTES:
            raise MemoryError(f'Volume exceeds {MAX_VOLUME_BYTES // 1024**2} MiB budget')
        return image
    with archive_handle(archive[0]).open(archive[1]) as zipped:
        stream = gzip.GzipFile(fileobj=zipped) if archive[1].endswith('.gz') else zipped
        try:
            prefix = stream.read(540)
            size = int.from_bytes(prefix[:4], 'little')
            if size not in (348, 540):
                size = int.from_bytes(prefix[:4], 'big')
            if size not in (348, 540):
                raise ValueError('Unsupported NIfTI header')
            cls = nib.Nifti1Header if size == 348 else nib.Nifti2Header
            # Geometry is entirely in the fixed header; do not read extensions.
            header = cls(binaryblock=prefix[:size])
            if header_only:
                return HeaderView(header)
            estimate = int(np.prod(header.get_data_shape())) * header.get_data_dtype().itemsize + int(header['vox_offset'])
            if estimate > MAX_VOLUME_BYTES:
                raise MemoryError(f'Volume exceeds {MAX_VOLUME_BYTES // 1024**2} MiB budget')
            remaining = stream.read(MAX_VOLUME_BYTES + 1 - len(prefix))
            data = prefix + remaining
            if len(data) > MAX_VOLUME_BYTES:
                raise MemoryError('Decompressed NIfTI exceeds byte budget')
            image_cls = nib.Nifti1Image if size == 348 else nib.Nifti2Image
            return image_cls.from_bytes(data)
        finally:
            if stream is not zipped:
                stream.close()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def identify(image_id):
    for source, prefix in [('BraTS_GLI', 'BraTS-GLI-'), ('BraTS_MEN', 'BraTS-MEN-'), ('BraTS_MET', 'BraTS-MET-')]:
        if image_id.startswith(prefix):
            case, sequence = image_id.rsplit('-', 1)
            if sequence not in MODALITIES:
                raise ValueError(f'Unknown sequence: {image_id}')
            # Patient grouping is a conservative ID proxy, not verified identity.
            return source, case, case.rsplit('-', 1)[0], MODALITIES[sequence]
    if image_id.startswith('sub-strokecase'):
        case, sequence = image_id.rsplit('_', 1)
        return 'ISLES22', case, case.split('_ses-')[0], MODALITIES[sequence]
    if re.match(r'^(GE3T|Singapore|Utrecht)_\d+_(T1|FLAIR)$', image_id):
        case, sequence = image_id.rsplit('_', 1)
        return 'WMH', case, case, MODALITIES[sequence]
    raise ValueError(f'Unrecognized image ID: {image_id}')

def load_annotations(annotation_dir):
    annotation_dir = Path(annotation_dir)
    splits = read_json(annotation_dir / 'train_val_test_split.json')
    split_map = defaultdict(list)
    for split, ids in splits.items():
        for image_id in ids:
            split_map[image_id].append(split)
    reports, globals_, impressions = {}, {}, {}
    schema = []
    for source in SOURCES:
        for filename, target in [('modal_wise_finding.json', reports), ('global_finding.json', globals_), ('impression.json', impressions)]:
            data = read_json(annotation_dir / source / filename)
            if not isinstance(data, dict):
                raise ValueError(f'{source}/{filename}: expected a dictionary')
            schema.append({'source': source, 'file': filename, 'records': len(data), 'value_types': ','.join(sorted({type(v).__name__ for v in data.values()}))})
            for key, value in data.items():
                if (source, key) in target:
                    raise ValueError(f'Duplicate annotation key: {source}/{key}')
                target[source, key] = value
    ids = set(split_map) | {key for source, key in reports}
    rows = []
    for image_id in sorted(ids):
        source, case, patient, modality = identify(image_id)
        imp = impressions.get((source, case), {})
        if not isinstance(imp, dict):
            raise ValueError(f'Unexpected impression schema for {case}')
        modal = reports.get((source, image_id))
        global_text = globals_.get((source, case))
        if modal is not None and not isinstance(modal, str):
            raise ValueError(f'Unexpected modal report for {image_id}')
        if global_text is not None and not isinstance(global_text, str):
            raise ValueError(f'Unexpected global report for {case}')
        disease = imp.get('disease', [])
        if not isinstance(disease, list):
            raise ValueError(f'Unexpected disease list for {case}')
        rows.append({'source': source, 'image_id': image_id, 'case_id': case,
                     'patient_id_proxy': patient, 'patient_identity_verified': False,
                     'modality': modality, 'split': '|'.join(sorted(set(split_map[image_id]))) or 'unassigned',
                     'split_occurrences': len(split_map[image_id]), 'modal_finding': modal,
                     'global_finding': global_text, 'impression': imp.get('impression'),
                     'disease_terms': json.dumps(disease, ensure_ascii=False),
                     'modal_report_origin': 'GPT-4 modality decomposition' if modal else 'missing',
                     'global_report_origin': 'radiologist-written case findings',
                     'anatomy_label_origin': 'not supplied; model pseudo-label if generated later'})
    return pd.DataFrame(rows), pd.DataFrame(schema), splits

def leakage_audit(df):
    rows = []
    for level in ['image_id', 'case_id', 'patient_id_proxy']:
        for (source, key), group in df.groupby(['source', level]):
            splits = set(s for value in group['split'] for s in value.split('|') if s != 'unassigned')
            if len(splits) > 1:
                rows.append({'level': level, 'source': source, 'id': key, 'splits': '|'.join(sorted(splits))})
    return pd.DataFrame(rows, columns=['level', 'source', 'id', 'splits'])

def nifti_stem(path):
    name = Path(path).name
    return name[:-7] if name.endswith('.nii.gz') else name[:-4]

def discover_files(source_dir, excluded_archives=(), active_sources=None):
    """Index supported original filenames; multiple candidates remain ambiguous."""
    index = defaultdict(list)
    inventory = []
    for source in (SOURCES if active_sources is None else active_sources):
        entries = []
        for local in sorted((Path(source_dir) / source).rglob('*')):
            if not local.is_file(): continue
            if local.name.endswith(('.nii', '.nii.gz')):
                entries.append((local, str(local.resolve()), '', local.stat().st_size))
            elif local.suffix.lower() == '.zip':
                try:
                    with zipfile.ZipFile(local) as listing:
                        members = listing.infolist()
                except (OSError, zipfile.BadZipFile) as exc:
                    inventory.append({'source': source, 'path': str(local.resolve()),
                                      'kind': 'archive_error', 'match_key': None, 'archive': local.name,
                                      'member_bytes': local.stat().st_size, 'selected_for_linking': False,
                                      'archive_error': type(exc).__name__})
                    continue
                for member in members:
                    if not member.is_dir() and member.filename.endswith(('.nii', '.nii.gz')) and '__MACOSX' not in member.filename:
                        entries.append((Path(member.filename), f'zip://{local.resolve()}!{member.filename}', local.name, member.file_size))
        for p, ref, container, size_bytes in entries:
            stem = nifti_stem(p)
            key, kind = None, 'unrecognized'
            if source.startswith('BraTS'):
                if re.fullmatch(r'BraTS-(GLI|MEN|MET)-\d+-\d+-(t1n|t1c|t2w|t2f)', stem):
                    key, kind = stem, 'image'
                elif re.fullmatch(r'BraTS-(GLI|MEN|MET)-\d+-\d+-seg', stem):
                    key, kind = stem[:-4], 'lesion_mask'
            elif source == 'ISLES22':
                if re.fullmatch(r'sub-strokecase\d+_ses-\d+_(adc|dwi|flair|FLAIR)', stem):
                    key, kind = stem, 'image'
                elif re.fullmatch(r'sub-strokecase\d+_ses-\d+_msk', stem):
                    key, kind = stem[:-4], 'lesion_mask'
            elif source == 'WMH':
                parts = p.parts
                scanner = next((x for x in parts if x in ('GE3T', 'Singapore', 'Utrecht')), None)
                numeric = [x for x in p.parent.parts if x.isdigit()]
                if scanner and numeric:
                    case = f'{scanner}_{numeric[-1]}'
                    # Prefer preprocessed volumes in the mask coordinate space.
                    if stem in ('T1', 'FLAIR') and p.parent.name == 'pre':
                        key, kind = f'{case}_{stem}', 'image'
                    elif stem == 'wmh':
                        key, kind = case, 'lesion_mask'
            selected = container not in excluded_archives
            # Full WMH release also has challenge test and extra observers.
            # Retain them in inventory, but link RadGenome only to the training GT.
            if source == 'WMH' and any(part in ('test', 'additional_annotations') for part in p.parts):
                selected = False
            if key and selected:
                index[source, kind, key].append(ref)
            inventory.append({'source': source, 'path': ref, 'kind': kind, 'match_key': key,
                              'archive': container, 'member_bytes': size_bytes,
                              'selected_for_linking': selected, 'archive_error': None})
    return index, pd.DataFrame(inventory, columns=['source', 'path', 'kind', 'match_key', 'archive', 'member_bytes',
                                                 'selected_for_linking', 'archive_error'])

def link_files(df, source_dir, override_file=None, excluded_archives=(), active_sources=None):
    index, inventory = discover_files(source_dir, excluded_archives=excluded_archives, active_sources=active_sources)
    overrides = {}
    if override_file and Path(override_file).exists():
        tab = pd.read_csv(override_file, dtype=str).fillna('')
        if tab.duplicated(['source', 'image_id']).any():
            raise ValueError('Duplicate override keys')
        overrides = {(r.source, r.image_id): r._asdict() for r in tab.itertuples(index=False)}
        unknown = set(overrides) - set(zip(df.source, df.image_id))
        if unknown:
            raise ValueError(f'Unknown override IDs: {sorted(unknown)[:5]}')
    rows = []
    for row in df.to_dict('records'):
        override = overrides.get((row['source'], row['image_id']), {})
        for kind, key in [('image', row['image_id']), ('lesion_mask', row['case_id'])]:
            value = override.get(kind + '_path', '')
            candidates = index.get((row['source'], kind, key), [])
            if value:
                p = Path(value)
                ref = value if split_archive_ref(value) else str((p if p.is_absolute() else Path(source_dir) / row['source'] / p).resolve())
                row[kind + '_path'] = ref
                row[kind + '_status'] = 'manual_override' if ref_exists(ref) else 'override_missing'
            else:
                row[kind + '_path'] = candidates[0] if len(candidates) == 1 else None
                row[kind + '_status'] = 'matched' if len(candidates) == 1 else ('ambiguous' if candidates else 'missing')
        row['source_version'] = override.get('source_version') or 'unverified'
        row['mapping_evidence'] = override.get('mapping_evidence') or 'filename/path match only; verify release identity'
        rows.append(row)
    return pd.DataFrame(rows), inventory

def header_audit(df):
    import nibabel as nib
    rows = []
    for row in df.to_dict('records'):
        out = {k: row[k] for k in ['source', 'image_id', 'case_id', 'modality']}
        out['image_path'], out['lesion_mask_path'] = row['image_path'], row['lesion_mask_path']
        out['geometry_ok'] = False
        if not ref_exists(row['image_path']):
            out['qc_status'] = 'missing_image'
            rows.append(out)
            continue
        try:
            image = load_image(row['image_path'], header_only=True)
            units = image.header.get_xyzt_units()[0]
            spacing = image.header.get_zooms()[:3]
            out.update(shape=str(image.shape), dtype=str(image.get_data_dtype()),
                       orientation=''.join(x or '?' for x in nib.aff2axcodes(image.affine)),
                       spatial_unit=units, sx=float(spacing[0]), sy=float(spacing[1]), sz=float(spacing[2]),
                       qform_code=int(image.header['qform_code']), sform_code=int(image.header['sform_code']))
            if len(image.shape) != 3:
                out['qc_status'] = 'not_3d'
            elif not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
                out['qc_status'] = 'invalid_affine'
            elif not np.isfinite(spacing).all() or min(spacing) <= 0:
                out['qc_status'] = 'invalid_spacing'
            elif not ref_exists(row['lesion_mask_path']):
                out['qc_status'] = 'missing_mask'
            else:
                mask = load_image(row['lesion_mask_path'], header_only=True)
                shape_ok = image.shape == mask.shape
                affine_ok = np.allclose(image.affine, mask.affine, rtol=0, atol=1e-4)
                unit_ok = units == mask.header.get_xyzt_units()[0]
                out.update(shape_match=shape_ok, affine_match=affine_ok, unit_match=unit_ok,
                           geometry_ok=bool(shape_ok and affine_ok and unit_ok),
                           qc_status='aligned' if shape_ok and affine_ok and unit_ok else 'geometry_mismatch')
                q, qc = image.get_qform(coded=True)
                s, sc = image.get_sform(coded=True)
                out['image_qform_sform_conflict'] = bool(qc and sc and not np.allclose(q, s, rtol=0, atol=1e-4))
                mq, mqc = mask.get_qform(coded=True)
                ms, msc = mask.get_sform(coded=True)
                out['mask_qform_sform_conflict'] = bool(mqc and msc and not np.allclose(mq, ms, rtol=0, atol=1e-4))
                if out['image_qform_sform_conflict'] or out['mask_qform_sform_conflict']:
                    out.update(geometry_ok=False, qc_status='qform_sform_conflict')
        except Exception as exc:
            out.update(qc_status='read_error', error=str(exc))
        rows.append(out)
    return pd.DataFrame(rows)

def voxel_audit(headers, max_cases_per_source=10, seed=42):
    """Read one aligned volume per case. Component count is a geometric proxy."""
    import nibabel as nib
    from scipy.ndimage import label, generate_binary_structure
    eligible = headers.loc[headers['geometry_ok']]
    chunks = []
    for _, group in eligible.groupby('source'):
        cases = group[['case_id']].drop_duplicates()
        cases = cases.sample(n=min(len(cases), max_cases_per_source), random_state=seed) if max_cases_per_source else cases
        modalities = sorted(group.modality.unique())
        # Keep one mask observation per case; rotate modalities for intensity EDA.
        chosen = []
        for i, case in enumerate(cases.case_id):
            options = group.loc[group.case_id.eq(case)]
            preferred = options.loc[options.modality.eq(modalities[i % len(modalities)])]
            chosen.append((preferred if not preferred.empty else options).iloc[[0]])
        if chosen: chunks.append(pd.concat(chosen, ignore_index=True))
    selected = pd.concat(chunks, ignore_index=True) if chunks else eligible
    rows = []
    for r in selected.itertuples(index=False):
        out = {'source': r.source, 'case_id': r.case_id, 'image_id': r.image_id,
               'modality': r.modality,
               'image_path': r.image_path, 'lesion_mask_path': r.lesion_mask_path}
        try:
            image_obj = load_image(r.image_path)
            image = image_obj.get_fdata(dtype=np.float32)
            mask_obj = load_image(r.lesion_mask_path)
            mask = mask_obj.get_fdata(dtype=np.float32)
            finite = np.isfinite(image)
            unique = np.unique(mask)
            mask_valid = np.isfinite(mask).all() and np.allclose(mask, np.rint(mask)) and np.min(mask) >= 0
            allowed = {0, 1, 2} if r.source == 'WMH' else ({0, 1} if r.source == 'ISLES22' else {0, 1, 2, 3, 4})
            unexpected = set(unique.tolist()) - allowed
            mask_valid = mask_valid and not unexpected
            # WMH label 2 represents other pathology, not WMH: only label 1 is target.
            binary = mask == 1 if r.source == 'WMH' else mask > 0
            out.update(image_nonfinite=int((~finite).sum()), zero_fraction=float(np.mean(image == 0)),
                       mask_labels=json.dumps(unique.tolist()), mask_valid=bool(mask_valid),
                       unexpected_labels=json.dumps(sorted(unexpected)))
            if finite.any():
                out.update(zip(['p01', 'p50', 'p99'], map(float, np.percentile(image[finite], [1, 50, 99]))))
            if mask_valid:
                nvox = int(binary.sum())
                units = image_obj.header.get_xyzt_units()[0]
                scale = {'mm': 1., 'meter': 1000., 'micron': .001}.get(units)
                out.update(lesion_voxels=nvox, empty_mask=nvox == 0,
                           components_26=int(label(binary, generate_binary_structure(3, 3))[1]),
                           lesion_ml=float(nvox * abs(np.linalg.det(image_obj.affine[:3, :3])) * scale**3 / 1000) if scale else np.nan,
                           volume_unit_status='known' if scale else 'unknown; physical volume withheld')
            out['qc_status'] = 'read_ok' if mask_valid else 'invalid_mask_values'
        except Exception as exc:
            out.update(qc_status='read_error', error=str(exc))
        rows.append(out)
    return pd.DataFrame(rows, columns=None if rows else ['source', 'case_id', 'image_id', 'qc_status'])

def cohort_mask_audit(headers):
    """Audit one mask per case; no sampling, disk extraction, or intensity scan."""
    candidates = headers.loc[headers.lesion_mask_path.map(ref_exists)].sort_values('geometry_ok', ascending=False)
    candidates = candidates.drop_duplicates(['source', 'case_id'])
    rows = []
    for r in candidates.itertuples(index=False):
        out = {'source': r.source, 'case_id': r.case_id, 'lesion_mask_path': r.lesion_mask_path,
               'geometry_eligible': bool(r.geometry_ok)}
        try:
            obj = load_image(r.lesion_mask_path)
            data = obj.get_fdata(dtype=np.float32)
            values, counts = np.unique(data, return_counts=True)
            allowed = {0, 1, 2} if r.source == 'WMH' else ({0, 1} if r.source == 'ISLES22' else {0, 1, 2, 3, 4})
            valid = np.isfinite(values).all() and np.allclose(values, np.rint(values)) and set(values.tolist()) <= allowed
            out.update(mask_valid=bool(valid), mask_labels=json.dumps(values.tolist()),
                       label_voxel_counts=json.dumps(dict(zip(map(str, values.tolist()), map(int, counts)))),
                       qc_status='read_ok' if valid else 'invalid_mask_values')
            if valid:
                target = values == 1 if r.source == 'WMH' else values > 0
                nvox = int(counts[target].sum())
                unit = obj.header.get_xyzt_units()[0]
                scale = {'mm': 1., 'meter': 1000., 'micron': .001}.get(unit)
                out.update(lesion_voxels=nvox, empty_mask=nvox == 0, spatial_unit=unit,
                           lesion_ml=float(nvox * abs(np.linalg.det(obj.affine[:3, :3])) * scale**3 / 1000)
                           if scale and r.geometry_ok else np.nan,
                           volume_unit_status='known_and_geometry_eligible' if scale and r.geometry_ok
                           else 'withheld_unknown_units_or_geometry')
        except Exception as exc:
            out.update(mask_valid=False, qc_status='read_error', error=str(exc))
        rows.append(out)
    return pd.DataFrame(rows)

def image_intensity_audit(headers, existing=None):
    """Read every available image; reuse intensity results from the case audit."""
    fields = ['image_nonfinite', 'zero_fraction', 'p01', 'p50', 'p99']
    cached = {}
    if existing is not None and not existing.empty:
        cached = {r['image_id']: r for r in existing.to_dict('records') if r.get('qc_status') == 'read_ok'}
    rows = []
    for r in headers.itertuples(index=False):
        if not ref_exists(r.image_path):
            continue
        out = {k: getattr(r, k) for k in ['source', 'case_id', 'image_id', 'modality']}
        try:
            if r.image_id in cached:
                out.update({k: cached[r.image_id][k] for k in fields})
            else:
                obj = load_image(r.image_path)
                data = obj.get_fdata(dtype=np.float32)
                finite = np.isfinite(data)
                out.update(image_nonfinite=int((~finite).sum()), zero_fraction=float(np.mean(data == 0)))
                if finite.any():
                    out.update(zip(['p01', 'p50', 'p99'], map(float, np.percentile(data[finite], [1, 50, 99]))))
                else:
                    out.update(p01=np.nan, p50=np.nan, p99=np.nan)
            out['qc_status'] = 'read_ok'
        except Exception as exc:
            out.update(qc_status='read_error', error=str(exc))
        rows.append(out)
    return pd.DataFrame(rows)

def plot_overlay(image_path, mask_path, source, title=''):
    import nibabel as nib
    import matplotlib.pyplot as plt
    image_obj, mask_obj = load_image(image_path), load_image(mask_path)
    if image_obj.shape != mask_obj.shape or not np.allclose(image_obj.affine, mask_obj.affine, rtol=0, atol=1e-4):
        raise ValueError('Geometry mismatch: overlay refused; review registration first')
    image = nib.as_closest_canonical(image_obj).get_fdata(dtype=np.float32)
    mask = nib.as_closest_canonical(mask_obj).get_fdata(dtype=np.float32)
    if not np.isfinite(mask).all() or np.min(mask) < 0 or not np.allclose(mask, np.rint(mask)):
        raise ValueError('Invalid mask values')
    binary = mask == 1 if source == 'WMH' else mask > 0
    foreground = image[np.isfinite(image) & (image != 0)]
    if not foreground.size:
        raise ValueError('No finite nonzero image values')
    lo, hi = np.percentile(foreground, [1, 99])
    if lo == hi:
        hi = lo + 1
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, (ax, name) in enumerate(zip(axes, ['Sagittal', 'Coronal', 'Axial'])):
        area = binary.sum(axis=tuple(i for i in range(3) if i != axis))
        idx = int(np.argmax(area)) if binary.any() else image.shape[axis] // 2
        sl = np.take(image, idx, axis=axis).T
        ms = np.take(binary, idx, axis=axis).T
        zoom = image_obj.header.get_zooms()[:3]
        # Canonical orientation can permute voxel sizes; use canonical image header.
        zoom = nib.as_closest_canonical(image_obj).header.get_zooms()[:3]
        remaining = [i for i in range(3) if i != axis]
        ax.imshow(sl, origin='lower', cmap='gray', vmin=lo, vmax=hi, aspect=zoom[remaining[1]] / zoom[remaining[0]])
        ax.imshow(np.ma.masked_where(~ms, ms), origin='lower', cmap='autumn', alpha=.45, aspect=zoom[remaining[1]] / zoom[remaining[0]])
        ax.set_title(f'{name}, voxel index {idx}'); ax.axis('off')
    fig.suptitle(title + ' | canonical RAS; voxel views, no radiological side labels')
    fig.tight_layout()
    return fig

def file_sha256(path):
    h = hashlib.sha256()
    archive = split_archive_ref(path)
    if archive:
        with archive_handle(archive[0]).open(archive[1]) as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
