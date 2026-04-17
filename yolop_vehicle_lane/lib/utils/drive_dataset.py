"""Utilities to resolve packaged datasets from Google Drive in Colab notebooks.
Mirrors the robust path recovery logic used in the working DETR_GeoLane line:
- every notebook resolves its own local SSD copy
- global paths_config.yaml is honored
- raw BDD zips in EcoCAR/downloads can be auto-extracted back into /content
"""

import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _read_paths_config(ecocar_root: str) -> dict:
    candidate_paths = []
    if ecocar_root:
        candidate_paths.append(os.path.join(ecocar_root, 'paths_config.yaml'))
        parent = str(Path(ecocar_root).parent)
        if parent and parent != ecocar_root:
            candidate_paths.append(os.path.join(parent, 'paths_config.yaml'))
    for cfg_path in candidate_paths:
        if not os.path.isfile(cfg_path) or yaml is None:
            continue
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
        except Exception:
            continue
    return {}


def _normalize_lane_candidates(lane_dir_candidates: Optional[Iterable[str]]) -> List[str]:
    if not lane_dir_candidates:
        return []
    return [x for x in lane_dir_candidates if isinstance(x, str) and x.strip()]


def _has_dataset_layout(
    root: str,
    lane_dir_candidates: Optional[Iterable[str]] = ('masks', 'lane_masks'),
    require_lane_dir: bool = True,
) -> bool:
    root = str(root)
    if not os.path.isdir(os.path.join(root, 'images', 'train')):
        return False
    if not os.path.isdir(os.path.join(root, 'labels', 'train')):
        return False
    lane_dir_candidates = _normalize_lane_candidates(lane_dir_candidates)
    if not require_lane_dir:
        return True
    for lane_name in lane_dir_candidates:
        if os.path.isdir(os.path.join(root, lane_name, 'train')):
            return True
    return False


def _find_dataset_roots(
    search_roots: List[str],
    max_depth: int = 4,
    lane_dir_candidates: Optional[Iterable[str]] = ('masks', 'lane_masks'),
    require_lane_dir: bool = True,
) -> List[str]:
    found = []
    seen = set()
    for base in search_roots:
        if not base or not os.path.isdir(base):
            continue
        base = os.path.abspath(base)
        for cur, dirs, files in os.walk(base):
            rel = os.path.relpath(cur, base)
            depth = 0 if rel == '.' else rel.count(os.sep) + 1
            if depth > max_depth:
                dirs[:] = []
                continue
            if _has_dataset_layout(cur, lane_dir_candidates=lane_dir_candidates, require_lane_dir=require_lane_dir) and cur not in seen:
                found.append(cur)
                seen.add(cur)
    found.sort(key=lambda p: (p.count(os.sep), len(p)))
    return found


def _candidate_drive_dirs(dataset_name: str, ecocar_root: str) -> List[str]:
    cfg = _read_paths_config(ecocar_root)
    cands = []
    for key in ['dataset_root', 'dataset_dir', 'bdd100k_vehicle5_dir', 'local_dataset_dir']:
        v = cfg.get(key)
        if isinstance(v, str) and v.strip():
            cands.append(v)
    cands += [
        os.path.join(ecocar_root, 'datasets', dataset_name),
        os.path.join(ecocar_root, dataset_name),
    ]
    out = []
    seen = set()
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _candidate_tar_paths(dataset_name: str, ecocar_root: str) -> List[str]:
    return [
        os.path.join(ecocar_root, 'datasets', f'{dataset_name}.tar'),
        os.path.join(ecocar_root, 'datasets', f'{dataset_name}.tar.gz'),
        os.path.join(ecocar_root, f'{dataset_name}.tar'),
        os.path.join(ecocar_root, f'{dataset_name}.tar.gz'),
    ]


def ensure_local_dataset_from_drive(
    dataset_name: str,
    ecocar_root: str,
    local_base: Optional[str] = None,
    force_reextract: bool = False,
    lane_dir_candidates: Optional[Iterable[str]] = ('masks', 'lane_masks'),
    require_lane_dir: bool = True,
) -> str:
    """Return a valid packaged dataset root for the current notebook runtime.

    Priority:
    1) reuse an already-extracted local SSD copy
    2) extract the tar from Drive into /content
    3) fall back to using the Drive directory directly
    """
    if local_base is None:
        local_base = f'/content/{dataset_name}'

    if force_reextract and os.path.isdir(local_base):
        shutil.rmtree(local_base, ignore_errors=True)

    os.makedirs(local_base, exist_ok=True)
    search_roots = [local_base, os.path.join(local_base, dataset_name), '/content']
    existing = _find_dataset_roots(
        search_roots,
        lane_dir_candidates=lane_dir_candidates,
        require_lane_dir=require_lane_dir,
    )
    if existing:
        return existing[0]

    tar_candidates = _candidate_tar_paths(dataset_name, ecocar_root)
    for tar_path in tar_candidates:
        if os.path.isfile(tar_path):
            print(f'Extracting {tar_path} into this notebook runtime ...')
            with tarfile.open(tar_path, 'r:*') as tar:
                tar.extractall('/content', filter='data')
            found = _find_dataset_roots(
                search_roots,
                lane_dir_candidates=lane_dir_candidates,
                require_lane_dir=require_lane_dir,
            )
            if found:
                return found[0]
            break

    for drive_dir in _candidate_drive_dirs(dataset_name, ecocar_root):
        if _has_dataset_layout(drive_dir, lane_dir_candidates=lane_dir_candidates, require_lane_dir=require_lane_dir):
            print(f'Using Drive dataset directory directly: {drive_dir}')
            return drive_dir

    raise FileNotFoundError(
        f'Could not resolve dataset {dataset_name}. Expected one of: {tar_candidates + _candidate_drive_dirs(dataset_name, ecocar_root)}'
    )


def _extract_zip_if_needed(zip_path: str, dest_root: str) -> bool:
    marker = os.path.join(dest_root, f'.extracted_{Path(zip_path).stem}')
    if os.path.exists(marker):
        return True
    if not os.path.isfile(zip_path):
        return False
    os.makedirs(dest_root, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_root)
    Path(marker).write_text('ok\n', encoding='utf-8')
    return True


def find_raw_bdd_root(ecocar_root: str, auto_extract: bool = True) -> str:
    cfg = _read_paths_config(ecocar_root)
    candidates = []
    for k in ['bdd_raw_dir', 'bdd100k_raw', 'bdd_root', 'bdd100k_root', 'raw_bdd100k_dir']:
        v = cfg.get(k)
        if isinstance(v, str) and v.strip():
            candidates.append(v)

    project_root = Path(ecocar_root)
    shared_root = project_root.parent if project_root.name == 'yolop_vehicle_lane' else project_root

    candidates += [
        os.path.join(ecocar_root, 'datasets', 'bdd100k_raw'),
        os.path.join(ecocar_root, 'bdd100k_raw'),
        os.path.join(str(shared_root), 'datasets', 'bdd100k_raw'),
        os.path.join(str(shared_root), 'bdd100k_raw'),
        os.path.join(str(shared_root), 'downloads', 'bdd100k_raw'),
        '/content/bdd100k_raw',
        '/content/bdd100k',
    ]

    ordered_candidates = []
    seen = set()
    for cand in candidates:
        if cand and cand not in seen:
            ordered_candidates.append(cand)
            seen.add(cand)

    for cand in ordered_candidates:
        if not cand or not os.path.isdir(cand):
            continue
        laneish = [
            os.path.join(cand, '100k'),
            os.path.join(cand, 'labels', '100k'),
            os.path.join(cand, 'bdd100k', '100k'),
            os.path.join(cand, 'images', '100k'),
        ]
        if any(os.path.isdir(p) for p in laneish):
            return cand

    if auto_extract:
        raw_root = '/content/bdd100k_raw'
        extracted_any = False
        for downloads in [os.path.join(ecocar_root, 'downloads'), os.path.join(str(shared_root), 'downloads')]:
            label_zip = os.path.join(downloads, 'bdd100k_labels.zip')
            image_zip = os.path.join(downloads, 'bdd100k_images_100k.zip')
            seg_zip = os.path.join(downloads, 'bdd100k_seg_maps.zip')
            extracted_any |= _extract_zip_if_needed(label_zip, raw_root)
            extracted_any |= _extract_zip_if_needed(image_zip, raw_root)
            if os.path.isfile(seg_zip):
                _extract_zip_if_needed(seg_zip, raw_root)

        if extracted_any and os.path.isdir(raw_root):
            return raw_root

    raise FileNotFoundError(f'Could not find raw BDD root. Tried: {ordered_candidates}')


def _first_existing_dir(candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return None


def resolve_bdd_images_100k_dir(raw_bdd_root: str) -> str:
    """Return a directory that contains `train/` and `val/` image folders.

    The raw BDD snapshot can be laid out in several ways — handoff note §3:
      * `raw/images/100k/train`
      * `raw/bdd100k/images/100k/train`
      * `raw/train` (rare, pre-extracted flat layout)
    We try all three. Callers can still override via config.
    """
    if not raw_bdd_root:
        raise FileNotFoundError('raw_bdd_root is empty')
    candidates = [
        os.path.join(raw_bdd_root, 'images', '100k'),
        os.path.join(raw_bdd_root, 'bdd100k', 'images', '100k'),
        os.path.join(raw_bdd_root, '100k'),
        raw_bdd_root,
    ]
    for root in candidates:
        if (os.path.isdir(os.path.join(root, 'train'))
                and os.path.isdir(os.path.join(root, 'val'))):
            return root
    # Softer: return first candidate that even has `train`.
    for root in candidates:
        if os.path.isdir(os.path.join(root, 'train')):
            return root
    raise FileNotFoundError(
        f'Could not find an images/100k-style layout under: {candidates}')


def resolve_bdd_labels_100k_dir(raw_bdd_root: str) -> str:
    """Return a directory holding per-split detection JSONs (see handoff §3).

    Tries `labels/100k/{split}`, `bdd100k/labels/100k/{split}`, and — for
    the old per-image layout — `100k/{split}`.
    """
    if not raw_bdd_root:
        raise FileNotFoundError('raw_bdd_root is empty')
    candidates = [
        os.path.join(raw_bdd_root, 'labels', '100k'),
        os.path.join(raw_bdd_root, 'bdd100k', 'labels', '100k'),
        os.path.join(raw_bdd_root, '100k'),   # old per-image dir layout
    ]
    for root in candidates:
        if (os.path.isdir(os.path.join(root, 'train'))
                and os.path.isdir(os.path.join(root, 'val'))):
            return root
    for root in candidates:
        if os.path.isdir(os.path.join(root, 'train')):
            return root
    raise FileNotFoundError(
        f'Could not find a labels/100k-style layout under: {candidates}')


def find_lane_polygon_jsons(raw_bdd_root: str):
    candidates = {
        'train': [
            os.path.join(raw_bdd_root, 'labels', 'lane', 'polygons', 'lane_train.json'),
            os.path.join(raw_bdd_root, 'bdd100k', 'labels', 'lane', 'polygons', 'lane_train.json'),
            os.path.join(raw_bdd_root, '100k', 'train'),
            os.path.join(raw_bdd_root, 'bdd100k', '100k', 'train'),
        ],
        'val': [
            os.path.join(raw_bdd_root, 'labels', 'lane', 'polygons', 'lane_val.json'),
            os.path.join(raw_bdd_root, 'bdd100k', 'labels', 'lane', 'polygons', 'lane_val.json'),
            os.path.join(raw_bdd_root, '100k', 'val'),
            os.path.join(raw_bdd_root, 'bdd100k', '100k', 'val'),
        ],
    }
    out = {}
    for split, paths in candidates.items():
        out[split] = next((p for p in paths if os.path.isfile(p) or os.path.isdir(p)), None)
    return out
