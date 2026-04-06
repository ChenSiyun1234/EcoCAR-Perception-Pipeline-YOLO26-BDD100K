import os
import shutil
import tarfile
from typing import Optional, List


def _has_dataset_layout(root: str) -> bool:
    return os.path.isdir(os.path.join(root, 'images', 'train')) and os.path.isdir(os.path.join(root, 'labels', 'train'))


def _find_dataset_roots(search_roots: List[str], max_depth: int = 4) -> List[str]:
    found = []
    seen = set()
    for base in search_roots:
        if not os.path.isdir(base):
            continue
        base = os.path.abspath(base)
        for cur, dirs, files in os.walk(base):
            rel = os.path.relpath(cur, base)
            depth = 0 if rel == '.' else rel.count(os.sep) + 1
            if depth > max_depth:
                dirs[:] = []
                continue
            if _has_dataset_layout(cur):
                if cur not in seen:
                    found.append(cur)
                    seen.add(cur)
    found.sort(key=lambda p: (p.count(os.sep), len(p)))
    return found


def _sync_paths_config(dataset_root: str, global_paths_cfg: str) -> None:
    local_paths_cfg = os.path.join(dataset_root, 'paths_config.yaml')
    if os.path.isfile(local_paths_cfg):
        shutil.copy2(local_paths_cfg, global_paths_cfg)
        print(f'Synced paths_config -> {global_paths_cfg}')


def ensure_local_dataset_from_drive(
    dataset_name: str,
    ecocar_root: str,
    local_base: Optional[str] = None,
    force_reextract: bool = False,
) -> str:
    """Prepare a per-notebook local SSD copy from Drive and return the true dataset root.

    This function is notebook-safe: every notebook/runtime must call it independently.
    It never assumes another notebook's /content state exists.
    """
    if local_base is None:
        local_base = f'/content/{dataset_name}'

    dataset_drive = os.path.join(ecocar_root, 'datasets', dataset_name)
    dataset_tar = os.path.join(ecocar_root, 'datasets', f'{dataset_name}.tar')
    global_paths_cfg = os.path.join(ecocar_root, 'paths_config.yaml')

    if force_reextract and os.path.isdir(local_base):
        shutil.rmtree(local_base, ignore_errors=True)

    os.makedirs(local_base, exist_ok=True)

    search_roots = [local_base, os.path.join(local_base, dataset_name), '/content']
    existing = _find_dataset_roots(search_roots)
    if existing:
        root = existing[0]
        _sync_paths_config(root, global_paths_cfg)
        print(f'Notebook-local dataset root: {root}')
        return root

    if os.path.isfile(dataset_tar):
        print(f'Extracting {dataset_tar} into this notebook runtime ...')
        with tarfile.open(dataset_tar, 'r') as tar:
            tar.extractall('/content', filter='data')
        print('Done.')
    elif _has_dataset_layout(dataset_drive):
        print(f'Using Drive dataset directory directly: {dataset_drive}')
        return dataset_drive
    else:
        raise FileNotFoundError(
            f'Dataset not found on Drive. Expected {dataset_tar} or a valid dataset dir at {dataset_drive}'
        )

    found = _find_dataset_roots(search_roots)
    if found:
        root = found[0]
        _sync_paths_config(root, global_paths_cfg)
        print(f'Notebook-local dataset root: {root}')
        return root

    raise FileNotFoundError(
        'Dataset archive extracted, but no valid dataset root was found. '
        f'Searched under: {search_roots}. '
        'Expected a directory containing images/train and labels/train.'
    )
