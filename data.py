import pickle, time, warnings
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler, BatchSampler

from utils.tools import Config as cfg
from utils.tools import DataProcessing as DP

class PointCloudsDataset(Dataset):
    def __init__(self, dir, labels_available=True):
        self.paths = sorted(dir.glob('*.npy'))
        self.labels_available = labels_available
        self.ignore_labels = getattr(cfg, 'ignore_labels', []) or []
        self.samples_per_class = getattr(cfg, 'samples_per_class', None)
        self._memmaps = {}
        self._indices = {}
        self._rng = np.random.default_rng()

        if self.labels_available:
            self._prepare_indices()

    def __getitem__(self, idx):
        path = self.paths[idx]
        points, labels = self.load_npy(path)

        points_tensor = torch.from_numpy(points).float()
        if self.labels_available and labels.size > 0:
            labels_tensor = torch.from_numpy(labels).long()
        else:
            labels_tensor = torch.empty((0,), dtype=torch.long)

        return points_tensor, labels_tensor

    def __len__(self):
        return len(self.paths)

    def __del__(self):
        self._close_memmaps()

    def _prepare_indices(self):
        for path in self.paths:
            memmap = self._get_memmap(path)
            cloud = self._normalize_shape(memmap)

            labels = cloud[:, -1]
            valid_mask = np.ones(labels.shape[0], dtype=bool)
            if self.ignore_labels:
                valid_mask &= ~np.isin(labels, self.ignore_labels)

            valid_indices = np.nonzero(valid_mask)[0]
            if valid_indices.size == 0:
                self._indices[path] = valid_indices
                continue

            filtered_labels = labels[valid_indices]

            if self.samples_per_class and self.samples_per_class > 0:
                selected = []
                for label in np.unique(filtered_labels):
                    label_indices = valid_indices[filtered_labels == label]
                    if label_indices.size == 0:
                        continue
                    if label_indices.size > self.samples_per_class:
                        label_indices = self._rng.choice(label_indices, self.samples_per_class, replace=False)
                    selected.append(label_indices)
                if selected:
                    valid_indices = np.concatenate(selected)

            self._indices[path] = np.sort(valid_indices.astype(np.int64, copy=False))

    def _close_memmaps(self):
        for memmap in self._memmaps.values():
            mmap_obj = getattr(memmap, '_mmap', None)
            if mmap_obj is not None:
                mmap_obj.close()
        self._memmaps.clear()

    def _get_memmap(self, path):
        path = Path(path)
        memmap = self._memmaps.get(path)
        if memmap is None:
            memmap = np.load(path, mmap_mode='r')
            self._memmaps[path] = memmap
        return memmap

    @staticmethod
    def _normalize_shape(array):
        if array.ndim != 2:
            raise ValueError('Point cloud array is expected to be 2D')
        if array.shape[1] < array.shape[0]:
            return array.T
        return array

    def load_npy(self, path):
        memmap = self._get_memmap(path)
        cloud = self._normalize_shape(memmap)

        if self.labels_available:
            points = cloud[:, :-1]
            labels = cloud[:, -1]
            indices = self._indices.get(Path(path))
            if indices is not None and indices.size > 0:
                points = points[indices]
                labels = labels[indices]
            elif self.ignore_labels:
                mask = ~np.isin(labels, self.ignore_labels)
                points = points[mask]
                labels = labels[mask]

            points = np.ascontiguousarray(points, dtype=np.float32)
            labels = np.ascontiguousarray(labels, dtype=np.int64)
        else:
            points = np.ascontiguousarray(cloud, dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)

        return points, labels

class CloudsDataset(Dataset):
    def __init__(self, dir, data_type='npy'):
        self.root = Path(dir)
        self.data_type = data_type
        self.validation_pattern = getattr(cfg, 'validation_split_pattern', '1_')
        self.ignore_labels = getattr(cfg, 'ignore_labels', []) or []

        self.samples = {'training': [], 'validation': []}
        self._memmaps = {}
        self._trees = {}

        self.load_data()
        self.size = sum(len(items) for items in self.samples.values())
        print('Size of training : ', len(self.samples['training']))
        print('Size of validation : ', len(self.samples['validation']))

    def __len__(self):
        return sum(len(items) for items in self.samples.values())

    def load_data(self):
        for npy_file in sorted(self.root.glob(f'*.{self.data_type}')):
            t0 = time.time()
            cloud_name = npy_file.stem
            split = 'validation' if self.validation_pattern and self.validation_pattern in cloud_name else 'training'

            kd_tree_file = self.root / f'{cloud_name}_KDTree.pkl'
            proj_file = self.root / f'{cloud_name}_proj.pkl'

            point_count, valid_point_count, class_histogram = self._inspect_sample(npy_file)

            sample_info = {
                'name': cloud_name,
                'npy_path': npy_file,
                'tree_path': kd_tree_file,
                'proj_path': proj_file if proj_file.exists() else None,
                'point_count': point_count,
                'valid_point_count': valid_point_count,
                'class_histogram': class_histogram,
            }

            if not kd_tree_file.exists():
                warnings.warn(f'KDTree file missing for {cloud_name}: {kd_tree_file}')
            self.samples[split].append(sample_info)

            approx_size_mb = point_count * 4 * 7 * 1e-6
            print(f'{kd_tree_file.name} metadata ready in {time.time() - t0:.1f}s (approx {approx_size_mb:.1f} MB)')

    def load_sample_data(self, split, index, *, cache=True):
        sample = self.samples[split][index]

        memmap = self._load_memmap(sample['npy_path'], cache)
        cloud = self._normalize_cloud(memmap)

        points = cloud[:, :3].astype(np.float32, copy=False)
        colors = cloud[:, 3:6].astype(np.float32, copy=False)
        labels = cloud[:, -1].astype(np.int64, copy=False)

        tree = self._load_tree(sample['tree_path'], cache)

        return {
            'name': sample['name'],
            'points': points,
            'colors': colors,
            'labels': labels,
            'tree': tree,
        }

    def close(self):
        for memmap in self._memmaps.values():
            mmap_obj = getattr(memmap, '_mmap', None)
            if mmap_obj is not None:
                mmap_obj.close()
        self._memmaps.clear()
        self._trees.clear()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _inspect_sample(self, npy_path):
        memmap = np.load(npy_path, mmap_mode='r')
        cloud = self._normalize_cloud(memmap)
        point_count = cloud.shape[0]
        labels = np.array(cloud[:, -1], copy=True)
        valid_mask = np.ones(labels.shape[0], dtype=bool)
        if self.ignore_labels:
            valid_mask &= ~np.isin(labels, self.ignore_labels)
        filtered_labels = labels[valid_mask]
        unique, counts = (np.unique(filtered_labels.astype(np.int64), return_counts=True)
                          if filtered_labels.size else (np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
        histogram = {int(label): int(count) for label, count in zip(unique, counts)}
        valid_point_count = int(filtered_labels.shape[0])
        mmap_obj = getattr(memmap, '_mmap', None)
        if mmap_obj is not None:
            mmap_obj.close()
        return int(point_count), valid_point_count, histogram

    def _load_memmap(self, path, cache):
        path = Path(path)
        if cache and path in self._memmaps:
            return self._memmaps[path]
        memmap = np.load(path, mmap_mode='r')
        if cache:
            self._memmaps[path] = memmap
        return memmap

    def _load_tree(self, path, cache):
        path = Path(path)
        if cache and path in self._trees:
            return self._trees[path]
        with open(path, 'rb') as f:
            tree = pickle.load(f)
        if cache:
            self._trees[path] = tree
        return tree

    @staticmethod
    def _normalize_cloud(array):
        if array.ndim != 2:
            raise ValueError('Point cloud array is expected to be 2D')

        rows, cols = array.shape
        if rows == 0 or cols == 0:
            return array

        feature_dims = {3, 4, 6, 7, 8, 9, 10, 13}
        if rows in feature_dims and cols not in feature_dims:
            return array.T
        if cols in feature_dims and rows not in feature_dims:
            return array

        if rows == cols:
            return array
        if rows < cols:
            return array.T
        return array

class ActiveLearningSampler(IterableDataset):

    def __init__(self, dataset, batch_size=6, split='training'):
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.samples = list(self.dataset.samples.get(split, []))
        self._rng = np.random.default_rng()

        if split == 'training':
            self.n_samples = cfg.train_steps
        else:
            self.n_samples = cfg.val_steps

        self.possibility = []
        self.min_possibility = []
        for sample in self.samples:
            count = int(sample.get('point_count', 0))
            if count <= 0:
                self.possibility.append(np.array([], dtype=np.float32))
                self.min_possibility.append(np.inf)
                continue
            poss = self._rng.random(count) * 1e-3
            self.possibility.append(poss)
            self.min_possibility.append(float(np.min(poss)))

    def __iter__(self):
        return self.spatially_regular_gen()

    def __len__(self):
        return self.n_samples

    def spatially_regular_gen(self):
        num_iterations = self.n_samples * self.batch_size
        for _ in range(num_iterations):
            if cfg.sampling_type == 'active_learning':
                cloud_idx = int(np.argmin(self.min_possibility))
                possibilities = self.possibility[cloud_idx]
                if possibilities.size == 0 or not np.isfinite(self.min_possibility[cloud_idx]):
                    self.min_possibility[cloud_idx] = np.inf
                    continue

                point_ind = int(np.argmin(possibilities))
                sample_data = self.dataset.load_sample_data(self.split, cloud_idx)
                tree = sample_data['tree']
                points_xyz = np.asarray(tree.data, copy=False)

                center_point = points_xyz[point_ind, :].reshape(1, -1)
                noise = self._rng.normal(scale=3.5 / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                neighbor_count = min(points_xyz.shape[0], cfg.num_points)
                queried_idx = tree.query(pick_point, k=neighbor_count)[1][0]
                queried_idx = DP.shuffle_idx(queried_idx)

                queried_pc_xyz = points_xyz[queried_idx] - pick_point
                queried_pc_colors = sample_data['colors'][queried_idx]
                queried_pc_labels = sample_data['labels'][queried_idx]

                if points_xyz.shape[0] < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels =                         DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

                dists = np.sum(np.square((points_xyz[queried_idx] - pick_point).astype(np.float32)), axis=1)
                max_dist = np.max(dists) if dists.size else 0.0
                if max_dist > 0:
                    delta = np.square(1 - dists / max_dist)
                else:
                    delta = np.zeros_like(dists)
                possibilities[queried_idx] += delta
                self.min_possibility[cloud_idx] = float(np.min(possibilities))

            elif cfg.sampling_type == 'random':
                if not self.samples:
                    continue
                cloud_idx = int(self._rng.integers(len(self.samples)))
                sample_data = self.dataset.load_sample_data(self.split, cloud_idx)
                points_xyz = np.asarray(sample_data['tree'].data, copy=False)

                if points_xyz.shape[0] == 0:
                    continue

                neighbor_count = min(points_xyz.shape[0], cfg.num_points)
                queried_idx = self._rng.choice(points_xyz.shape[0], neighbor_count, replace=False)

                queried_pc_xyz = points_xyz[queried_idx]
                queried_pc_colors = sample_data['colors'][queried_idx]
                queried_pc_labels = sample_data['labels'][queried_idx]

                if points_xyz.shape[0] < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels =                         DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

            else:
                raise ValueError(f"Unsupported sampling type: {cfg.sampling_type}")

            queried_pc_xyz = np.ascontiguousarray(queried_pc_xyz, dtype=np.float32)
            queried_pc_colors = np.ascontiguousarray(queried_pc_colors, dtype=np.float32)
            queried_pc_labels = np.ascontiguousarray(queried_pc_labels, dtype=np.int64)

            points = torch.from_numpy(np.concatenate((queried_pc_xyz, queried_pc_colors), axis=1)).float()
            labels = torch.from_numpy(queried_pc_labels).long()

            yield points, labels

def data_loaders(dir, sampling_method='active_learning', **kwargs):
    if sampling_method == 'active_learning':
        dataset = CloudsDataset(dir / 'train')
        batch_size = kwargs.get('batch_size', 6)
        val_sampler = ActiveLearningSampler(
            dataset,
            batch_size=batch_size,
            split='validation'
        )
        train_sampler = ActiveLearningSampler(
            dataset,
            batch_size=batch_size,
            split='training'
        )
        return DataLoader(train_sampler, **kwargs), DataLoader(val_sampler, **kwargs)

    if sampling_method == 'naive':
        train_dataset = PointCloudsDataset(dir / 'train')
        val_dataset = PointCloudsDataset(dir / 'val')
        return DataLoader(train_dataset, shuffle=True, **kwargs), DataLoader(val_dataset, **kwargs)

    raise ValueError(f"Dataset sampling method '{sampling_method}' does not exist.")

if __name__ == '__main__':
    dataset = CloudsDataset(Path('datasets') / 's3dis' / 'subsampled' / 'train')
    sampler = ActiveLearningSampler(dataset)
    points, labels = next(iter(sampler))
    print('Points batch shape:', points.shape)
    print('Labels batch shape:', labels.shape)
