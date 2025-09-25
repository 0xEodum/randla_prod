import time

import torch
import torch.nn as nn

class _KNNResolver:
    def __init__(self):
        self._loaders = [
            self._load_torch_points,
            self._load_torch_points_kernels,
            self._load_pyg,
        ]
        self._cached_backend = None

    def __call__(self, support, query, k):
        if self._cached_backend is not None:
            try:
                return self._cached_backend(support, query, k)
            except (RuntimeError, ValueError):
                self._cached_backend = None
        last_error = None
        for loader in self._loaders:
            backend = loader()
            if backend is None:
                continue
            try:
                result = backend(support, query, k)
            except (RuntimeError, ValueError) as exc:
                last_error = exc
                continue
            self._cached_backend = backend
            return result
        if last_error is not None:
            raise last_error
        raise ImportError('No supported k-NN backend found. Install torch_points, torch_points_kernels, or torch_geometric.')

    @staticmethod
    def _load_torch_points():
        try:
            from torch_points import knn as backend
        except (ModuleNotFoundError, ImportError):
            return None
        return backend

    @staticmethod
    def _load_torch_points_kernels():
        try:
            from torch_points_kernels import knn as backend
        except (ModuleNotFoundError, ImportError):
            return None
        return backend

    @staticmethod
    def _load_pyg():
        try:
            from torch_geometric.nn import knn as pyg_knn
        except (ModuleNotFoundError, ImportError):
            return None

        def backend(support, query, k):
            return _pyg_knn(pyg_knn, support, query, k)
        return backend


def _pyg_knn(pyg_knn, support, query, k):
    B, Ns, C = support.shape
    _, Nq, _ = query.shape
    if Ns == 0 or Nq == 0:
        raise ValueError('Support and query must contain at least one point.')
    effective_k = min(k, Ns)
    support_flat = support.reshape(B * Ns, C)
    query_flat = query.reshape(B * Nq, C)
    device = support.device
    batch_template = torch.arange(B, device=device, dtype=torch.long)
    batch_support = batch_template.repeat_interleave(Ns)
    batch_query = batch_template.repeat_interleave(Nq)

    row, col = pyg_knn(
        support_flat,
        query_flat,
        effective_k,
        batch_x=batch_support,
        batch_y=batch_query,
        num_workers=0
    )
    if row.numel() != query_flat.size(0) * effective_k:
        raise RuntimeError('PyG knn returned an unexpected number of neighbors.')

    diff = query_flat[col] - support_flat[row]
    dist = diff.pow(2).sum(dim=1).sqrt()

    perm = torch.argsort(col)
    row = row[perm]
    col = col[perm]
    dist = dist[perm]

    expected = torch.arange(query_flat.size(0), device=col.device, dtype=col.dtype).repeat_interleave(effective_k)
    if not torch.equal(col, expected):
        raise RuntimeError('PyG knn returned neighbors in an unexpected order.')

    idx = row.view(B, Nq, effective_k)
    dist = dist.view(B, Nq, effective_k)

    if effective_k < k:
        pad_repeat = k - effective_k
        pad_idx = idx[:, :, -1:].expand(B, Nq, pad_repeat)
        pad_dist = dist[:, :, -1:].expand(B, Nq, pad_repeat)
        idx = torch.cat((idx, pad_idx), dim=-1)
        dist = torch.cat((dist, pad_dist), dim=-1)

    return idx, dist


knn = _KNNResolver()

class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network

            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors, device):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

        self.device = device

    def forward(self, coords, features, knn_output):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d, N, 1)
                features of the point cloud
            neighbors: tuple

            Returns
            -------
            torch.Tensor, shape (B, 2*d, N, K)
        """
        # finding neighboring points
        idx, dist = knn_output
        B, N, K = idx.size()
        # idx(B, N, K), coords(B, N, 3)
        # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2,-1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx) # shape (B, 3, N, K)
        # if USE_CUDA:
        #     neighbors = neighbors.cuda()

        # relative point position encoding
        concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)
        ), dim=-3).to(self.device)
        return torch.cat((
            self.mlp(concat),
            features.expand(B, -1, N, K)
        ), dim=-3)



class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

    def forward(self, x):
        r"""
            Forward pass

            Parameters
            ----------
            x: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.permute(0,2,3,1)).permute(0,3,1,2)

        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True) # shape (B, d_in, N, 1)

        return self.mlp(features)



class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors, device):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out//2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2*d_out)
        self.shortcut = SharedMLP(d_in, 2*d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors, device)
        self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors, device)

        self.pool1 = AttentivePooling(d_out, d_out//2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features, *, cache=None, cache_key=None):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d_in, N, 1)
                features of the point cloud
            cache: dict, optional
                cache used to store previously computed neighborhoods
            cache_key: hashable, optional
                key that helps retrieving cached neighborhoods

            Returns
            -------
            torch.Tensor, shape (B, 2*d_out, N, 1)
        """
        coords_contiguous = coords if coords.is_contiguous() else coords.contiguous()
        features = features.contiguous()

        knn_output = None
        if cache is not None and cache_key is not None:
            effective_key = (cache_key, coords_contiguous.data_ptr())
            knn_output = cache.get(effective_key)
            if knn_output is None:
                knn_output = knn(coords_contiguous, coords_contiguous, self.num_neighbors)
                cache[effective_key] = knn_output
        if knn_output is None:
            knn_output = knn(coords_contiguous, coords_contiguous, self.num_neighbors)

        x = self.mlp1(features)

        x = self.lse1(coords_contiguous, x, knn_output)
        x = self.pool1(x)

        x = self.lse2(coords_contiguous, x, knn_output)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))



class RandLANet(nn.Module):
    def __init__(self, d_in, num_classes, num_neighbors=16, decimation=4, device=torch.device('cpu')):
        super(RandLANet, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors, device),
            LocalFeatureAggregation(32, 64, num_neighbors, device),
            LocalFeatureAggregation(128, 128, num_neighbors, device),
            LocalFeatureAggregation(256, 256, num_neighbors, device)
        ])

        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())

        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.ReLU()
        )
        self.decoder = nn.ModuleList([
            SharedMLP(1024, 256, **decoder_kwargs),
            SharedMLP(512, 128, **decoder_kwargs),
            SharedMLP(256, 32, **decoder_kwargs),
            SharedMLP(64, 8, **decoder_kwargs)
        ])

        # final semantic prediction
        self.fc_end = nn.Sequential(
            SharedMLP(8, 64, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
            nn.Dropout(),
            SharedMLP(32, num_classes)
        )
        self.device = device

        self = self.to(device)

    def forward(self, input):
        r"""
            Forward pass

            Parameters
            ----------
            input: torch.Tensor, shape (B, N, d_in)
                input points

            Returns
            -------
            torch.Tensor, shape (B, num_classes, N)
                segmentation scores for each point
        """
        N = input.size(1)
        d = self.decimation

        coords = input[..., :3].contiguous()
        x = self.fc_start(input).transpose(-2, -1).unsqueeze(-1)
        x = self.bn_start(x)

        knn_cache = {}

        def cached_knn(query, support, k, key):
            support_c = support if support.is_contiguous() else support.contiguous()
            query_c = query if query.is_contiguous() else query.contiguous()
            cache_id = (key, support_c.data_ptr(), query_c.data_ptr(), support_c.shape, query_c.shape, k)
            result = knn_cache.get(cache_id)
            if result is None:
                result = knn(support_c, query_c, k)
                knn_cache[cache_id] = result
            return result

        decimation_ratio = 1
        x_stack = []

        permutation = torch.randperm(N, device=input.device)
        coords = coords.index_select(1, permutation)
        x = x.index_select(2, permutation)

        for layer_idx, lfa in enumerate(self.encoder):
            points_per_layer = max(1, N // decimation_ratio)
            coords_slice = coords[:, :points_per_layer]
            features_slice = x[:, :, :points_per_layer]

            cache_key = ('encoder', layer_idx, coords_slice.data_ptr())
            x = lfa(coords_slice, features_slice, cache=knn_cache, cache_key=cache_key)
            x_stack.append(x.clone())

            decimation_ratio *= d
            next_points = max(1, N // decimation_ratio)
            x = x[:, :, :next_points]

        x = self.mlp(x)

        for layer_idx, mlp in enumerate(self.decoder):
            support_count = max(1, N // decimation_ratio)
            query_count = min(N, max(1, (d * N) // decimation_ratio))

            support = coords[:, :support_count]
            query = coords[:, :query_count]

            cache_key = ('decoder', layer_idx, support_count, query_count)
            neighbors, _ = cached_knn(query, support, 1, cache_key)

            extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)

            x_neighbors = torch.gather(x, -2, extended_neighbors)

            x = torch.cat((x_neighbors, x_stack.pop()), dim=1)

            x = mlp(x)

            decimation_ratio //= d

        inverse_permutation = torch.argsort(permutation)
        x = x.index_select(2, inverse_permutation)

        scores = self.fc_end(x)

        return scores.squeeze(-1)


if __name__ == '__main__':
    import time
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    d_in = 7
    cloud = 1000*torch.randn(1, 2**16, d_in).to(device)
    model = RandLANet(d_in, 6, 16, 4, device)
    # model.load_state_dict(torch.load('checkpoints/checkpoint_100.pth'))
    model.eval()

    t0 = time.time()
    pred = model(cloud)
    t1 = time.time()
    # print(pred)
    print(t1-t0)
