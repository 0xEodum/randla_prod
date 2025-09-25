import time

import torch
import torch.nn as nn

try:
    from torch_points import knn as _knn_backend
except (ModuleNotFoundError, ImportError):
    _knn_backend = None



def _fallback_knn(support, query, k, *, chunk_size=2048):
    """Compute k-NN using torch.cdist as a backend."""
    if support.dim() != 3 or query.dim() != 3:
        raise ValueError('Support and query tensors must have shape (B, N, C)')

    if support.size(-1) != query.size(-1):
        raise ValueError('Support and query tensors must have the same feature dimension')

    batch_size, query_count, _ = query.shape
    support_count = support.size(1)
    if support_count == 0:
        raise ValueError('Support set must contain at least one point')

    effective_k = min(k, support_count)

    device = query.device
    dtype = query.dtype

    indices = torch.empty((batch_size, query_count, effective_k), device=device, dtype=torch.long)
    distances = torch.empty((batch_size, query_count, effective_k), device=device, dtype=dtype)

    chunk_size = max(1, min(chunk_size, query_count))

    for start in range(0, query_count, chunk_size):
        end = min(start + chunk_size, query_count)
        query_chunk = query[:, start:end, :]
        try:
            dist = torch.cdist(query_chunk, support, compute_mode='donot_care')
        except (TypeError, ValueError):
            dist = torch.cdist(query_chunk, support)
        chunk_dist, chunk_idx = torch.topk(dist, effective_k, dim=-1, largest=False)
        indices[:, start:end, :] = chunk_idx
        distances[:, start:end, :] = chunk_dist

    if effective_k < k:
        repeat_count = k - effective_k
        last_indices = indices[:, :, -1:].expand(-1, -1, repeat_count)
        last_distances = distances[:, :, -1:].expand(-1, -1, repeat_count)
        indices = torch.cat((indices, last_indices), dim=-1)
        distances = torch.cat((distances, last_distances), dim=-1)

    return indices, distances


def knn(support, query, k):
    global _knn_backend

    support = support.contiguous()
    query = query.contiguous()

    if support.size(1) < k:
        return _fallback_knn(support, query, k)

    if _knn_backend is not None:
        try:
            return _knn_backend(support, query, k)
        except (RuntimeError, ValueError) as exc:
            message = str(exc)
            if 'CUDA version not implemented' not in message:
                raise
            _knn_backend = None

    return _fallback_knn(support, query, k)

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


