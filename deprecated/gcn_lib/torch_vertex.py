# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath

from torch_kmeans import KMeans
from torch_kmeans.utils.distances import LpDistance

class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])        # j -> i update, i are the target nodes to update
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])    # neighbors
        else:
            x_j = batched_index_select(x, edge_index[0])    # neighbors
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)     # B, C, N, 1
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _) ## DP: NOTE this will "alternately mix-up the features instead of linearly aligning along channels, naively"
        return self.nn(x)


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])      # B, C, N, K
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])   # B, C, N, K
        else: 
            x_j = batched_index_select(x, edge_index[0])   # B, C, N, K
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)   # B, C, N, 1 (updated tensor)
        return max_value
    

class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])    # B, C, N, K
        else:
            x_j = batched_index_select(x, edge_index[0])    # B, C, N, K  
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)     # B, C, N, 1
        return self.nn2(torch.cat([x, x_j], dim=1))       # B, C, N, 1


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)            # B, C, N, 1
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1,
                 
                 init_method="rnd", num_init=8, max_iter=100, distance=LpDistance, p_norm=2, tol=1e-4, normalize=None, n_clusters=4, verbose=False, seed=None):
        
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r      # What is r? Why is it needed?
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

        self.init_method_kmeans = init_method
        self.num_init_kmeans = num_init
        self.max_iter_kmeans = max_iter
        self.distance_kmeans = distance
        self.p_norm_kmeans = p_norm
        self.tol_kmeans = tol
        self.normalize_kmeans = normalize
        self.n_clusters_kmeans = n_clusters
        self.verbose_kmeans = verbose
        self.seed_kmeans = seed
        
        self.cluster = KMeans(init_method=init_method,
                              num_init=num_init,
                              max_iter=max_iter,
                              distance=distance,
                              p_norm=p_norm,
                              tol=tol,
                              normalize=normalize,
                              n_clusters=n_clusters,
                              verbose=verbose,
                              seed=seed)

    def forward(self, x: torch.Tensor, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)          # B, C, H', W'
            y = y.reshape(B, C, -1, 1).contiguous()      # B, C, H'W' = N', 1      
        x = x.reshape(B, C, -1, 1).contiguous()          # B, C, HW = N, 1

        x = x.squeeze(-1)                               # B, C, HW = N
        x = x.transpose(1, 2)                           # B, N = HW, C

        cluster_output = self.cluster(x=x)
        labels, centroids = cluster_output.labels, cluster_output.centers      # labels -> B, N | centroids -> B, clusters, C

        pass
    
        edge_index = self.dilated_knn_graph(x, y, relative_pos)    # y is a "reduced-edge-set" obtained through pooling - edges for nodes in x are with respect to nodes in y and edge features correspond to node-indexing with respect to y
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)   # B, C', HW = N, 1  
        return x.reshape(B, -1, H, W).contiguous() # B, C', H, W


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)    # 1, 1, HW, HW
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)   # 1, n, n/(r^2) NOTE the negation!

    def _get_relative_pos(self, relative_pos, H, W):
        # What is going on here?
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            # relative_pos not NONE and H*W != self.n (H, W mismatches with total tokens)
            # WE need relative_pos and H*W mismatch with respect to total tokens
            # interpolate nominal self.n (with h_base*w_base) to current N (H*W, current one)
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x                    # B, C, H, W
        x = self.fc1(x)             # B, C, H, W          
        B, C, H, W = x.shape    
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)   # 1, HW = N, H'W' = N'
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x
