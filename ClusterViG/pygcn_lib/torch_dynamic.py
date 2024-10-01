from typing import Callable, Optional, Union

import math

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor

from torch_kmeans import KMeans
from torch_kmeans.utils.distances import LpDistance
from torch_cluster import knn, knn_graph

from .torch_centroid import ArgsUpCentroidsEdgeConv, ArgsUpCentroidsGATConv
from .torch_vertex import StaticGraphConv, ArgsStaticGraphConv

from timm.models.layers import DropPath


'''
torch_dynamic is the top-level that ties everything together
'''


def organize_and_map(x_: Tensor, batch_: Tensor):

    '''
    x_ -> B X N X D
    batch_ -> B X N

    Copied from ClusterConv
    '''

    total_clusters = torch.max(batch_) + 1
    total_graphs = batch_.shape[0]
    device = batch_.device

    _mapping_indices = batch_.detach().clone()

    for _graph_i in range(total_graphs):
        batch_current_reference = batch_[_graph_i]
        _mapping_indices_current_to_modify = _mapping_indices[_graph_i]
        count = 0
        for _cluster_j in range(total_clusters):
            slice = (batch_current_reference == _cluster_j)
            elements = torch.sum(slice)
            vals = torch.arange(start=count,
                                end=count + elements,
                                dtype=torch.int64,
                                device=device)
            _mapping_indices_current_to_modify[slice] = vals
            count = count + elements

    _mapping_indices_for_X = _mapping_indices.unsqueeze(dim=2)
    _mapping_indices_for_X = _mapping_indices_for_X.expand(_mapping_indices_for_X.shape[0], _mapping_indices_for_X.shape[1], x_.shape[-1])

    
    batch_unshuffled_ = torch.full(size=batch_.shape,
                                   fill_value=-1,
                                   dtype=batch_.dtype,
                                   device=batch_.device)
           
    batch_unshuffled_.scatter_(dim=1,
                               index=_mapping_indices, 
                               src=batch_)
    
    x_unshuffled_ = torch.full(size=x_.shape,
                               fill_value=0.0,
                               dtype=x_.dtype,
                               device=x_.device)
            
    x_unshuffled_.scatter_(dim=1,
                           index=_mapping_indices_for_X,
                           src=x_)

    
    return (x_unshuffled_,                          # unshuffled  [B, N, D]
            batch_unshuffled_,                      # unshuffled  [B, N]
            _mapping_indices)                       # map         [B, N]




def get_processed_edge_index_deprecated_(x: Tensor,
                                        batch: Tensor,
                                        normalize: bool,       # normalizing tensors prior to computing neighbors
                                        k: int,                # neighbors
                                        d: int,                # dilation
                                        stoch: bool,           # stochastic
                                        eps: float,            # epsilon
                                        drop: float,           # drop edge rate
                                        method: str,           # ['simple', 'dropout', 'dilated'] 
                                        training: bool,        # whether we are training or not
                                        flow: str,             # flow for edges
                                        ):
    
    '''
    NOTE that reduction not supported !

    method = 'simple'
    - generic knn 
    - train: neighbors = k
    - test:  neighbors = k

    method = 'dropout'
    - drop edges
    - train: neighbors = k/(1 - drop); drop each edge/neighbor with probability drop
    - test:  neighbors = k 

    method = 'dilated'
    - dilated knn
    - replica of dilated knn in DeepGCNs
    
    Initial implementation grounds things at min_nodes (Not optimized) (ensures homogeneity in edge_index tensor)

    `inputs`
    x:              [N, C]
    batch:          [N, ]

    `returns`
    edge_index:     [2, E]
    
    '''

    device = x.device

    if not method in ['simple', 'dropout', 'dilated']:
        raise Exception(f'method must be in [simple, dropout, dilated] but {method} was passed !!')
    
    if not flow in ['source_to_target', 'target_to_source']:
        raise Exception(f'flow must be in [source_to_target, target_to_source] but {flow} was passed !!')

    # Find min_nodes for neighbor bound    
    node_counts_in_subgraphs = torch.bincount(batch)
    min_nodes = torch.min(node_counts_in_subgraphs)

    # Normalize
    x = F.normalize(x, p=2.0, dim=-1) if normalize else x

    #------------------------- SIMPLE -----------------------------------------------------------
    if method == 'simple':
        edge_index = knn(x=x, y=x, k=int(min(k, min_nodes)), batch_x=batch, batch_y=batch)
    #------------------------- DROPOUT ----------------------------------------------------------
    elif method == 'dropout':
        # unrolled for readability
        if training:
            k_ = k / (1 - drop)
            if math.floor(k_) == math.ceil(k_):
                # k_ is int
                edge_index = knn(x=x, y=x, k=int(min(k_, min_nodes)), batch_x=batch, batch_y=batch)
            else:
                # k_ is not int
                if k_ <= min_nodes:
                    # mix
                    if (torch.rand(1) < (math.ceil(k_) - k_)):
                        # jump in with ceil(k_) - k_ probability
                        k_sel = math.floor(k_)
                    else:
                        # jump in with k_ - floor(k_) probability
                        k_sel = math.ceil(k_)
                    edge_index = knn(x=x, y=x, k=k_sel, batch_x=batch, batch_y=batch)
                else:
                    # k_ > min_nodes
                    edge_index = knn(x=x, y=x, k=min_nodes, batch_x=batch, batch_y=batch)
        else:
            # testing
            edge_index = knn(x=x, y=x, k=int(min(k, min_nodes)), batch_x=batch, batch_y=batch)

    #------------------------- DILATED ----------------------------------------------------------
    elif method == 'dilated':
        
        if k*d <= min_nodes:
            edge_index_raw = knn(x=x, y=x, k=k*d, batch_x=batch, batch_y=batch)        # 2, E = nodes*(neighbors = k*d)
            edge_index_raw = edge_index_raw.reshape(2, -1, k*d)                        # 2, nodes, neighbors = k*d
            if stoch:
                if (torch.rand(1) < eps) and training:
                    # jump in with eps probability when training
                    idx = torch.randperm(k*d)[:k]
                    edge_index = edge_index_raw[:, :, idx].reshape(2, -1)
                else:
                    # jump in with 1 - eps probability or when not training
                    edge_index = edge_index_raw[:, :, ::d].reshape(2, -1)
            else:
                edge_index = edge_index_raw[:, :, ::d].reshape(2, -1)
        
        else:
            # k*d > min_nodes
            if k >= min_nodes:
                # saturate
                edge_index = knn(x=x, y=x, k=min_nodes, batch_x=batch, batch_y=batch)
            else:
                # k < min_nodes and k*d > min_nodes
                d_ = min_nodes // k                                                         # >= 1 (integral) and < d (input), maximum d that is in bound
                # repeat with smaller d -> d_
                edge_index_raw = knn(x=x, y=x, k=k*d_, batch_x=batch, batch_y=batch)        # 2, E = nodes*(neighbors = k*d_)
                edge_index_raw = edge_index_raw.reshape(2, -1, k*d_)                        # 2, nodes, neighbors = k*d_
                if stoch:
                    if (torch.rand(1) < eps) and training:
                        # jump in with eps probability when training
                        idx = torch.randperm(k*d_)[:k]
                        edge_index = edge_index_raw[:, :, idx].reshape(2, -1)
                    else:
                        # jump in with 1 - eps probability or when not training
                        edge_index = edge_index_raw[:, :, ::d_].reshape(2, -1)
                else:
                    edge_index = edge_index_raw[:, :, ::d_].reshape(2, -1)                
    #------------------------ EXCEPTION ---------------------------------------------------------
    else:
        raise Exception(f'method: {method} unsupported')
    
    
    
    if flow == 'source_to_target':
        return edge_index.flip([0])
    else:
        # flow == 'target_to_source'
        return edge_index
    
    





def get_processed_edge_index(x: Tensor,
                             batch: Tensor,
                             normalize: bool,       # normalizing tensors prior to computing neighbors
                             k: int,                # neighbors
                             d: int,                # dilation
                             stoch: bool,           # stochastic
                             eps: float,            # epsilon
                             drop: float,           # drop edge rate
                             method: str,           # ['simple', 'dropout', 'dilated'] 
                             training: bool,        # whether we are training or not
                             flow: str,             # flow for edges
                             ):
    '''
    Overriding with standard/generic knn which will lead to NON-HOMOGENEOUS neighbor set per node per sub-graph based on
    sub-graph size and node count

    No dilation,
        d, stoch, eps
    No dropout,
        drop
    method, training irrelevant

    '''
    x = F.normalize(x, p=2.0, dim=-1) if normalize else x
    edge_index = knn(x=x, y=x, k=k, batch_x=batch, batch_y=batch)

    if flow == 'source_to_target':
        return edge_index.flip([0])
    else:
        # flow == 'target_to_source'
        return edge_index












class VertexParams:
    '''Internal Use, DO NOT expose'''
    def __init__(self,
                 
                 conv,

                 in_channels,
                 out_channels,
    
                 groups,
                 dropout,
                 act,
                 norm,
                
                 aggr,                 
                 **kwargs):
        conv = conv.lower()
        self.conv = conv

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.groups = groups
        self.dropout = dropout
        self.act = act
        self.norm = norm

        self.aggr = aggr
        self.kwargs = kwargs





class CenterParams:
    '''Internal Use, DO NOT expose'''
    def __init__(self,
                 
                 conv,
                 
                 num_centroids,
                 
                 in_channels,
                 out_channels,
                 
                 dropout,
                 
                 info,
                 
                 aggr,
                 **kwargs):
        conv = conv.lower()
        self.conv = conv

        self.num_centroids = num_centroids

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.aggr = aggr
        self.kwargs = kwargs

        self.dropout = dropout

        '''
        conv = 'edge'
        --------------------------------------------------------------------------------------------
        info = {'groups': __, 'act': __, 'norm': __}
        ============================================================================================
        conv = 'gat'
        --------------------------------------------------------------------------------------------
        info = {'heads': __, 'concat': __, 'negative_slope': __, 'version': __}
        '''
        if conv == 'edge':
            self.groups, self.act, self.norm = info['groups'], info['act'], info['norm']
        elif conv == 'gat':
            self.heads, self.concat, self.negative_slope, self.version = info['heads'], info['concat'], info['negative_slope'], info['version']
        else:
            raise Exception(f'conv must be gat or edge but {conv} was passed !!')
    


class ArgsClusterKMeans:
    def __init__(self,
                 init_method='rnd', 
                 num_init=8, 
                 max_iter=100, 
                 distance=LpDistance,
                 p_norm=2, 
                 tol=1e-4, 
                 normalize=True, 
                 verbose=False, 
                 seed=None, 
                 **kwargs):
        self.init_method = init_method
        self.num_init = num_init
        self.max_iter = max_iter
        self.distance = distance
        self.p_norm = p_norm
        self.tol = tol
        self.normalize = normalize
        self.verbose = verbose
        self.seed = seed
        self.kwargs = kwargs
    
    def parse_args(self):
        args_dict = {
                     'init_method': self.init_method, 
                     'num_init': self.num_init, 
                     'max_iter': self.max_iter, 
                     'distance': self.distance,
                     'p_norm': self.p_norm, 
                     'tol': self.tol, 
                     'normalize': self.normalize, 
                     'verbose': self.verbose, 
                     'seed': self.seed,
                     **self.kwargs
                    }
        return args_dict








class DynamicGraphConv(StaticGraphConv):
    
    def __init__(self,
                 
                 args_gconv: ArgsStaticGraphConv,
                 conv_gconv: str,
                 
                 neighbors: int,
                 dilation: int,
                 normalize_for_edges: bool,
                 stochastic: bool,
                 epsilon: float,
                 drop_rate_neighbors: float,
                 method_for_edges: str,
                 
                 args_cluster: ArgsClusterKMeans,
                 reduction: Optional[int] = None,
                 ):
        
        super().__init__(args_staticGraphConv=args_gconv,
                         conv_staticGraphConv=conv_gconv)
        
        conv_gconv = conv_gconv.lower()
        


        '''
        GraphConv Level Arguments (Vertex)
        '''
        self.vertex_ = VertexParams(
                                    conv            =       conv_gconv,                         # GraphConv used
                                    
                                    in_channels     =       args_gconv.in_channels,             # Net input channels        (vertex)
                                    out_channels    =       args_gconv.out_channels,            # Net output channels       (vertex)
                                    
                                    groups          =       args_gconv.groups,                  # groups    for     StandardGConv in GraphConv used  
                                    dropout         =       args_gconv.dropout,                 # dropout   for     StandardGConv in GraphConv used
                                    act             =       args_gconv.act,                     # act       for     StandardGConv in GraphConv used
                                    norm            =       args_gconv.norm,                    # norm      for     StandardGConv in GraphConv used
                                    
                                    aggr            =       args_gconv.aggr,                    # aggregation       for     MessagePassing in GraphConv used
                                    
                                    **(args_gconv.kwargs)                                       # kwargs            for     MessagePassing in GraphConv used 
                                    # .kwargs
                                   )
        


        '''
        CenterConv Level Arguments (Center)
        '''
        if args_gconv.conv_centroid == 'edge':
            info_center = {'groups': args_gconv.args_centroid.groups, 'act': args_gconv.args_centroid.act, 'norm': args_gconv.args_centroid.norm}
            out_channels_center = args_gconv.args_centroid.out_channels
        elif args_gconv.conv_centroid == 'gat':
            info_center = {'heads': args_gconv.args_centroid.heads, 'concat': args_gconv.args_centroid.concat, 'negative_slope': args_gconv.args_centroid.negative_slope, 'version': args_gconv.args_centroid.version}
            out_channels_center = args_gconv.args_centroid.out_channels_total
        else:
            raise Exception(f'conv_centroid must be "gat" or "edge" but "{args_gconv.conv_centroid}" was passed !!!')
        

        self.center_ = CenterParams(
                                    conv                =       args_gconv.conv_centroid,                           # conv used                 for     CENTERS
                                        
                                    num_centroids       =       args_gconv.args_centroid.num_centroids,             # total centroids/clusters  for     CLUSTERING
                                    
                                    in_channels         =       args_gconv.args_centroid.in_channels,               # input channels            for     CENTERS
                                    out_channels        =       out_channels_center,                                # output channels           for     CENTERS
                                    
                                    dropout             =       args_gconv.args_centroid.dropout,                   # dropout                   for     CENTERS
                                    
                                    info                =       info_center,                                        # info (unrolled inside)    for     CENTERS     
                                    # .groups/.act/.norm                        (edge)
                                    # .heads/.concat/.negative_slope/.version   (gat) 

                                    aggr                =       args_gconv.args_centroid.aggr,                      # aggregation               for     MessagePassing for CENTERS  
                                    
                                    **(args_gconv.args_centroid.kwargs)                                             # kwargs                    for     MessagePassing for CENTERS
                                    # .kwargs
                                    )
        


        '''
        Module Globals
        '''
        self.neighbors = neighbors
        self.dilation = dilation
        self.normalize_for_edges = normalize_for_edges
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.drop_rate_neighbors = drop_rate_neighbors
        self.method_for_edges = method_for_edges

        self.reduction = reduction              # NOTE is not supported !!!

        if reduction is not None:
            raise NotImplementedError('reduction in current version is not implemented !!!')
        

        '''
        Clustering
        '''
        if isinstance(args_cluster, ArgsClusterKMeans):
            self.cluster = KMeans(n_clusters=self.center_.num_centroids,
                                  **args_cluster.parse_args())
        else:
            raise NotImplementedError(f'Only KMeans implemented, not {type(args_cluster)} or {args_cluster} !!!')
        

    
    def forward(self, 
                x: Tensor, 
                rpe: Tensor = None):

        '''
        `input`
        x       ->      B, C, H, W

        `returns`
        x'      ->      B, C', H, W  

        NOTE: rpe use is not supported !!!
        '''
        
        # B, C, H, W        ->      [B, H*W = N, C]
        B, C, H, W = x.shape
        device = x.device
        x = x.reshape(B, C, -1)           # [B, C, H*W = N]
        x = x.transpose(1, 2)             # [B, H*W = N, C]
        
        # Cluster
        cluster_output = self.cluster(x=x)
        labels, x_center = cluster_output.labels, cluster_output.centers        # [B, N] and [B, clusters, C]

        # Organize
        x, batch, mapping = organize_and_map(x_=x,
                                             batch_=labels)     # [B, N, C] and [B, N] and [B, N]
        
        # Preparation for Update
        x = x.reshape(-1, C)                                                                                                                # [N' = B*N, C]                     (ready for GraphConv)
        batch = batch + (torch.arange(start=0, end=B, device=device, dtype=torch.int64)*self.center_.num_centroids).unsqueeze(-1)
        batch = batch.reshape(-1)                                                                                                           # [N' = B*N, ]                      (ready for GraphConv)
        x_center = x_center.reshape(-1, C)                                                                                                  # [subgraphs = B*clusters, C]       (ready for GraphConv)
        batch_center = torch.arange(start=0, end=B, device=device, dtype=torch.int64).repeat_interleave(self.center_.num_centroids)         # [subgraphs = B*clusters, ]        (ready for GraphConv)
        edge_index = get_processed_edge_index(x=x,
                                              batch=batch,
                                              normalize=self.normalize_for_edges,
                                              k=self.neighbors,
                                              d=self.dilation,
                                              stoch=self.stochastic,
                                              eps=self.epsilon,
                                              drop=self.drop_rate_neighbors,
                                              method=self.method_for_edges,
                                              training=self.training,
                                              flow=self.layer.flow)                                                                         # [2, E], processed                 (ready for GraphConv)
        
        # GraphConv Update
        x = super().forward(x=x, 
                            batch=batch,
                            edge_index=edge_index,
                            x_center=x_center,
                            batch_center=batch_center)     # [N' = B*N, C'], updated vertices                                                                       
        
        # Re-order
        N_, C_ = x.shape
        x = x.reshape(B, -1, C_)                                    # [B, N = H*W, C']
        mapping = mapping.unsqueeze(-1).expand(-1, -1, C_)          # [B, N = H*W, C']
        x = torch.gather(input=x,
                         dim=1,
                         index=mapping)                             # [B, N = H*W, C']
        x = x.transpose(1, 2)                                       # [B, C', N = H*W]
        x = x.reshape(B, C_, H, W)                                  # [B, C', H, W]
        return x
    


class ConditionalPositionEncoding(nn.Module):
    '''
    Source: GreedyViG
    NOTE: kernel_size MUST be an odd number !!
    '''
    """
    Implementation of conditional positional encoding. For more details refer to paper: 
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_
    """
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.pe = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=in_channels
        )

    def forward(self, x):
        x = self.pe(x) + x
        return x





class Grapher(torch.nn.Module):

    '''
    Grapher module for ClusterViG
    '''

    def __init__(self,
                 
                 in_channels: int,
                 out_channels: int,
                 factor: int,

                 dropout: float,
                 act: str,
                 norm: str,
                 
                 drop_path: float,

                 clusters: int,
                 neighbors: int,
                 dilation: int,
                 stochastic: bool,
                 epsilon: float,
                 drop_rate_neighbors: float,
                 method_for_edges: str,

                 init_method: str,
                 num_init: int,
                 max_iter: int,
                 tol: float,

                 vertex_conv: str,
                 center_conv: str,
                 
                 use_conditional_pos: bool,

                 use_relative_pos: Optional[bool] = False):
        
        super().__init__()

        if not use_relative_pos in [False, None]:
            raise NotImplementedError(f'use_relative_pos must be False or None but {use_relative_pos} was passed !! relative positional embeddings for kNN enhancement not currently supported !!')

        factor = factor if not vertex_conv == 'gin' else 1

        if center_conv == 'edge':
            _center_args = ArgsUpCentroidsEdgeConv(num_centroids=clusters,
                                                   in_channels=in_channels,
                                                   out_channels=in_channels*factor,
                                                   groups=1,
                                                   dropout=dropout,
                                                   act=act,
                                                   norm=norm,
                                                   aggr='max')
        elif center_conv == 'gat':
            _center_args = ArgsUpCentroidsGATConv(num_centroids=clusters,
                                                  in_channels=in_channels,
                                                  out_channels_total=in_channels*factor,
                                                  heads=factor,
                                                  concat=True,
                                                  dropout=dropout,
                                                  negative_slope=0.2,
                                                  aggr='add',
                                                  version='v2')
        else:
            raise Exception(f'center_conv must be in [edge, gat] but {center_conv} was passed !!!')
        
        
        _vertex_args = ArgsStaticGraphConv(in_channels=in_channels,
                                           out_channels=out_channels,
                                           args_centroid=_center_args,
                                           conv_centroid=center_conv,
                                           groups=1,
                                           dropout=dropout,
                                           act=act,
                                           norm=norm,
                                           aggr='max' if vertex_conv in ['edge', 'mr', 'sage'] else 'add')      # GIN uses 'add'
        
        _cluster_args = ArgsClusterKMeans(init_method=init_method,
                                          num_init=num_init,
                                          max_iter=max_iter,
                                          distance=LpDistance,
                                          p_norm=2,
                                          tol=tol,
                                          normalize=True,
                                          verbose=False,
                                          seed=None,)
        



        self.cpe = ConditionalPositionEncoding(in_channels=in_channels, kernel_size=7) if use_conditional_pos else nn.Identity()
        self.fc1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                           out_channels=in_channels,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0),
                                 nn.BatchNorm2d(in_channels))
        self.graphconv = DynamicGraphConv(args_gconv=_vertex_args,
                                          conv_gconv=vertex_conv,
                                          neighbors=neighbors,
                                          dilation=dilation,
                                          normalize_for_edges=True,
                                          stochastic=stochastic,
                                          epsilon=epsilon,
                                          drop_rate_neighbors=drop_rate_neighbors,
                                          method_for_edges=method_for_edges,
                                          args_cluster=_cluster_args,
                                          reduction=None,
                                          )
        self.fc2 = nn.Sequential(nn.Conv2d(in_channels=out_channels,
                                           out_channels=in_channels,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,),
                                 nn.BatchNorm2d(in_channels))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        

        
    def forward(self, x: Tensor):
        '''
        B, C, H, W -> B, C, H, W
        '''
        _tmp = x
        x = self.cpe(x)                         # B, C, H, W
        x = self.fc1(x)                         # B, C, H, W
        x = self.graphconv(x)                   # B, C', H, W
        x = self.fc2(x)                         # B, C, H, W
        x = self.drop_path(x) + _tmp            # B, C, H, W

        return x