from typing import Callable, Optional, Union

import numpy as np

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor

from .torch_nn import StandardGConv
from .torch_centroid import UpdateCentroids, ArgsUpCentroidsEdgeConv, ArgsUpCentroidsGATConv

# Reference comments: Initial
'''

---------------------------
Level 1: static gconv
---------------------------

`inputs`
----------------
x               ->      [N, D]      organized node feature tensor
batch_sub_x     ->      [N, ]       organized sub-graph level batch tensor
batch_whole_x   ->      [N, ]       organized graph level batch tensor              (may not be needed)
edge_index      ->      [2, E]      sub-graph localized, from source to target      (j to i)      (j's in row 0 to common target i in row 1)

c               ->      [N, D]      centroid tensor, naturally indexed at the sub-graph level
batch_c         ->      [N, ]       batch tensor for centroid organizing each centroid to the graph it belongs to

`returns`
----------------
x'              ->      [N, D]      updated node tensor
batch_x         ->      [N, ]       unchanged, organized sub-graph level batch tensor     


--------------------------
Level 2: dynamic gconv
--------------------------

`inputs`
----------------
x               ->      [B, C, H, W]    image tensor
relative_pos    ->      [1, N, N']      unused relative positional embeddings in current version

`returns`
----------------
x'              ->      [B, C', H, W]   updated image tensor (one-to-one correspondence to pixels)

NOTE dynamic gconv,
- clusters x
- organizes the clusters into sub-graphs
- creates edge_index edge tensor within sub-graphs through knn
- passes above to static gconv
- static gconv returns updated node features
- dynamic gconv re-organizes and re-orders nodes to pixels in image (maintaining correspondence)


'''

'''
NOTE: Warning that using groups > 1 (like 2, 4, etc.) with centroid feature concatenation REDUCES feature diversity and may be 
DETRIMENTAL to performance !!!

A possible solution, 
x_i <------ f(x_i, x_c')           (mix and destroy x_c')

+ Potentially better feature diversity (?)
+ Reduction in parameters
- Implementational complexity
'''




class EdgeConvStaticGConv(MessagePassing):
    r"""The edge convolutional operator from the `"Dynamic Graph CNN for
    Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
        \mathbf{x}_j - \mathbf{x}_i),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.*, defined by :class:`torch.nn.Sequential`.
        aggr (str, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V}|, F_{in}), (|\mathcal{V}|, F_{in}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self,
                
                 in_channels: int,
                 out_channels: int,

                 args_centroid: Union[ArgsUpCentroidsEdgeConv, ArgsUpCentroidsGATConv],
                 conv_centroid: str,
                 
                 groups: int = 4,
                 dropout: float = 0.0,
                 act: str = 'relu',
                 norm: str = 'batch_norm',

                 aggr: str = 'max', 
                 **kwargs):
        
        super().__init__(aggr=aggr, **kwargs)
        
        if not conv_centroid.lower() in ['edge', 'gat']:
            raise Exception(f'conv_centroid must be in [edge, gat] but {conv_centroid.lower()} was passed!')
        
        conv_centroid = conv_centroid.lower()
        if (isinstance(args_centroid, ArgsUpCentroidsEdgeConv) and conv_centroid != 'edge') or \
           (isinstance(args_centroid, ArgsUpCentroidsGATConv) and conv_centroid != 'gat') or \
           ((not isinstance(args_centroid, ArgsUpCentroidsEdgeConv)) and (not isinstance(args_centroid, ArgsUpCentroidsGATConv))):
            raise Exception(f'args_centroid must be of type Union[ArgsUpCentroidsEdgeConv, ArgsUpCentroidsGATConv] and must match with conv_centroid passed! However, type(args_centroid): {type(args_centroid)} and conv_centroid: {conv_centroid} arguments were passed!')
        
        if not (args_centroid.in_channels == in_channels):
            raise Exception(f'in_channels for args_centroid to update feature centroids must match the in_channels to the StaticGConv layer! However args_centroid.in_channels: {args_centroid.in_channels} != in_channels: {in_channels}')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.dropout = dropout
        self.act = act
        self.norm = norm
        
        self.aggr = aggr
        self.kwargs = kwargs
        
        self.args_centroid = args_centroid
        self.conv_centroid = conv_centroid
        
        out_channels_centroid = args_centroid.out_channels_total if conv_centroid == 'gat' else args_centroid.out_channels
        self.out_channels_centroid = out_channels_centroid

        self.nn_vertex = StandardGConv(channel_list=[2*in_channels + out_channels_centroid, out_channels],
                                       groups=groups,
                                       bias=True,
                                       lin_kwargs=None,
                                       dropout=dropout,
                                       act=act,
                                       act_kwargs=None,
                                       norm=norm,
                                       norm_kwargs=None)
        
        self.nn_centroids = UpdateCentroids(args=args_centroid,
                                            conv=conv_centroid)
        
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn_vertex)
        reset(self.nn_centroids)


    def forward(self, 
                x: Union[Tensor, PairTensor],
                batch: Tensor, 
                edge_index: Adj,
                x_center: Tensor,
                batch_center: Tensor) -> Tensor:
        '''
        x               ->      [N, D]          organized sub-graph level node features
        batch           ->      [N,  ]          organized sub-graph level batch tensor
        edge_index      ->      [2, E]          within sub-graph "processed-final" edge tensor (after dilation, edge dropping, etc. as applicable)
        
        x_center        ->      [centers, D]    nominal centroids, natural sub-graph level indexed
        batch_center    ->      [centers,  ]    assigns each centroid in x_center to the graph it belongs to (graph-level indexing)
        '''
        if isinstance(x, Tensor):
            x = (x, x)

        x_center = self.nn_centroids(x_center, batch_center)                            # updated centroids
        y = torch.gather(input=x_center,
                         dim=0,
                         index=batch.unsqueeze(-1).repeat(1, x_center.shape[-1]))       # expanded for in-built _lift
        
        return self.propagate(edge_index, x=x, y=y)                                     # updated vertices

    def message(self, x_i: Tensor, x_j: Tensor, y_j: Tensor) -> Tensor:
        '''
        y_j and y_i are identical as we restrict j -> i (or i -> j) edges (equivalently, here) within sub-graph
        y features are shared for all nodes within a sub-graph
        Thus, y_i or y_j would both give same results 
        '''
        return self.nn_vertex(torch.cat([x_i, x_j - x_i, y_j], dim=-1))                 

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn_vertex={self.nn_vertex})(nn_centroids={self.nn_centroids})'





class MRConvStaticGConv(MessagePassing):
    
    def __init__(self,
                
                 in_channels: int,
                 out_channels: int,

                 args_centroid: Union[ArgsUpCentroidsEdgeConv, ArgsUpCentroidsGATConv],
                 conv_centroid: str,
                 
                 groups: int = 4,
                 dropout: float = 0.0,
                 act: str = 'relu',
                 norm: str = 'batch_norm',

                 aggr: str = 'max', 
                 **kwargs):
        
        super().__init__(aggr=aggr, **kwargs)

        if not conv_centroid.lower() in ['edge', 'gat']:
            raise Exception(f'conv_centroid must be in [edge, gat] but {conv_centroid.lower()} was passed!')
        
        conv_centroid = conv_centroid.lower()
        if (isinstance(args_centroid, ArgsUpCentroidsEdgeConv) and conv_centroid != 'edge') or \
           (isinstance(args_centroid, ArgsUpCentroidsGATConv) and conv_centroid != 'gat') or \
           ((not isinstance(args_centroid, ArgsUpCentroidsEdgeConv)) and (not isinstance(args_centroid, ArgsUpCentroidsGATConv))):
            raise Exception(f'args_centroid must be of type Union[ArgsUpCentroidsEdgeConv, ArgsUpCentroidsGATConv] and must match with conv_centroid passed! However, type(args_centroid): {type(args_centroid)} and conv_centroid: {conv_centroid} arguments were passed!')
        
        if not (args_centroid.in_channels == in_channels):
            raise Exception(f'in_channels for args_centroid to update feature centroids must match the in_channels to the StaticGConv layer! However args_centroid.in_channels: {args_centroid.in_channels} != in_channels: {in_channels}')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.dropout = dropout
        self.act = act
        self.norm = norm
        
        self.aggr = aggr
        self.kwargs = kwargs
        
        self.args_centroid = args_centroid
        self.conv_centroid = conv_centroid

        out_channels_centroid = args_centroid.out_channels_total if conv_centroid == 'gat' else args_centroid.out_channels
        self.out_channels_centroid = out_channels_centroid

        self.nn_vertex = StandardGConv( channel_list=[2*in_channels + out_channels_centroid, out_channels],
                                        groups=groups,
                                        bias=True,
                                        lin_kwargs=None,
                                        dropout=dropout,
                                        act=act,
                                        act_kwargs=None,
                                        norm=norm,
                                        norm_kwargs=None)
        
        self.nn_centroids = UpdateCentroids(args=args_centroid,
                                            conv=conv_centroid)
        
        self.reset_parameters()

    
    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn_vertex)
        reset(self.nn_centroids)


    def forward(self, 
                x: Union[Tensor, PairTensor],
                batch: Tensor, 
                edge_index: Adj,
                x_center: Tensor,
                batch_center: Tensor) -> Tensor:
        
        '''
        x               ->      [N, D]          organized sub-graph level node features
        batch           ->      [N,  ]          organized sub-graph level batch tensor
        edge_index      ->      [2, E]          within sub-graph "processed-final" edge tensor (after dilation, edge dropping, etc. as applicable)
        
        x_center        ->      [centers, D]    nominal centroids, natural sub-graph level indexed
        batch_center    ->      [centers,  ]    assigns each centroid in x_center to the graph it belongs to (graph-level indexing)

        '''

        x_center = self.nn_centroids(x_center, batch_center)                            # updated centroids
        y = torch.gather(input=x_center,
                         dim=0,
                         index=batch.unsqueeze(-1).repeat(1, x_center.shape[-1]))
        x_ = self.propagate(edge_index=edge_index, x=x)
        x = torch.concat([x.unsqueeze(-1), x_.unsqueeze(-1)], dim=-1)                   # [N, D, 2]
        N, D, _ = x.shape
        x = x.reshape(N, D*2)                                                           # [N, D*2]
        x = torch.concat([x, y], dim=-1)
        return self.nn_vertex(x)                                                        # [N, D]
    
    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return x_j - x_i
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn_vertex={self.nn_vertex})(nn_centroids={self.nn_centroids})'
    









class GraphSAGEStaticGConv(MessagePassing):

    def __init__(self,
                
                 in_channels: int,
                 out_channels: int,

                 args_centroid: Union[ArgsUpCentroidsEdgeConv, ArgsUpCentroidsGATConv],
                 conv_centroid: str,
                 
                 groups: int = 4,
                 dropout: float = 0.0,
                 act: str = 'relu',
                 norm: str = 'batch_norm',

                 aggr: str = 'max', 
                 **kwargs):
        
        super().__init__(aggr=aggr, **kwargs)

        if not conv_centroid.lower() in ['edge', 'gat']:
            raise Exception(f'conv_centroid must be in [edge, gat] but {conv_centroid.lower()} was passed!')
        
        conv_centroid = conv_centroid.lower()
        if (isinstance(args_centroid, ArgsUpCentroidsEdgeConv) and conv_centroid != 'edge') or \
           (isinstance(args_centroid, ArgsUpCentroidsGATConv) and conv_centroid != 'gat') or \
           ((not isinstance(args_centroid, ArgsUpCentroidsEdgeConv)) and (not isinstance(args_centroid, ArgsUpCentroidsGATConv))):
            raise Exception(f'args_centroid must be of type Union[ArgsUpCentroidsEdgeConv, ArgsUpCentroidsGATConv] and must match with conv_centroid passed! However, type(args_centroid): {type(args_centroid)} and conv_centroid: {conv_centroid} arguments were passed!')
        
        if not (args_centroid.in_channels == in_channels):
            raise Exception(f'in_channels for args_centroid to update feature centroids must match the in_channels to the StaticGConv layer! However args_centroid.in_channels: {args_centroid.in_channels} != in_channels: {in_channels}')

        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.dropout = dropout
        self.act = act
        self.norm = norm
        
        self.aggr = aggr
        self.kwargs = kwargs
        
        self.args_centroid = args_centroid
        self.conv_centroid = conv_centroid

        out_channels_centroid = args_centroid.out_channels_total if conv_centroid == 'gat' else args_centroid.out_channels
        self.out_channels_centroid = out_channels_centroid



        self.nn_edge = StandardGConv(channel_list=[in_channels, in_channels],
                                     groups=groups,
                                     bias=True,
                                     lin_kwargs=None,
                                     dropout=dropout,
                                     act=act,
                                     act_kwargs=None,
                                     norm=norm,
                                     norm_kwargs=None)


        self.nn_centroids = UpdateCentroids(args=args_centroid,
                                            conv=conv_centroid)    


        self.nn_vertex = StandardGConv(channel_list=[2*in_channels + out_channels_centroid, out_channels],
                                       groups=groups,
                                       bias=True,
                                       lin_kwargs=None,
                                       dropout=dropout,
                                       act=act,
                                       act_kwargs=None,
                                       norm=norm,
                                       norm_kwargs=None)
        
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn_edge)
        reset(self.nn_centroids)
        reset(self.nn_vertex)

    
    def forward(self, 
                x: Union[Tensor, PairTensor],
                batch: Tensor, 
                edge_index: Adj,
                x_center: Tensor,
                batch_center: Tensor) -> Tensor:
        '''
        x               ->      [N, D]          organized sub-graph level node features
        batch           ->      [N,  ]          organized sub-graph level batch tensor
        edge_index      ->      [2, E]          within sub-graph "processed-final" edge tensor (after dilation, edge dropping, etc. as applicable)
        
        x_center        ->      [centers, D]    nominal centroids, natural sub-graph level indexed
        batch_center    ->      [centers,  ]    assigns each centroid in x_center to the graph it belongs to (graph-level indexing)

        '''        

        x_center = self.nn_centroids(x_center, batch_center)                            # updated centroids
        y = torch.gather(input=x_center,
                         dim=0,
                         index=batch.unsqueeze(-1).repeat(1, x_center.shape[-1]))        
        x_ = self.propagate(edge_index=edge_index, x=x)
        x = torch.concat([x, x_, y], dim=-1)
        return self.nn_vertex(x)
    

    def message(self, x_j: Tensor) -> Tensor:
        return self.nn_edge(x_j)
    

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn_vertex={self.nn_vertex})(nn_centroids={self.nn_centroids})(nn_edge={self.nn_edge})'
    














class GINConvStaticGConv(MessagePassing):
    
    def __init__(self,
                 
                 in_channels: int,
                 out_channels: int,

                 args_centroid: Union[ArgsUpCentroidsEdgeConv, ArgsUpCentroidsGATConv],
                 conv_centroid: str,
                 
                 groups: int = 4,
                 dropout: float = 0.0,
                 act: str = 'relu',
                 norm: str = 'batch_norm',
                 
                 aggr: str = 'add',                 # NOTE the 'add' aggregation here !!
                 **kwargs):
        
        
        super().__init__(aggr=aggr, **kwargs)

        if not conv_centroid.lower() in ['edge', 'gat']:
            raise Exception(f'conv_centroid must be in [edge, gat] but {conv_centroid.lower()} was passed!')
        
        conv_centroid = conv_centroid.lower()
        if (isinstance(args_centroid, ArgsUpCentroidsEdgeConv) and conv_centroid != 'edge') or \
           (isinstance(args_centroid, ArgsUpCentroidsGATConv) and conv_centroid != 'gat') or \
           ((not isinstance(args_centroid, ArgsUpCentroidsEdgeConv)) and (not isinstance(args_centroid, ArgsUpCentroidsGATConv))):
            raise Exception(f'args_centroid must be of type Union[ArgsUpCentroidsEdgeConv, ArgsUpCentroidsGATConv] and must match with conv_centroid passed! However, type(args_centroid): {type(args_centroid)} and conv_centroid: {conv_centroid} arguments were passed!')
        
        if not (args_centroid.in_channels == in_channels):
            raise Exception(f'in_channels for args_centroid to update feature centroids must match the in_channels to the StaticGConv layer! However args_centroid.in_channels: {args_centroid.in_channels} != in_channels: {in_channels}')

        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.dropout = dropout
        self.act = act
        self.norm = norm
        
        self.aggr = aggr
        self.kwargs = kwargs
        
        self.args_centroid = args_centroid
        self.conv_centroid = conv_centroid

        out_channels_centroid = args_centroid.out_channels_total if conv_centroid == 'gat' else args_centroid.out_channels
        self.out_channels_centroid = out_channels_centroid

        if not out_channels_centroid == in_channels:
            raise Exception(f'For GINConvStaticGConv, out_channels for centroid must match the in_channels of features! However, out_channels for centroids are: {out_channels_centroid} and in_channels of input features are: {in_channels}')
        
        eps_init = 0.0
        self.eps_init = eps_init

        delta_init = 0.0
        self.delta_init = delta_init
        
        self.nn_vertex = StandardGConv(channel_list=[in_channels, out_channels],
                                       groups=groups,
                                       bias=True,
                                       lin_kwargs=None,
                                       dropout=dropout,
                                       act=act,
                                       act_kwargs=None,
                                       norm=norm,
                                       norm_kwargs=None) 
        

        self.nn_centroids = UpdateCentroids(args=args_centroid,
                                            conv=conv_centroid)
                
        self.eps = torch.nn.Parameter(torch.Tensor([eps_init]))
        
        self.delta = torch.nn.Parameter(torch.Tensor([delta_init]))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn_vertex)
        reset(self.nn_centroids)
        reset(self.eps)                 # Does nothing, done only for completion
        reset(self.delta)               # Does nothing, done only for completion


    def forward(self, 
                x: Union[Tensor, PairTensor],
                batch: Tensor, 
                edge_index: Adj,
                x_center: Tensor,
                batch_center: Tensor) -> Tensor:
        '''
        x               ->      [N, D]          organized sub-graph level node features
        batch           ->      [N,  ]          organized sub-graph level batch tensor
        edge_index      ->      [2, E]          within sub-graph "processed-final" edge tensor (after dilation, edge dropping, etc. as applicable)
        
        x_center        ->      [centers, D]    nominal centroids, natural sub-graph level indexed
        batch_center    ->      [centers,  ]    assigns each centroid in x_center to the graph it belongs to (graph-level indexing)

        '''        

        x_center = self.nn_centroids(x_center, batch_center)                            # updated centroids
        y = torch.gather(input=x_center,
                         dim=0,
                         index=batch.unsqueeze(-1).repeat(1, x_center.shape[-1]))
        
        x_ = self.propagate(edge_index=edge_index, x=x)
        x = ((1 + self.eps)*x) + ((1 + self.delta)*y) + x_

        return self.nn_vertex(x)

    
    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn_vertex={self.nn_vertex})(nn_centroids={self.nn_centroids})(eps={self.eps})(delta={self.delta})'













class ArgsStaticGraphConv:
    def __init__(self,
                 
                 in_channels: int,
                 out_channels: int,

                 args_centroid: Union[ArgsUpCentroidsEdgeConv, ArgsUpCentroidsGATConv],
                 conv_centroid: str,
                 
                 groups: int,
                 dropout: float,
                 act: str,
                 norm: str,
                
                 aggr: str,                 
                 **kwargs
                 ):
        conv_centroid = conv_centroid.lower()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.args_centroid = args_centroid
        self.conv_centroid = conv_centroid

        self.groups = groups
        self.dropout = dropout
        self.act = act
        self.norm = norm

        self.aggr = aggr
        self.kwargs = kwargs

    def parse_args(self):
        args_dict = {'in_channels': self.in_channels,
                     'out_channels': self.out_channels,
                     
                     'args_centroid': self.args_centroid,
                     'conv_centroid': self.conv_centroid,
                     
                     'groups': self.groups,
                     'dropout': self.dropout,
                     'act': self.act,
                     'norm': self.norm,
                     
                     'aggr': self.aggr,
                     **self.kwargs}
        return args_dict











class StaticGraphConv(torch.nn.Module):
    
    def __init__(self, 
                 args_staticGraphConv: ArgsStaticGraphConv,
                 conv_staticGraphConv: str):
        
        super().__init__()

        if not conv_staticGraphConv.lower() in ['edge', 'mr', 'sage', 'gin']:
            raise Exception(f'conv_staticGraphConv must be in [edge, mr, sage, gin] but conv_staticGraphConv: {conv_staticGraphConv.lower()} was passed !!')

        conv_staticGraphConv = conv_staticGraphConv.lower()

        self.args_staticGraphConv = args_staticGraphConv
        self.conv_staticGraphConv = conv_staticGraphConv

        if conv_staticGraphConv == 'edge':
            self.layer = EdgeConvStaticGConv(**self.args_staticGraphConv.parse_args())
        elif conv_staticGraphConv == 'mr':
            self.layer = MRConvStaticGConv(**self.args_staticGraphConv.parse_args())
        elif conv_staticGraphConv == 'sage':
            self.layer = GraphSAGEStaticGConv(**self.args_staticGraphConv.parse_args())
        elif conv_staticGraphConv == 'gin':
            self.layer = GINConvStaticGConv(**self.args_staticGraphConv.parse_args())
        else:
            raise Exception(f'conv_staticGraphConv must be in [edge, mr, sage, gin] but {conv_staticGraphConv} was passed !!')
        
        # self.reset_parameters()           # Not done to reduce redundancy

    
    def forward(self, 
                x: Union[Tensor, PairTensor],
                batch: Tensor, 
                edge_index: Adj,
                x_center: Tensor,
                batch_center: Tensor) -> Tensor:
        return self.layer(x=x, batch=batch, edge_index=edge_index, x_center=x_center, batch_center=batch_center)
    

    def reset_parameters(self):
        '''For top-level'''
        self.layer.reset_parameters()