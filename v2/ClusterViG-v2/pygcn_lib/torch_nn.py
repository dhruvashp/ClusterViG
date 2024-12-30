import inspect
import warnings
from typing import Any, Callable, Dict, Final, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Identity

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import NoneType



'''

PyG stand-in for gcn_lib -> torch_nn.py file

'''

class LinearGrouped(torch.nn.Module):

    '''

    A grouped implementation for Linear function in PyG format/notion

    Likely not optimal and likely can be better implemented via batched grouped Conv2D/Conv1D, implemented below as Linear extension for simplicity

    '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 groups: int,
                 bias: bool = True,
                 weight_initializer: Optional[str] = None,
                 bias_initializer: Optional[str] = None,):
        
        super().__init__()

        if not (((in_channels % groups) == 0) and ((out_channels % groups) == 0)):
            raise Exception('in_channels and out_channels must be divisible (multiple) of groups!')

        in_channels_per_head = in_channels // groups
        out_channels_per_head = out_channels // groups

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.bias = bias
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        self.in_channels_per_head = in_channels_per_head
        self.out_channels_per_head = out_channels_per_head

        self.head_list = torch.nn.ModuleList()
        for i in range(groups):
            self.head_list.append(Linear(in_channels=in_channels_per_head,
                                         out_channels=out_channels_per_head,
                                         bias=bias,
                                         weight_initializer=weight_initializer,
                                         bias_initializer=bias_initializer))
            
        # self.reset_parameters()       # Not done, will reset parameters again, redundantly
            
    def forward(self, x: Tensor):
        '''
        x       ->  [N, D]
        returns ->  [N, D']
        '''
        heads = x.split(split_size=self.in_channels_per_head, dim=-1)       # tuple([N, in_channels // groups]), len(tuple) = groups
        outs = [layer(heads[i]) for i, layer in enumerate(self.head_list)]  # tuple([N, out_channels // groups]), len(tuple) = groups
        return torch.concat(outs, dim=-1)     


    def reset_parameters(self):
        '''
        Only added for top-level support
        '''
        for layer in self.head_list:
            layer.reset_parameters()





class MLPGrouped(torch.nn.Module):
    '''
    
    Copied from torch_geometric.nn.models -> mlp.py MLP class
    
    Modified to include LinearGrouped instead of Linear to allow for grouped Linear operation in original gcn_lib/ViG style

    '''
    r"""A Multi-Layer Perception (MLP) model.

    There exists two ways to instantiate an :class:`MLP`:

    1. By specifying explicit channel sizes, *e.g.*,

       .. code-block:: python

          mlp = MLP([16, 32, 64, 128])

       creates a three-layer MLP with **differently** sized hidden layers.

    1. By specifying fixed hidden channel sizes over a number of layers,
       *e.g.*,

       .. code-block:: python

          mlp = MLP(in_channels=16, hidden_channels=32,
                    out_channels=128, num_layers=3)

       creates a three-layer MLP with **equally** sized hidden layers.

    Args:
        channel_list (List[int] or int, optional): List of input, intermediate
            and output channels such that :obj:`len(channel_list) - 1` denotes
            the number of layers of the MLP (default: :obj:`None`)
        in_channels (int, optional): Size of each input sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        hidden_channels (int, optional): Size of each hidden sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        out_channels (int, optional): Size of each output sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        num_layers (int, optional): The number of layers.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        dropout (float or List[float], optional): Dropout probability of each
            hidden embedding. If a list is provided, sets the dropout value per
            layer. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`"batch_norm"`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        plain_last (bool, optional): If set to :obj:`False`, will apply
            non-linearity, batch normalization and dropout to the last layer as
            well. (default: :obj:`True`)
        bias (bool or List[bool], optional): If set to :obj:`False`, the module
            will not learn additive biases. If a list is provided, sets the
            bias per layer. (default: :obj:`True`)
        **kwargs (optional): Additional deprecated arguments of the MLP layer.
    """
    supports_norm_batch: Final[bool]

    def __init__(
        self,
        channel_list: Optional[Union[List[int], int]] = None,
        *,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: Optional[int] = None,
        groups: Union[int, List[int]] = 1,                  # default single group
        lin_kwargs: Optional[Dict[str, Any]] = None,        # to add initializers
        dropout: Union[float, List[float]] = 0.,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        plain_last: bool = True,
        bias: Union[bool, List[bool]] = True,
        **kwargs,
    ):
        super().__init__()

        # Backward compatibility:
        act_first = act_first or kwargs.get("relu_first", False)
        batch_norm = kwargs.get("batch_norm", None)
        if batch_norm is not None and isinstance(batch_norm, bool):
            warnings.warn("Argument `batch_norm` is deprecated, "
                          "please use `norm` to specify normalization layer.")
            norm = 'batch_norm' if batch_norm else None
            batch_norm_kwargs = kwargs.get("batch_norm_kwargs", None)
            norm_kwargs = batch_norm_kwargs or {}

        if isinstance(channel_list, int):
            in_channels = channel_list

        if in_channels is not None:
            if num_layers is None:
                raise ValueError("Argument `num_layers` must be given")
            if num_layers > 1 and hidden_channels is None:
                raise ValueError(f"Argument `hidden_channels` must be given "
                                 f"for `num_layers={num_layers}`")
            if out_channels is None:
                raise ValueError("Argument `out_channels` must be given")

            channel_list = [hidden_channels] * (num_layers - 1)
            channel_list = [in_channels] + channel_list + [out_channels]

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.act_first = act_first
        self.plain_last = plain_last

        if isinstance(groups, int):
            groups = [groups] * (len(channel_list) - 1)
        else:
            if len(groups) != len(channel_list) - 1:
                raise ValueError(f"Number of groups values provided ({len(groups)}) does not "
                                 f"match the number of layers specified "
                                 f"({len(channel_list) - 1})")
        self.groups = groups

        if isinstance(dropout, float):
            dropout = [dropout] * (len(channel_list) - 1)
            if plain_last:
                dropout[-1] = 0.
        if len(dropout) != len(channel_list) - 1:
            raise ValueError(
                f"Number of dropout values provided ({len(dropout)} does not " # type: ignore
                f"match the number of layers specified "
                f"({len(channel_list)-1})")
        self.dropout = dropout                  # NOTE that with plain_last = True and dropout list passed explicitly, user must explicitly specify dropout[-1] = 0 to ensure no dropout after last (assumed plain) layer, user provided dropout list will not be overwritten at dropout[-1] to NULL with plain_last = True

        if isinstance(bias, bool):
            bias = [bias] * (len(channel_list) - 1)
        if len(bias) != len(channel_list) - 1:
            raise ValueError(
                f"Number of bias values provided ({len(bias)}) does not match "
                f"the number of layers specified ({len(channel_list)-1})")

        self.lins = torch.nn.ModuleList()
        iterator = zip(channel_list[:-1], channel_list[1:], bias, groups)
        for in_channels, out_channels, _bias, _groups in iterator:
            self.lins.append(LinearGrouped(in_channels=in_channels, 
                                           out_channels=out_channels, 
                                           bias=_bias,
                                           groups=_groups,
                                           **(lin_kwargs or {}))
                            )

        self.norms = torch.nn.ModuleList()
        iterator = channel_list[1:-1] if plain_last else channel_list[1:]
        for hidden_channels in iterator:
            if norm is not None:
                norm_layer = normalization_resolver(
                    norm,
                    hidden_channels,
                    **(norm_kwargs or {}),
                )
            else:
                norm_layer = Identity()
            self.norms.append(norm_layer)

        self.supports_norm_batch = False
        if len(self.norms) > 0 and hasattr(self.norms[0], 'forward'):
            norm_params = inspect.signature(self.norms[0].forward).parameters
            self.supports_norm_batch = 'batch' in norm_params

        self.reset_parameters()         # kind of redundant but not removed to maintain source

    @property
    def in_channels(self) -> int:
        r"""Size of each input sample."""
        return self.channel_list[0]

    @property
    def out_channels(self) -> int:
        r"""Size of each output sample."""
        return self.channel_list[-1]

    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.channel_list) - 1

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def forward(
        self,
        x: Tensor,
        batch: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        return_emb: NoneType = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The source tensor.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            return_emb (bool, optional): If set to :obj:`True`, will
                additionally return the embeddings before execution of the
                final output layer. (default: :obj:`False`)
        """
        # `return_emb` is annotated here as `NoneType` to be compatible with
        # TorchScript, which does not support different return types based on
        # the value of an input argument.
        emb: Optional[Tensor] = None

        # If `plain_last=True`, then `len(norms) = len(lins) -1, thus skipping
        # the execution of the last layer inside the for-loop.
        for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
            x = lin(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.supports_norm_batch:
                x = norm(x, batch, batch_size)
            else:
                x = norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout[i], training=self.training)
            if isinstance(return_emb, bool) and return_emb is True:
                emb = x

        if self.plain_last:
            x = self.lins[-1](x)
            x = F.dropout(x, p=self.dropout[-1], training=self.training)

        return (x, emb) if isinstance(return_emb, bool) else x

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.channel_list)[1:-1]})'
    






class StandardGConv(torch.nn.Module):

    def __init__(self,
                 
                 channel_list: List[int],
                 groups: Union[int, List[int]] = 4,
                 bias: Union[bool, List[bool]] = True,
                 
                 lin_kwargs: Optional[Dict[str, Any]] = None,
                 
                 dropout: Union[float, List[float]] = 0.0,
                 
                 act: Union[str, Callable, None] = 'relu',
                 act_kwargs: Optional[Dict[str, Any]] = None,
                 
                 norm: Union[str, Callable, None] = 'batch_norm',
                 norm_kwargs: Optional[Dict[str, Any]] = None,
                 ):
        
        '''
        
        NOTE that StandardGConv is not identical to BasicConv (and is not intended to be)

        Minor differences exist; in contrast to deliberate initializations in BasicConv (most of which anyways follow defaults),
        we allow initializations in StandardGConv to happen exactly per defaults

        StandardGConv   ->      PyG, 2D graph inputs, 1D batch tensors
        BasicConv       ->      PyTorch, 4D image inputs, no explicit batch (index) tensors (inputs have a batch dimension) 
        
        ------------------------------------------------------------------------------------------------------------
        
        channel_list    ->      [c_0, c_1, c_2, ...., c_N]
        groups          ->           [g_1, g_2, ...., g_N]
        bias            ->           [b_1, b_2, ...., b_N]
        dropout         ->           [d_1, d_2, ...., d_N]

        creates an N layer MLP which operates as,
        - LinearGrouped    
        - Norm
        - Activation
        - Dropout
        for each of the N layers (with above list of parameters for the N layers)
    
        '''

        super().__init__()

        self.layer = MLPGrouped(channel_list=channel_list,
                                in_channels=None,
                                hidden_channels=None,
                                out_channels=None,
                                num_layers=None,
                                groups=groups,
                                lin_kwargs=lin_kwargs,
                                dropout=dropout,
                                act=act,
                                act_first=False,
                                act_kwargs=act_kwargs,
                                norm=norm,
                                norm_kwargs=norm_kwargs,
                                plain_last=False,
                                bias=bias)
        
        # self.reset_parameters()           # Not done, will be redundant
        
    
    def forward(self, x: Tensor):
        return self.layer(x)
    

    def reset_parameters(self):
        '''
        For top-level
        '''
        self.layer.reset_parameters()