import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

from pygcn_lib import Grapher

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


'''
Also from GreedyViG, unsure of usage/utility
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'


'''
From GreedyViG (style)
NOTE: Unsure of usage/utility, need to cross-check/verify
'''
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


'''
From GreedyViG (style)
NOTE: Unsure of usage/utility, need to cross-check/verify
'''
default_cfgs = {
    'clustervig': _cfg(crop_pct=0.9, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
}


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    '''
    From ViG
    '''
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer




class FFN(nn.Module):
    '''
    From ViG
    '''
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x
    



class Stem(nn.Module):
    '''
    Native support for ImageNet image size (224 x 224)
    Allows for various reductions in powers of 2 for input to output resolution
    '''

    def __init__(self, in_img_size=224, out_img_size=14, in_dim=3, out_dim=768, act='relu', net_layers=5):
        super().__init__()
        ratio = math.log2(in_img_size/out_img_size)
        if not (ratio.is_integer() and ratio > 0.0):
            raise Exception('out_img_size must divide in_img_size with powers of 2 !!')
        ratio = int(ratio)
        self.convs = nn.ModuleList()
        __current_dim = in_dim
        net_layers = max(ratio + 1, net_layers) 
        dim_growth_layers = net_layers - 1
        # Resolution halving layers
        for i in range(ratio):
            __out_dim = max((out_dim // 2**(dim_growth_layers - 1 - i)), 64)
            self.convs.append(nn.Conv2d(in_channels=__current_dim, 
                                        out_channels=__out_dim,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1))                     # L x L image -> L/2 x L/2 (if L even) or (L + 1)/2 x (L + 1)/2 (if L odd) [ceil(L/2) in general]
            self.convs.append(nn.BatchNorm2d(__out_dim))
            self.convs.append(act_layer(act))
            __current_dim = __out_dim
        # Resolution retaining layers
        for i in range(ratio, dim_growth_layers):
            __out_dim = max((out_dim // 2**(dim_growth_layers - 1 - i)), 64)
            self.convs.append(nn.Conv2d(in_channels=__current_dim,
                                        out_channels=__out_dim,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1))
            self.convs.append(nn.BatchNorm2d(__out_dim))
            self.convs.append(act_layer(act))
            __current_dim = __out_dim
        # Final layer
        self.convs.append(nn.Conv2d(in_channels=out_dim,
                                    out_channels=out_dim,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1))
        self.convs.append(nn.BatchNorm2d(out_dim))

        self.convs = Seq(*self.convs)

    def forward(self, x: Tensor) -> Tensor:
        return self.convs(x)






class IsoClusterViG(nn.Module):
    def __init__(self,
                 n_classes: int,
                 in_image_size: int,
                 out_image_size: int,
                 in_dim: int,
                 iso_dim: int,
                 
                 act: str,
                 dropout_head: float,
                 norm: str,
                 
                 stem_layers: int,
                 n_blocks: int,
                 
                 neighbors: int,
                 clusters: int,

                 vertex_conv: str,
                 center_conv: str,
                 
                 edge_option: str,
                 drop_edge: float,
                 stoch_edge: bool,
                 eps_edge: float,

                 drop_path: float,
                 
                 distillation: bool,
                 
                 forced_init: bool):
        super().__init__()

        self.stem = Stem(in_img_size=in_image_size,
                         out_img_size=out_image_size,
                         in_dim=in_dim,
                         out_dim=iso_dim,
                         act=act,
                         net_layers=stem_layers,)
        
        self.n_classes = n_classes
        self.n = out_image_size*out_image_size                  # nodes, fixed/isotropic much like channels
        self.dim = iso_dim                                      # isotropic dimension            

        self.pos_embed = nn.Parameter(torch.zeros(1, iso_dim, out_image_size, out_image_size), requires_grad=True)

        neighbors_list = [int(x.item()) for x in torch.linspace(neighbors, 2*neighbors, n_blocks)]
        clusters_list = [clusters for i in range(n_blocks)]
        drop_path_list = [x.item() for x in torch.linspace(0, drop_path, n_blocks)]

        self.neighbors_list = neighbors_list
        self.clusters_list = clusters_list
        self.drop_path_list = drop_path_list
        
        self.n_blocks = n_blocks
        
        self.drop_edge = drop_edge if edge_option == 'dropout' else None
        self.stoch_edge = stoch_edge if edge_option == 'dilated' else None
        self.eps_edge = eps_edge if edge_option == 'dilated' else None

        if edge_option == 'dilated':
            min_nodes_per_subgraph = self.n // max(clusters_list)               # assuming homogeneous cluster size
            max_dilation = min_nodes_per_subgraph // max(neighbors_list)        # tightest possible bound that is always followed (assuming homogeneous cluster size)
            dilation_list = [min((i // 4) + 1, max_dilation) for i in range(n_blocks)]
        
        self.dilation_list = dilation_list if edge_option == 'dilated' else [None for i in range(n_blocks)]

        self.dropout_head = dropout_head
        self.act = act
        self.norm = norm

        self.edge_option = edge_option

        self.vertex_conv = vertex_conv
        self.center_conv = center_conv

        self.distillation = distillation
        
        self.forced_init = forced_init

        self.backbone = nn.ModuleList()

        for i in range(self.n_blocks):
            self.backbone.append(Grapher(in_channels            =   self.dim,
                                         out_channels           =   self.dim*2,
                                         factor                 =   1,
                                         dropout                =   0.0,
                                         act                    =   self.act,
                                         norm                   =   self.norm,
                                         drop_path              =   self.drop_path_list[i],
                                         clusters               =   self.clusters_list[i],
                                         neighbors              =   self.neighbors_list[i],
                                         
                                         ############# overridden params ##########################
                                         dilation               =   self.dilation_list[i],
                                         stochastic             =   self.stoch_edge,
                                         epsilon                =   self.eps_edge,
                                         drop_rate_neighbors    =   self.drop_edge,
                                         method_for_edges       =   self.edge_option,
                                         ##########################################################

                                         init_method            =   'rnd',
                                         num_init               =   4,
                                         max_iter               =   50,
                                         tol                    =   5e-4,
                                         vertex_conv            =   self.vertex_conv,
                                         center_conv            =   self.center_conv,
                                         use_conditional_pos    =   True,
                                         use_relative_pos       =   None))
            self.backbone.append(FFN(in_features        =   self.dim,
                                     hidden_features    =   self.dim*4,
                                     out_features       =   self.dim,
                                     act                =   self.act,
                                     drop_path          =   self.drop_path_list[i]))
        
        self.backbone = Seq(*self.backbone)

        self.prediction = Seq(nn.AdaptiveAvgPool2d(output_size=1),
                              nn.Conv2d(in_channels=self.dim,
                                        out_channels=1024,
                                        kernel_size=1,
                                        stride=1,
                                        bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(self.act),
                              nn.Dropout(self.dropout_head),)
        
        self.head = nn.Conv2d(in_channels=1024,
                              out_channels=self.n_classes,
                              kernel_size=1,
                              stride=1,
                              bias=True)
        
        if self.distillation:
            self.dist_head = nn.Conv2d(in_channels=1024,
                                       out_channels=self.n_classes,
                                       kernel_size=1,
                                       stride=1,
                                       bias=True)
            
        if self.forced_init:
            self.model_init()

    

    def model_init(self):
        '''
        Will be mixed - PyG based modules should NOT be affected by this ONLY Conv2d
        This will lead to mixed initializations
        '''
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x) + self.pos_embed
        B, C, H, W = x.shape
        x = self.backbone(x)
        x = self.prediction(x)
        if self.distillation:
            x = self.head(x).squeeze(-1).squeeze(-1), self.dist_head(x).squeeze(-1).squeeze(-1)       # ([B, n_classes], [B, n_classes])
            if not self.training:
                x = (x[0] + x[1])/2
        else:
            x = self.head(x).squeeze(-1).squeeze(-1)
        return x                                        # distillation + training               ->          (logits_1, logits_2)
                                                        # distillation + testing                ->          (logits_aggregated)
                                                        # no distillation (training/testing)    ->          (logits)






@register_model
def IsoClusterViG_Ti_n196_c4(pretrained=False, 
                             **kwargs):
    model = IsoClusterViG(n_classes=1000,
                          in_image_size=224,
                          out_image_size=14,
                          in_dim=3,
                          iso_dim=192,
                          act='gelu',
                          dropout_head=0.0,
                          norm='batch_norm',
                          stem_layers=5,
                          n_blocks=12,
                          neighbors=9,
                          clusters=4,
                          vertex_conv='mr',
                          center_conv='gat',
                          ###################### overridden params ##################
                          edge_option=None,
                          drop_edge=None,
                          stoch_edge=None,
                          eps_edge=None,
                          ###########################################################
                          drop_path=0.0,
                          distillation=True,
                          forced_init=True,)
    model.default_cfg = default_cfgs['clustervig']
    return model





@register_model
def IsoClusterViG_S_n196_c4(pretrained=False,
                            **kwargs):
    model = IsoClusterViG(n_classes=1000,
                          in_image_size=224,
                          out_image_size=14,
                          in_dim=3,
                          iso_dim=320,
                          act='gelu',
                          dropout_head=0.0,
                          norm='batch_norm',
                          stem_layers=5,
                          n_blocks=16,
                          neighbors=9,
                          clusters=4,
                          vertex_conv='mr',
                          center_conv='gat',
                          ################## overridden params ###################
                          edge_option=None,
                          drop_edge=None,
                          stoch_edge=None,
                          eps_edge=None,
                          ########################################################
                          drop_path=0.0,
                          distillation=True,
                          forced_init=True,
                          )
    model.default_cfg = default_cfgs['clustervig']
    return model



@register_model
def IsoClusterViG_B_n196_c4(pretrained=False,
                            **kwargs):
    model = IsoClusterViG(n_classes=1000,
                          in_image_size=224,
                          out_image_size=14,
                          in_dim=3,
                          iso_dim=640,
                          act='gelu',
                          dropout_head=0.0,
                          norm='batch_norm',
                          stem_layers=5,
                          n_blocks=16,
                          neighbors=9,
                          clusters=4,
                          vertex_conv='mr',
                          center_conv='gat',
                          ################## overridden params ###############
                          edge_option=None,
                          drop_edge=None,
                          stoch_edge=None,
                          eps_edge=None,
                          #####################################################
                          drop_path=0.0,
                          distillation=True,
                          forced_init=True
                          )
    model.default_cfg = default_cfgs['clustervig']
    return model










############################### Higher resolution variants (2x resolution, 4x nodes) ######################





@register_model
def IsoClusterViG_Ti_n784_c6(pretrained=False,
                              **kwargs):
    model = IsoClusterViG(n_classes=1000,
                          in_image_size=224,
                          out_image_size=28,
                          in_dim=3,
                          iso_dim=192,
                          act='gelu',
                          dropout_head=0.0,
                          norm='batch_norm',
                          stem_layers=5,
                          n_blocks=12,
                          neighbors=9,
                          clusters=6,
                          vertex_conv='mr',
                          center_conv='gat',
                          ################# overridden params ###############
                          edge_option=None,
                          drop_edge=None,
                          stoch_edge=None,
                          eps_edge=None,
                          ###################################################
                          drop_path=0.0,
                          distillation=True,
                          forced_init=True)
    model.default_cfg = default_cfgs['clustervig']
    return model
    


@register_model
def IsoClusterViG_S_n784_c6(pretrained=False,
                            **kwargs):
    model = IsoClusterViG(n_classes=1000,
                          in_image_size=224,
                          out_image_size=28,
                          in_dim=3,
                          iso_dim=320,
                          act='gelu',
                          dropout_head=0.0,
                          norm='batch_norm',
                          stem_layers=5,
                          n_blocks=16,
                          neighbors=9,
                          clusters=6,
                          vertex_conv='mr',
                          center_conv='gat',
                          #################### overridden params #################
                          edge_option=None,
                          drop_edge=None,
                          stoch_edge=None,
                          eps_edge=None,
                          ########################################################
                          drop_path=0.0,
                          distillation=True,
                          forced_init=True)
    model.default_cfg = default_cfgs['clustervig']
    return model


@register_model
def IsoClusterViG_B_n784_c6(pretrained=False,
                            **kwargs):
    model = IsoClusterViG(n_classes=1000,
                          in_image_size=224,
                          out_image_size=28,
                          in_dim=3,
                          iso_dim=640,
                          act='gelu',
                          dropout_head=0.0,
                          norm='batch_norm',
                          stem_layers=5,
                          n_blocks=16,
                          neighbors=9,
                          clusters=6,
                          vertex_conv='mr',
                          center_conv='gat',
                          #################### overridden params #################
                          edge_option=None,
                          drop_edge=None,
                          stoch_edge=None,
                          eps_edge=None,
                          #########################################################
                          drop_path=0.0,
                          distillation=True,
                          forced_init=True)
    model.default_cfg = default_cfgs['clustervig']
    return model