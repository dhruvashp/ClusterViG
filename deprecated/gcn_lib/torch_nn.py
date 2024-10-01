# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d


##############################
#    Basic layers
##############################
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
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


def norm_layer(norm, nc):
    # normalization layer 2d
    norm = norm.lower()
    if norm == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


class MLP(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias))
            if act is not None and act.lower() != 'none':            # act is first
                m.append(act_layer(act))
            if norm is not None and norm.lower() != 'none':          # then norm
                m.append(norm_layer(norm, channels[-1]))             # again channels[-1] appears !
        super(MLP, self).__init__(*m)


class BasicConv(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=bias, groups=4))
            if norm is not None and norm.lower() != 'none':           # norm is first
                m.append(norm_layer(norm, channels[-1]))                                    # DP: why channels[-1]? should it not be channels[i]? SOFT ASSUMPTION that all are same?
            if act is not None and act.lower() != 'none':             # then act
                m.append(act_layer(act))
            if drop > 0:
                m.append(nn.Dropout2d(drop))

        super(BasicConv, self).__init__(*m)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()




def batched_index_select(x, idx):
    r"""fetches neighbors features from a given neighbor idx

    -------------------------------
    DP
    -------------------------------
    Lifts from x according to idx
    That is,
    x       ->      B, C, N_reference, 1              (why? think B, C, H, W)
    idx     ->      B, N, K                           (value is an index to be feature-lifted) 
    
    output  ->      B, C, N, K

    Essentially,
    
    for, 
    idx(B=b, N=i, K=j) = INDEX
    map in, 
    output(B=b, :, N=i, K=j) 
    the feature vector
    x[B=b, :, N_reference=INDEX, :]

    NOTE that PyTorch scatter/gather functionality might implement this batched_index_select in 1-2 lines and
    more efficiently, likely
    -------------------------------
    Args:
        x (Tensor): input feature Tensor  # DP: (B, C, N, 1)
                :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times 1}`.
        idx (Tensor): edge_idx            # DP: (B, N, K) ?
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times l}`.
    Returns:
        Tensor: output neighbors features
            :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times k}`.
    """
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]      # x -> B, C, N, 1 ? why num_vertices_reduced ?
    _, num_vertices, k = idx.shape                                # x -> B, N', k
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)        # B, N, C, 1
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]          # num_features * C (num_features = B x num_vertices x k)
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()      # B, C, N, K      (lifted tensor)            
    return feature



def batched_index_select_improved(x: torch.Tensor, idx: torch.Tensor):
    '''
    Better implementation of batched_index_select, does the same thing
    
    x   ->  B, C, N, 1
    idx ->  B, N_out, K

    out ->  B, C, N_out, K

    '''

    n_channels = x.shape[1]
    idx = idx.unsqueeze(dim=1)                              # B, 1, N_out, K
    idx = idx.expand(size=(-1, n_channels, -1, -1))         # B, C, N_out, K
    B, C, N_out, K = idx.shape
    idx = idx.reshape(shape=(B, C, N_out*K))                # B, C, N_out*K
    x = x.squeeze(dim=-1)                                   # B, C, N
    out = torch.gather(input=x,
                       dim=-1,
                       index=idx)                           # B, C, N_out*K
    out = out.reshape(shape=(B, C, N_out, K))               # B, C, N_out, K
    return out