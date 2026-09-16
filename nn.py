import torch 
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import toml 
from pathlib import Path

from collections import OrderedDict



layers = {}

def register(layer):
    layer.name = layer.__name__.lower()
    layers[layer.name] = layer 
    return layer
 
register(torch.nn.ReLU)
register(torch.nn.Tanh)
register(torch.nn.RMSNorm)

@register 
class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias= True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias 
        self.linear = nn.Linear(
            in_features= in_features, out_features= out_features, bias= bias
        )

    def forward(self, x):
        return self.linear(x)

    def to_dict(self, include_weights = False):
        res = {
            "in_features" : self.in_features,
            "out_features" : self.out_features,
            "bias" : self.bias 
        }

        if include_weights:
            res['params'] = {
                'W' : self.linear.weight,
                'b' : self.linear.bias 
            }
        return res 


@register
class Swish(nn.SiLU):
    pass 


@register 
class Clamp(nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min 
        self.max = max 

    def forward(self,x):
        return torch.clamp(x, min=self.min, max=self.max)

    def to_dict(self,include_weights = False):
        return {
            'min' : self.min,
            'max' : self.max 
        }


@register 
class Serial(nn.Sequential):
    def __init__(self, sublayers):
        super().__init__(*sublayers)

    def forward(self, x, return_features = False):
        if return_features:
            intermediate_activations = []
            for layer in self:
                x = layer(x)
                intermediate_activations.append(x)
            return x, intermediate_activations
        return super().forward(x)

    def to_dict(self, include_weights = False):

        return {
            'sublayers': [to_dict(layer, include_weights) for layer in self._modules.values()]

        }

    def __repr__(self):
        return torch.nn.ModuleList.__repr__(self)

@register 
class Stack(Serial):

    @classmethod 
    def from_dict(cls, model_dict, layer_types = None):
        return cls([from_dict(model_dict["layer"], layer_types) for _ in range(model_dict["depth"])])


    def to_dict(self, include_weights = False):
        if include_weights:
           raise NotImplementedError

        layer_dicts = [to_dict(layer) for layer in self]

        for layer_dict in layer_dicts[1:]:

            assert layer_dict == layer_dicts[0], "All layers should be identical"

        return {"layer": layer_dicts[0], "depth": len(self)}

@register 
class NamedSerial(nn.Sequential):

    @classmethod
    def from_dict(cls, model_dict, layer_types=None):
        return cls({k : from_dict(v, layer_types) for k, v in model_dict.items()})


    def __init__(self, layers):
        super().__init__(OrderedDict(layers.items()))

    def to_dict(self, include_weights = False):
        if include_weights:
            raise NotImplementedError
        return {k: to_dict(v) for k, v in self.named_children()}


@register 
class SwiGLU(nn.Module):

    def __init__(self, in_features, hidden_features, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features * 2, bias = bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x):
        y = self.fc1(x)
        up, gate = y.chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * up)

@register 
class RotaryEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim 
        inv_freq = 1.0/(10000 ** (torch.arange(0, dim, 2).float()/dim))
        self.register_buffer("inv_freq", inv_freq.detach(), persistent=False)

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim = -1)

    def forward(self, qkv):
        N, T, _, H, D = qkv.shape

        pos = torch.arange(0,T, device=qkv.device, dtype=qkv.dtype)
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, None, None, :]
        sin = emb.sin()[None, :, None, None, :]
        q,k,v = torch.chunk(qkv, 3, dim= 2)

        q = q* cos + self.rotate_half(q) * sin 
        k = k * cos + self.rotate_half(k) * sin 
        return torch.cat((q,k,v), dim=2)

def sliding_window_mask(seq_len, window, device):
    band = torch.full((seq_len, seq_len), fill_value=1.0)
    band = torch.triu(band, diagonal=-window[0])
    band = band * torch.tril(band, diagonal=window[1])
    band = band.to(torch.bool).to(device)
    return band


@register 
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, qkv_bias=False, out_bias = True, rotary_dim=None, attn_window=None):
        super().__init__()
        assert d_model % nhead ==0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead =  nhead
        self.head_dim = d_model//nhead
        self.rotary_dim = self.head_dim if rotary_dim is None else rotary_dim

        self.Wqkv = nn.Linear(d_model, 3 *d_model, bias=qkv_bias)

        self.out_proj = nn.Linear(d_model, d_model, bias =out_bias)
        self.rotary_emb = RotaryEmbedding(self.rotary_dim)
        self.attn_window = (-1,1) if attn_window is None else tuple(attn_window)



    def attn_func(self, qkv):
        q, k, v = torch.chunk(qkv.permute(0,2,3,1,4), chunks = 3, dim = 1)
        mask = sliding_window_mask(qkv.shape[1],self.attn_window, q.device)
        attn_output = F.scaled_dot_product_attention(q,k,v, attn_mask=mask)
        attn_output = attn_output.permute(0,1,3,2,4) #(Batch, Heads, Time, Dim) ------> (Batch, Time, Heads, Dim).
        return attn_output

    def forward(self, x):
        N, T, _ = x.shape 
        qkv = self.Wqkv(x).view(N, T, 3, self.nhead, self.head_dim)
        qkv = self.rotary_emb(qkv)
        attn_output = self.attn_func(qkv).reshape(N, T, self.d_model)
        out = self.out_proj(attn_output)
        return out 

@register 
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, deepnorm_alpha, deepnorm_beta, attn_window=None):
        super().__init__()
        self.kwargs = {
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "deepnorm_alpha": deepnorm_alpha,
            "deepnorm_beta": deepnorm_beta,
            "attn_window": attn_window     
        }

        self.self_attn = MultiHeadAttention(d_model=d_model, nhead=nhead, attn_window=attn_window)
        self.ff = SwiGLU(d_model, hidden_features=dim_feedforward)
        self.norm1 = torch.nn.RMSNorm(d_model)
        self.norm2 = torch.nn.RMSNorm(d_model)
        self.register_buffer("deepnorm_alpha", torch.tensor(deepnorm_alpha))
        self.reset_parameters()

    def reset_parameters(self):
        db = self.kwargs["deepnorm_beta"]
        d_model = self.kwargs["d_model"]
        torch.nn.init.xavier_normal_(self.ff.fc1.weight, gain=db)
        torch.nn.init.xavier_normal_(self.ff.fc2.weight, gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.out_proj.weight, gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.Wqkv.weight[2*d_model:], gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.Wqkv.weight[:2*d_model], gain=1)           


    def forward(self, x):
        attention = self.self_attn(x)
        residual = x * self.deepnorm_alpha 
        x = self.norm1(attention + residual)
        y = self.ff(x)
        residual = x * self.deepnorm_alpha
        x = self.norm2(y + residual)
        return x

    
    def to_dict(self, include_weights=False):
        if include_weights:
            raise NotImplementedError
        return self.kwargs

    
class MakeContiguous(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.contiguous()

@register
class Convolution(nn.Module):

    def __init__(self, insize, size, winlen, stride=1, padding=0, bias=True, activation=None, norm=None):
        super().__init__()
        self.conv = nn.Conv1d(insize, size, winlen, stride=stride, padding=padding, bias=bias)
        self.activation = layers.get(activation, lambda: activation)()

        if isinstance(norm, dict):
            self.norm = from_dict(norm)
        elif isinstance(norm, str):
            self.norm = layers[norm](size)
        else:
            self.norm = norm


    def forward(self, x):
        h = self.conv(x)
        if self.norm is not None:
            h = self.norm(h)

        if self.activation is not None:
            h = self.activation(h)
        return h
    def to_dict(self, include_weights=False):
        res = {
            "insize": self.conv.in_channels,
            "size": self.conv.out_channels,
            "bias": self.conv.bias is not None,
            "winlen": self.conv.kernel_size[0],
            "stride": self.conv.stride[0],
            "padding": self.conv.padding[0],
        }
        if self.activation is not None:
            res["activation"] = self.activation.name
        if self.norm is not None:
            res["norm"] = to_dict(self.norm, include_weights)
            #simplify default case e.g. norm="batchnorm"
            if not include_weights and self.norm.name in layers:
                if res["norm"] == to_dict(layers[self.norm.name](res["size"])):
                    res["norm"] = self.norm.name

        if include_weights:
            res['params'] = {
                'W': self.conv.weight, 'b': self.conv.bias if self.conv.bias is not None else []
            }
        return res

    
@register
class BatchNorm(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        return self.bn(x)

    def to_dict(self, include_weights=False):
        res = {
            "num_features": self.bn.num_features,
            "eps": self.bn.eps,
            "momentum": self.bn.momentum,
            "affine": self.bn.affine,
            "track_running_stats": self.bn.track_running_stats
        }
        if include_weights:
            params = {}
            if res["affine"]:
                params["W"] = self.bn.weight
                params["b"] = self.bn.bias
            if res["track_running_stats"]:
                params["running_mean"] = self.bn.running_mean
                params["running_var"] = self.bn.running_var
            res["params"] = params
        return res



@register
class LinearUpsample(nn.Module):
    """
    Applies a linear transformation to upsample the sequence length by ``scale_factor``.
    """

    def __init__(self, d_model, scale_factor, batch_first=True):
        super().__init__()
        self.d_model = d_model
        self.scale_factor = scale_factor
        self.batch_first = batch_first
        self.linear = torch.nn.Linear(d_model, self.scale_factor * d_model)

    def forward(self, src):
        if not self.batch_first:
            src = src.permute([1, 0, 2])
        N, L, E = src.shape
        h = self.linear(src).reshape(N, self.scale_factor * L, E)
        if not self.batch_first:
            h = h.permute([1, 0, 2])
        return h

    def output_stride(self, input_stride):
        return input_stride // self.scale_factor

    def to_dict(self, include_weights=False):
        if include_weights:
            raise NotImplementedError
        return {
            "d_model": self.d_model,
            "scale_factor": self.scale_factor,
            "batch_first": self.batch_first
        }

@register
class LinearCRFEncoder(nn.Module):

    def __init__(self, insize, n_base, state_len, bias=True, scale=None, activation=None, blank_score=None, expand_blanks=True, permute=None):
        super().__init__()
        self.scale = scale
        self.n_base = n_base
        self.state_len = state_len
        self.blank_score = blank_score
        self.expand_blanks = expand_blanks
        size = (n_base + 1) * n_base**state_len if blank_score is None else n_base**(state_len + 1)
        self.linear = torch.nn.Linear(insize, size, bias=bias)
        self.activation = layers.get(activation, lambda: activation)()
        self.permute = permute

    def forward(self, x):
        if self.permute is not None:
            x = x.permute(*self.permute)
        scores = self.linear(x)
        if self.activation is not None:
            scores = self.activation(scores)
        if self.scale is not None:
            scores = scores * self.scale
        if self.blank_score is not None and self.expand_blanks:
            T, N, C = scores.shape
            scores = torch.nn.functional.pad(
                scores.view(T, N, C // self.n_base, self.n_base),
                (1, 0, 0, 0, 0, 0, 0, 0),
                value=self.blank_score
            ).view(T, N, -1)
        return scores

    def to_dict(self, include_weights=False):
        res = {
            'insize': self.linear.in_features,
            'n_base': self.n_base,
            'state_len': self.state_len,
            'bias': self.linear.bias is not None,
            'scale': self.scale,
            'blank_score': self.blank_score,
            'expand_blanks': self.expand_blanks,
        }
        if self.activation is not None:
            res['activation'] = self.activation.name
        if self.permute is not None:
            res['permute'] = self.permute
        if include_weights:
            res['params'] = {
                'W': self.linear.weight, 'b': self.linear.bias
                if self.linear.bias is not None else []
            }
        return res

    def extra_repr(self):
        rep = 'n_base={}, state_len={}, scale={}, blank_score={}, expand_blanks={}'.format(
            self.n_base, self.state_len, self.scale, self.blank_score, self.expand_blanks
        )
        if self.permute:
            rep += f', permute={self.permute}'
        return rep



@register
class Permute(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

    def to_dict(self, include_weights=False):
        return {'dims': self.dims}

    def extra_repr(self):
        return 'dims={}'.format(self.dims)


def truncated_normal(size, dtype=torch.float32, device=None, num_resample=5):
    x = torch.empty(size + (num_resample,), dtype=torch.float32, device=device).normal_()
    i = ((x < 2) & (x > -2)).max(-1, keepdim=True)[1]
    return torch.clamp_(x.gather(-1, i).squeeze(-1), -2, 2)





def to_dict(layer, include_weights=False):
    if hasattr(layer, 'to_dict'):
        return {'type': layer.name, **layer.to_dict(include_weights)}
    return {'type': layer.name}


def from_dict(model_dict, layer_types=None):
    if not isinstance(model_dict, dict):
        return model_dict

    model_dict = model_dict.copy()

    if layer_types is None:
        layer_types = layers 
    type_name = model_dict.pop("type")
    typ = layer_types[type_name]

    if hasattr(typ, "from_dict"):
        return typ.from_dict(model_dict, layer_types)

    if 'sublayers' in model_dict:

        sublayers = model_dict['sublayers']

        model_dict['sublayers'] = [
            from_dict(x, layer_types) for x in sublayers
        ] if isinstance(sublayers,list) else from_dict(sublayers, layer_types)

    try:
        layer = typ(**model_dict)
    except Exception as e:
        raise Exception(f'Failed to build layer of type {typ} with args {model_dict}') from e

    return layer 



def fuse_bn_(m):
    """
    Sets the module m to eval mode and if a Convolution fuses any batchnorm layer.
    """
    m.training = False
    if isinstance(m, Convolution) and isinstance(m.norm, BatchNorm):
        m.conv = fuse_conv_bn_eval(m.conv, m.norm.bn)
        m.norm = None
