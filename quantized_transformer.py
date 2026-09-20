import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat, Int32Bias
from brevitas.inject.enum import *
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.quant.solver import WeightQuantSolver, ActQuantSolver
import torch
import torch.nn as nn 
import torch.nn.functional as F
import re
import math
from nn import register, layers 



quant_configs_dict = {}


def save_quant_config(quant_config):
    name = quant_config.__name__.lower()
    pattern = r"(weights|activations)\d+"
    match = re.match(pattern, name)
    if match:
        key_name = match.group(0)
        quant_configs_dict[key_name] = quant_config

    return quant_config

def set_quantizer(quantizer_type, bit_width):
    dict_idx = (quantizer_type) + str(bit_width)
    return quant_configs_dict[dict_idx]


@save_quant_config
class WeightsQuantizer(WeightQuantSolver):
    quant_type = QuantType.INT 
    bit_width_impl_type = BitWidthImplType.CONST 
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_stats_op = StatsOp.MAX 
    restrict_scaling_type = RestrictValueType.FP 
    scaling_per_output_channel = False 
    zero_point_impl = ZeroZeroPoint
    bit_width = 8
    signed = True 
    narrow_range = False 
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS

@save_quant_config
class ActivationsQuantizer(ActQuantSolver):
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST 
    float_to_int_impl_type = FloatToIntImplType.ROUND 
    scaling_stats_op = StatsOp.PERCENTILE
    high_percentile_q = 99.9
    collect_stats_steps = 300 
    restrict_scaling_type = RestrictValueType.FP 
    scaling_per_output_channel = False 
    zero_point_impl = ZeroZeroPoint
    bit_width = 8
    signed = True 
    narrow_range = False 
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS

@save_quant_config
class Weights8bitQuantizer(WeightsQuantizer):
    bit_width = 8

@save_quant_config
class Weights4bitQuantizer(WeightsQuantizer):
    bit_width = 4

@save_quant_config
class Activations8bitQuantizer(ActivationsQuantizer):
    bit_width = 8

@save_quant_config
class Activations4bitQuantizer(ActivationsQuantizer):
    bit_width = 4

@save_quant_config
class Activations32bitQuantizer(ActivationsQuantizer):
    bit_width = 32



QuantizedTransformerEncoderConfig = {
    "MHA_INPUT_QUANTIZER" : set_quantizer('activations', 8),
    "WQ_PROJ" : set_quantizer('weights', 8),
    "WK_PROJ" : set_quantizer('weights', 8),
    "WV_PROJ" : set_quantizer('weights', 8),
    "ROPE_Q_QUANTIZER" : set_quantizer('activations', 8),
    "ROPE_K_QUANTIZER" : set_quantizer('activations', 8),
    "V_QUANTIZER" : set_quantizer('activations', 8),
    ### ADD QKT input quantizers
    "QKT_OP_QUANTIZER" : set_quantizer('activations', 8),
    "SOFTMAX_OP_QUANTIZER" : set_quantizer('activations', 8),
    "WV_OP_QUANTIZER" : set_quantizer('activations', 8),
    "WO_PROJ" : set_quantizer('weights', 8),
    "WUP_PROJ" : set_quantizer('weights', 8),
    "WGATE_PROJ" : set_quantizer('weights', 8),
    "WDOWN_PROJ" : set_quantizer('weights', 8),
    "SWIGLU_ELEMENTWISE_OP_QUANTIZER" : set_quantizer('activations', 8),
    "RMS_INPUT_QUANTIZER" : set_quantizer('activations', 32),
    "RMS_OUTPUT_QUANTIZER" : set_quantizer('activations', 8)
}


@register
class RoPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim 
        inv_freq = 1.0/(10000.0 ** (torch.arange(0, dim, 2).float()/dim))
        self.register_buffer("inv_freq", inv_freq.detach(), persistent=False)

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        batch, seqlen, nheads, headdim = x.shape

        pos = torch.arange(0, seqlen, device = x.device, dtype= x.dtype)
        freqs = torch.einsum("i,j -> ij", pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, None, :]
        sin = emb.sin()[None, :, None, :] 

        x = x * cos + self.rotate_half(x) * sin

        return x
@register
class InitialTransformerEncoderBlockQuantizer(nn.Module):
    def  __init__(self):
        super().__init__()
        self.transformer_block_input_quantizer = qnn.QuantIdentity(act_quant=QuantizedTransformerEncoderConfig["MHA_INPUT_QUANTIZER"], return_quant_tensor=True)

    def forward(self, x):
        return self.transformer_block_input_quantizer(x) ## returns quantized activation for the very first transformer layer


@register
class QuantizedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, qkv_bias = False, out_bias = True, rotary_dim = None, attn_window = None):
        super().__init__()
        assert d_model % nhead == 0

        self.d_model = d_model 
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rotary_dim = self.head_dim if rotary_dim is None else rotary_dim

        self.rotary_emb = RoPE(self.rotary_dim) 

        self.attn_window = (-1,1) if attn_window is None else tuple(attn_window)

        #transformer-encoder input quantizer

        self.Wqkv_bias_quant = None if not qkv_bias else Int32Bias

        #QKV quantized weights
        self.Wq = qnn.QuantLinear(d_model, d_model, qkv_bias, weight_quant=QuantizedTransformerEncoderConfig["WQ_PROJ"], bias_quant=self.Wqkv_bias_quant, return_quant_tensor=True)
        self.Wk = qnn.QuantLinear(d_model, d_model, qkv_bias, weight_quant=QuantizedTransformerEncoderConfig["WK_PROJ"], bias_quant=self.Wqkv_bias_quant, return_quant_tensor=True)
        self.Wv = qnn.QuantLinear(d_model, d_model, qkv_bias, weight_quant=QuantizedTransformerEncoderConfig["WV_PROJ"], bias_quant=self.Wqkv_bias_quant, return_quant_tensor=True)

        #RoPE output quantizer
        self.rope_q_quantize = qnn.QuantIdentity(act_quant=QuantizedTransformerEncoderConfig["ROPE_Q_QUANTIZER"], return_quant_tensor=True)
        self.rope_k_quantize = qnn.QuantIdentity(act_quant=QuantizedTransformerEncoderConfig["ROPE_K_QUANTIZER"], return_quant_tensor=True)
        self.v_quantize = qnn.QuantIdentity(act_quant=QuantizedTransformerEncoderConfig["V_QUANTIZER"], return_quant_tensor=True)

        # Attention block quantizers
        self.QKT_op_quantizer = qnn.QuantIdentity(act_quant=QuantizedTransformerEncoderConfig["QKT_OP_QUANTIZER"], return_quant_tensor=True)
        self.Softmax_op_quantizer = qnn.QuantIdentity(act_quant=QuantizedTransformerEncoderConfig["SOFTMAX_OP_QUANTIZER"], return_quant_tensor=True)
        self.WV_op_quantizer = qnn.QuantIdentity(act_quant= QuantizedTransformerEncoderConfig["WV_OP_QUANTIZER"], return_quant_tensor=True)

        self.Wo_bias_quant = None if not out_bias else Int32Bias

        ### WO_proj_weights
        self.out_proj = qnn.QuantLinear(d_model, d_model, bias = out_bias, weight_quant=QuantizedTransformerEncoderConfig["WO_PROJ"], bias_quant= self.Wo_bias_quant, return_quant_tensor=True)



    
    def attention_engine(self, q, k, v, mask):
        scores = (q @ k.transpose(-2,-1))/math.sqrt(self.head_dim)
        quantized_scores = self.QKT_op_quantizer(scores)
        # QuantIdentity here
        masked_scores = quantized_scores.masked_fill(~mask, -100000.0)
        probs = torch.softmax(masked_scores, dim=-1)
        quantized_probs =self.Softmax_op_quantizer(probs)
        WV = quantized_probs @ v 
        WV_op = self.WV_op_quantizer(WV)
        return WV_op

    def forward(self, q_x):
        N, T, _ = q_x.shape 
        # q_x = self.multihead_attention_input_quantizer(x)

        q_proj = self.Wq(q_x) 
        k_proj = self.Wk(q_x)
        v_proj = self.Wv(q_x)

        q_proj = q_proj.view(N, T, self.nhead, self.head_dim)
        k_proj = k_proj.view(N, T, self.nhead, self.head_dim)   
        v_proj = v_proj.view(N, T, self.nhead, self.head_dim)


        rotated_q = self.rotary_emb(q_proj)
        rotated_k = self.rotary_emb(k_proj)

        q_rotated_q  = self.rope_q_quantize(rotated_q).transpose(1, 2)
        q_rotated_k  = self.rope_k_quantize(rotated_k).transpose(1, 2)
        q_v = self.v_quantize(v_proj).transpose(1, 2)


        mask = sliding_window_mask(T, self.attn_window, q_x.device)

        q_attn_op = self.attention_engine(q_rotated_q, q_rotated_k, q_v, mask = mask)
        q_attn_op = q_attn_op.transpose(1, 2).contiguous().view(N, T, self.d_model)

        out = self.out_proj(q_attn_op)

        return out

@register
class QuantizedSwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, bias = False):
        super().__init__()
        self.Wup_bias = None if not bias else Int32Bias
        self.Wgate_bias = None if not bias else Int32Bias
        self.Wdown_bias = None if not bias else Int32Bias

        self.Wup =  qnn.QuantLinear(in_features, hidden_features, bias=bias, weight_quant=QuantizedTransformerEncoderConfig["WUP_PROJ"],bias_quant=self.Wup_bias, return_quant_tensor=True)
        self.Wgate = qnn.QuantLinear(in_features, hidden_features, bias=bias , weight_quant=QuantizedTransformerEncoderConfig["WGATE_PROJ"], bias_quant=self.Wgate_bias, return_quant_tensor=True)
        self.Wdown = qnn.QuantLinear(hidden_features, in_features, bias=bias, weight_quant=QuantizedTransformerEncoderConfig["WDOWN_PROJ"], bias_quant=self.Wdown_bias,  return_quant_tensor=True)
        self.swiglu_quantize = qnn.QuantIdentity(act_quant=QuantizedTransformerEncoderConfig["SWIGLU_ELEMENTWISE_OP_QUANTIZER"], return_quant_tensor=True)
        # self.down_proj_quantize = qnn.QuantIdentity(act_quant=QuantizedTransformerEncoderConfig["WDOWN_OP_QUANTIZER"], return_quant_tensor=True)

    def forward(self, x):
        up_op = self.Wup(x)
        gate_op = self.Wgate(x)
        silu_activated_op = F.silu(gate_op)
        element_wise_mul = silu_activated_op * up_op
        # insert quantizer here 
        q_element_wise = self.swiglu_quantize(element_wise_mul)
        return self.Wdown(q_element_wise)


@register
class QuantizedRMSNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-5,  zero_centered_weight = False, device = None, dtype = None):
        super().__init__()
        kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.eps = eps 
        self.zero_centered_weight = zero_centered_weight
        self.g_weights = nn.Parameter(torch.empty(d_model, **kwargs))
        self.RMS_in_quantizer = qnn.QuantIdentity(act_quant=QuantizedTransformerEncoderConfig["RMS_INPUT_QUANTIZER"])

        self.reset_parameters()

    def reset_parameters(self):
        if not self.zero_centered_weight:
            torch.nn.init.ones_(self.g_weights)
        else:
            torch.nn.init.zeros_(self.g_weights)

    def forward(self, x):

        x_int = self.RMS_in_quantizer(x)
        quant_in_features_rms = torch.sum(x_int **2, dim = -1, keepdim=True)
        quant_in_features_rms = quant_in_features_rms / self.d_model
        quant_in_features_rms = torch.rsqrt(quant_in_features_rms + self.eps)
        final_rms_norm = x * quant_in_features_rms * self.g_weights
        return final_rms_norm

    
@register
class QuantizedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, deepnorm_alpha, deepnorm_beta, attn_window = None):
        super().__init__()
        self.kwargs = {
            "d_model" : d_model,
            "nhead" : nhead,
            "dim_feedforward" : dim_feedforward,
            "deepnorm_alpha" : deepnorm_alpha,
            "deepnorm_beta" : deepnorm_beta,
            "attn_window" : attn_window
        }

        self.multihead_attention_input_quantizer = qnn.QuantIdentity(act_quant=QuantizedTransformerEncoderConfig["MHA_INPUT_QUANTIZER"], return_quant_tensor=True)
        self.self_attn = QuantizedMultiHeadAttention(d_model, nhead, attn_window=attn_window)
        self.ff = QuantizedSwiGLU(d_model, hidden_features=dim_feedforward)
        self.norm1 = QuantizedRMSNorm(d_model)
        self.rms1_out_quantizer = qnn.QuantIdentity(act_quant=QuantizedTransformerEncoderConfig["RMS_OUTPUT_QUANTIZER"], return_quant_tensor=True)
        self.norm2 = QuantizedRMSNorm(d_model)
        self.rms2_out_quantizer = qnn.QuantIdentity(act_quant=QuantizedTransformerEncoderConfig["RMS_OUTPUT_QUANTIZER"], return_quant_tensor=True)
        self.register_buffer("deepnorm_alpha", torch.tensor(deepnorm_alpha))
        self.reset_parameters()

    def reset_parameters(self):
        db = self.kwargs["deepnorm_beta"]
        torch.nn.init.xavier_normal_(self.ff.Wgate.weight, gain=db)
        torch.nn.init.xavier_normal_(self.ff.Wup.weight, gain=db)
        torch.nn.init.xavier_normal_(self.ff.Wdown.weight, gain=db)        
        torch.nn.init.xavier_normal_(self.self_attn.out_proj.weight, gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.Wq.weight, gain=1.0)
        torch.nn.init.xavier_normal_(self.self_attn.Wk.weight, gain=1.0)
        torch.nn.init.xavier_normal_(self.self_attn.Wv.weight, gain=db)

    def forward(self, x_q):
        # x_q = self.multihead_attention_input_quantizer(x) ---> ig this should be only there in the beginning of the very first layer
        attn_out = self.self_attn(x_q)
        residual1 = x_q * self.deepnorm_alpha + attn_out
        rms_norm_out1 = self.rms1_out_quantizer(self.norm1(residual1))
        ffn_op = self.ff(rms_norm_out1)
        residual2 = rms_norm_out1 * self.deepnorm_alpha + ffn_op
        encoder_block_op = self.rms2_out_quantizer(self.norm2(residual2)) 
        return encoder_block_op
        


def sliding_window_mask(seq_len, window, device):
    band = torch.full((seq_len, seq_len), fill_value=1.0)
    band = torch.triu(band, diagonal=-window[0])
    band = band * torch.tril(band, diagonal=window[1])
    band = band.to(torch.bool).to(device)
    return band


