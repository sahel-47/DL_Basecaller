import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
from brevitas.inject.enum import *
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.quant.solver import WeightQuantSolver, ActQuantSolver
import torch

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

class Weights8bitQuantizer(WeightsQuantizer):
    bit_width = 8

class Weights4bitQuantizer(WeightsQuantizer):
    bit_width = 4

class Activations8bitQuantizer(ActivationsQuantizer):
    bit_width = 8

class Activations4bitQuantizer(ActivationsQuantizer):
    bit_width = 4




