RMODE_DOUBLE = 0
RMODE_SINGLE = 1

import math

import numpy as np

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1

def quantize_multiplier(real: float) -> tuple[int, int]:
    if real == 0.0:
        return 0, 0
    q, shift = math.frexp(real)
    q_fixed = math.floor(q * (1 << 31) + 0.5)
    assert q_fixed <= (1 << 31)
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    if shift > 30:
        return INT32_MAX, 30
    return int(q_fixed), int(shift)

def effective_scale(in_scale, w_scale, out_scale) -> float:
    return float(np.float64(np.float32(in_scale)) * np.float64(np.float32(w_scale))
                 / np.float64(np.float32(out_scale)))

def wrap32(x):
    x = np.asarray(x, dtype=np.int64)
    return ((x + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)

def srdhm(a, b):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    overflow = (a == INT32_MIN) & (b == INT32_MIN)
    ab = a * b
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    s = ab + nudge
    hi = np.where(s >= 0, s >> 31, -((-s) >> 31))
    return np.where(overflow, INT32_MAX, hi)

def rounding_divide_by_pot(x, exponent):
    x = np.asarray(x, dtype=np.int64)
    exponent = np.asarray(exponent, dtype=np.int64)
    mask = (np.int64(1) << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (x < 0)
    return (x >> exponent) + (remainder > threshold)

def multiply_by_quantized_multiplier(x, mult, shift):
    shift = np.asarray(shift, dtype=np.int64)
    left = np.maximum(shift, 0)
    right = np.maximum(-shift, 0)
    return rounding_divide_by_pot(srdhm(wrap32(np.asarray(x, dtype=np.int64) << left), mult), right)

def multiply_by_quantized_multiplier_single(x, mult, shift):
    x = np.asarray(x, dtype=np.int64)
    total = 31 - np.asarray(shift, dtype=np.int64)
    return (x * np.asarray(mult, dtype=np.int64) + (np.int64(1) << (total - 1))) >> total

def requantize(acc, bias, mult, shift, out_zp, act_min, act_max, rmode=RMODE_DOUBLE):
    x = np.asarray(acc, dtype=np.int64) + np.asarray(bias, dtype=np.int64)
    assert np.all((x >= INT32_MIN) & (x <= INT32_MAX)), "int32 overflow before requant"
    if rmode == RMODE_SINGLE:
        y = multiply_by_quantized_multiplier_single(x, mult, shift)
    else:
        y = multiply_by_quantized_multiplier(x, mult, shift)
    return np.clip(y + out_zp, act_min, act_max).astype(np.int8)
