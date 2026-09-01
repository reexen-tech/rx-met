"""ONNX 输入 PTQ 主线要复用的 RX 量化原文。

只收这条主线用到的、且不依赖 Torch / ONNX quantizer API 的原函数：
选 Po2 scale、改 scale 后重算 min/max、以及 ``Sb = Sx * Sw`` 的范围。
不要把 QuantGRU、QAT freeze、导出后处理等其它路径塞进来。
"""

import math
from typing import Optional, Tuple, Union


def is_power_of_2(value: float, relative_tolerance: float = 0.02) -> bool:
    """
    判断一个值是否接近 2 的幂次方（考虑相对误差）
    
    由于量化校准时的浮点数误差和统计波动，real_range 可能不会精确等于 2^n。
    例如：real_range = 2.007884，非常接近 2.0 (误差 0.39%)，应该被视为"是 2^n"。
    
    Args:
        value: 要检查的值
        relative_tolerance: 相对容差（默认 2%）
            - 如果 |value - 2^n| / 2^n < relative_tolerance，则认为是 2^n
            - 例如：2% 容差允许 [1.96, 2.04] 范围内的值被视为 2.0
        
    Returns:
        是否接近 2 的幂次方
        
    Examples:
        >>> is_power_of_2(2.0)          # True (精确匹配)
        >>> is_power_of_2(2.007884)     # True (误差 0.39% < 2%)
        >>> is_power_of_2(2.05)         # False (误差 2.5% > 2%)
        >>> is_power_of_2(1.98)         # True (误差 1% < 2%)
    """
    if value <= 0:
        return False
    
    # 计算 log2(value)
    log2_val = math.log2(value)
    
    # 找到最接近的整数 n
    n_nearest = round(log2_val)
    
    # 计算最接近的 2^n
    nearest_power_of_2 = 2 ** n_nearest
    
    # 计算相对误差
    relative_error = abs(value - nearest_power_of_2) / nearest_power_of_2
    
    # 检查相对误差是否在容差范围内
    return relative_error < relative_tolerance


def find_closest_power_of_2_scale(scale: float, method: str = "round", 
                                   qmin: float = None, qmax: float = None,
                                   rmin: float = None, rmax: float = None,
                                   tolerance: float = 0.02) -> Tuple[float, int]:
    """
    找到最接近给定scale的2的幂次方scale (1/2^n)
    
    Args:
        scale: 原始scale值
        method: 选择策略
            - "round": 四舍五入到最近的 2^n（默认）
            - "cover_range": 智能策略
                - 如果 real_max - real_min 接近 2^n（相对误差 < tolerance）：使用四舍五入
                - 否则：向上取整以覆盖原范围
        qmin: 量化最小值（method="cover_range" 时需要）
        qmax: 量化最大值（method="cover_range" 时需要）
        rmin: 浮点最小值（method="cover_range" 时需要）
        rmax: 浮点最大值（method="cover_range" 时需要）
        tolerance: 判断是否为 2^n 的相对容差（默认 2%）
            - 例如：real_range=2.007884 与 2.0 的相对误差 0.39% < 2%，被视为 2^n
        
    Returns:
        (new_scale, n): 新的scale值和对应的n值，其中new_scale = 1/2^n
    """
    if scale <= 0:
        raise ValueError(f"Scale必须为正数，得到: {scale}")
    
    # 计算log2(1/scale) = -log2(scale)
    # 如果scale = 1/2^n, 那么log2(1/scale) = n
    n = -math.log2(scale)
    
    if method == "round":
        # 四舍五入到最近的整数（最小化 |new_scale - old_scale|）
        n_rounded = round(n)
    elif method == "cover_range":
        # 智能策略：判断 real_range 是否接近 2^n
        if qmin is None or qmax is None or rmin is None or rmax is None:
            raise ValueError("cover_range 方法需要提供 qmin, qmax, rmin, rmax 参数")
        
        # 计算 real_range
        real_range = abs(rmax - rmin)
        
        # 判断 real_range 是否接近 2^n（使用相对容差）
        if is_power_of_2(real_range, relative_tolerance=tolerance):
            # real_range 接近 2^n，使用四舍五入
            n_rounded = round(n)
        else:
            # real_range 不接近 2^n，向上取整以覆盖原范围
            # 向上取整 = floor(n)，这样 scale 会更大，覆盖范围更大
            n_rounded = math.floor(n)
    else:
        raise ValueError(f"不支持的 method: {method}，必须是 'round' 或 'cover_range'")
    
    # 计算新的scale = 1/2^n
    new_scale = 1.0 / (2 ** n_rounded)
    
    return new_scale, n_rounded


def verify_power_of_2_scale(scale: float, tolerance: float = 1e-9) -> Tuple[bool, int]:
    """
    验证给定的scale是否为2的幂次方
    
    Args:
        scale: 要验证的scale值
        tolerance: 容差
        
    Returns:
        (is_power_of_2, n): 是否为2的幂次方，以及对应的n值
    """
    if scale <= 0:
        return False, -1
    
    # 计算log2(1/scale)
    n = -math.log2(scale)
    n_rounded = round(n)
    
    # 检查是否接近整数
    if abs(n - n_rounded) < tolerance:
        expected_scale = 1.0 / (2 ** n_rounded)
        if abs(scale - expected_scale) / expected_scale < tolerance:
            return True, n_rounded
    
    return False, -1


def recompute_min_max_for_new_scale(
    new_scale: float,
    qmin: float,
    qmax: float,
    rmin: float,
    offset: Optional[float],
    symmetric: bool,
) -> Tuple[Optional[float], float, float, float, float]:
    """
    原文来自 ``aimet_torch.power_of_2_quantization.modify_quantizer_to_power_of_2``
    标量分支：保持 Qmin, Qmax 不变，按新 scale 重算 rmin / rmax / offset。
    """
    # ！！！新策略：保持 Qmin, Qmax 不变，修改 rmin, rmax！！！
    # 1. 修改 scale 为 power-of-2
    # 2. 保持 Qmin, Qmax 不变（量化域固定）
    # 3. 对称量化（权重）：offset 保持不变（通常为0），重新计算 rmin, rmax
    # 4. 非对称量化（激活）：重新计算 offset, rmin, rmax

    if symmetric or offset is None:
        # 对称量化：保持 qmin, qmax, offset 不变，重新计算 rmin, rmax
        new_qmin_val = qmin  # 保持不变
        new_qmax_val = qmax  # 保持不变

        if offset is not None:
            old_offset_val = offset
            new_offset = offset  # 保持不变
            # 重新计算 rmin, rmax
            new_min_val = new_scale * (new_qmin_val + old_offset_val)
            new_max_val = new_scale * (new_qmax_val + old_offset_val)
        else:
            new_offset = None
            new_min_val = new_scale * new_qmin_val
            new_max_val = new_scale * new_qmax_val
    else:
        # 非对称量化：保持 qmin, qmax 不变，保持 rmin 不变，调整 rmax
        new_qmin_val = qmin  # 保持不变
        new_qmax_val = qmax  # 保持不变

        # 保持 rmin 不变，重新计算 offset
        # rmin = scale * (qmin + offset)
        # offset = rmin / scale - qmin
        new_offset = rmin / new_scale - qmin

        # 重新计算 rmax（会变大，因为 scale 变大了）
        # rmax = scale * (qmax + offset)
        new_max_val = new_scale * (qmax + new_offset)
        new_min_val = rmin  # 保持不变

    return new_offset, new_min_val, new_max_val, new_qmin_val, new_qmax_val


def compute_aligned_bias_range(
    input_scale: float,
    weight_scale: float,
    bias_bitwidth: int,
) -> Tuple[float, Union[int, float], Union[int, float], float, float]:
    """
    原文来自 ``aimet_torch.utils_rx._align_conv_bias_scale`` 步骤 2-3：
    ``Sb = Sx * Sw``，再按对称整数域算新的浮点范围。
    """
    # ===== 步骤 2: 计算新的 bias scale =====
    Sb_new = float(input_scale) * float(weight_scale)

    # ===== 步骤 3: 计算新的浮点表示范围 =====
    # 计算量化整数范围（对称量化）
    Qmax = 2 ** (bias_bitwidth - 1) - 1
    Qmin = -2 ** (bias_bitwidth - 1)

    fmin_new = Qmin * Sb_new
    fmax_new = Qmax * Sb_new
    return Sb_new, Qmin, Qmax, fmin_new, fmax_new
