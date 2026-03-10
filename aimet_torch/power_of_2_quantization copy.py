"""
Power-of-2 量化工具
用于将量化器的scale修改为2的幂次方，以便硬件实现
"""

import torch
import math
from typing import Dict, List, Tuple
from aimet_torch.v2.nn import QuantizationMixin


def print_quantizer_info(model, title="量化器信息"):
    """
    打印模型中所有量化器的完整信息
    
    Args:
        model: 量化模型
        title: 打印标题
    """
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80)
    
    quantizer_info = []
    
    for name, module in model.named_modules():
        # 检查输入量化器（QuantizationMixin 模块）
        if isinstance(module, QuantizationMixin) and hasattr(module, 'input_quantizers'):
            for idx, quantizer in enumerate(module.input_quantizers):
                if quantizer is not None and quantizer.is_initialized():
                    info = get_single_quantizer_info(quantizer)
                    info['module_name'] = name
                    info['quantizer_type'] = f'input_quantizer[{idx}]'
                    quantizer_info.append(info)
        
        # 检查输出量化器（QuantizationMixin 模块）
        if isinstance(module, QuantizationMixin) and hasattr(module, 'output_quantizers'):
            for idx, quantizer in enumerate(module.output_quantizers):
                if quantizer is not None and quantizer.is_initialized():
                    info = get_single_quantizer_info(quantizer)
                    info['module_name'] = name
                    info['quantizer_type'] = f'output_quantizer[{idx}]'
                    quantizer_info.append(info)
        
        # 检查参数量化器（所有模块，包括非 QuantizationMixin）
        if hasattr(module, 'param_quantizers'):
            for param_name, quantizer in module.param_quantizers.items():
                if quantizer is not None and quantizer.is_initialized():
                    info = get_single_quantizer_info(quantizer)
                    info['module_name'] = name
                    info['quantizer_type'] = f'param_quantizer[{param_name}]'
                    quantizer_info.append(info)
    
    # 打印信息
    for i, info in enumerate(quantizer_info, 1):
        print(f"\n{i}. {info['module_name']} - {info['quantizer_type']}")
        print(f"   Qmin: {info['qmin']}, Qmax: {info['qmax']}")
        print(f"   Min: {info['min']:.6f}, Max: {info['max']:.6f}")
        
        # 打印 scale 信息
        if 'scale_tensor' in info:
            # Per-channel 量化
            scale_tensor = info['scale_tensor']
            num_channels = scale_tensor.numel()
            print(f"   Scale (per-channel): {num_channels} 个通道")
            
            if num_channels <= 10:
                # 通道数少，全部打印
                print(f"     {'Channel':>10s} {'Scale':>20s} {'Reciprocal':>15s}")
                for ch_idx in range(num_channels):
                    scale_val = scale_tensor.flatten()[ch_idx].item()
                    recip = 1/scale_val if scale_val != 0 else float('inf')
                    print(f"     {ch_idx:>10d} {scale_val:>20.10f} {recip:>15.2f}")
            else:
                # 通道数多，打印统计信息和前后几个
                print(f"   Scale 范围: [{scale_tensor.min().item():.10f}, {scale_tensor.max().item():.10f}]")
                print(f"   Scale 平均: {scale_tensor.mean().item():.10f}")
                print(f"\n   前5个通道:")
                print(f"     {'Channel':>10s} {'Scale':>20s} {'Reciprocal':>15s}")
                for ch_idx in range(min(5, num_channels)):
                    scale_val = scale_tensor.flatten()[ch_idx].item()
                    recip = 1/scale_val if scale_val != 0 else float('inf')
                    print(f"     {ch_idx:>10d} {scale_val:>20.10f} {recip:>15.2f}")
                
                if num_channels > 10:
                    print(f"     ... (省略中间 {num_channels-10} 个通道) ...")
                    print(f"\n   后5个通道:")
                    for ch_idx in range(num_channels - 5, num_channels):
                        scale_val = scale_tensor.flatten()[ch_idx].item()
                        recip = 1/scale_val if scale_val != 0 else float('inf')
                        print(f"     {ch_idx:>10d} {scale_val:>20.10f} {recip:>15.2f}")
        else:
            # 标量 scale
            print(f"   Scale: {info['scale']:.8f}")
            print(f"   Scale_reciprocal: {1/info['scale']:.2f}")
        
        print(f"   Zero Point (Offset): {info['zero_point']:.2f}")
        print(f"   Symmetric: {info['symmetric']}")
        print(f"   Shape: {info['shape']}")
    
    print(f"\n总共 {len(quantizer_info)} 个量化器")
    print("="*80 + "\n")
    
    return quantizer_info


def get_single_quantizer_info(quantizer) -> Dict:
    """
    获取单个量化器的信息
    
    Args:
        quantizer: 量化器对象
        
    Returns:
        包含量化器信息的字典
    """
    info = {}
    info['qmin'] = quantizer.qmin
    info['qmax'] = quantizer.qmax
    info['symmetric'] = quantizer.symmetric
    info['shape'] = quantizer.shape
    
    # 获取min和max
    min_val = quantizer.get_min()
    max_val = quantizer.get_max()
    
    # 转换为标量（如果是张量的话）
    if isinstance(min_val, torch.Tensor):
        info['min'] = min_val.min().item()  # 取最小值作为代表
        info['max'] = max_val.max().item()  # 取最大值作为代表
        info['min_tensor'] = min_val
        info['max_tensor'] = max_val
    else:
        info['min'] = min_val
        info['max'] = max_val
    
    # 获取scale和offset
    scale = quantizer.get_scale()
    offset = quantizer.get_offset()
    
    if isinstance(scale, torch.Tensor):
        info['scale'] = scale.mean().item()  # 取平均值作为代表
        info['scale_tensor'] = scale
    else:
        info['scale'] = scale
    
    if offset is not None:
        if isinstance(offset, torch.Tensor):
            info['zero_point'] = offset.mean().item()
            info['zero_point_tensor'] = offset
        else:
            info['zero_point'] = offset
    else:
        info['zero_point'] = 0.0
    
    return info


def find_closest_power_of_2_scale(scale: float) -> Tuple[float, int]:
    """
    找到最接近给定scale的2的幂次方scale (1/2^n)
    
    Args:
        scale: 原始scale值
        
    Returns:
        (new_scale, n): 新的scale值和对应的n值，其中new_scale = 1/2^n
    """
    if scale <= 0:
        raise ValueError(f"Scale必须为正数，得到: {scale}")
    
    # 计算log2(1/scale) = -log2(scale)
    # 如果scale = 1/2^n, 那么log2(1/scale) = n
    n = -math.log2(scale)
    
    # 向下取整
    n_rounded = int(n)
    
    # 计算新的scale = 1/2^n
    new_scale = 1.0 / (2 ** n_rounded)
    
    return new_scale, n_rounded


def apply_power_of_2_quantization(model, verbose=True):
    """
    将模型中所有量化器的scale修改为2的幂次方
    
    Args:
        model: 量化模型
        verbose: 是否打印详细信息
        
    Returns:
        修改统计信息
    """
    print("\n" + "="*80)
    print("应用 Power-of-2 量化")
    print("="*80)
    
    total_quantizers = 0
    modified_quantizers = 0
    
    for name, module in model.named_modules():
        # 处理输入量化器（只在 QuantizationMixin 模块中）
        if isinstance(module, QuantizationMixin) and hasattr(module, 'input_quantizers'):
            for idx, quantizer in enumerate(module.input_quantizers):
                if quantizer is not None and quantizer.is_initialized():
                    total_quantizers += 1
                    if modify_quantizer_to_power_of_2(quantizer, f"{name}.input_quantizer[{idx}]", verbose):
                        modified_quantizers += 1
        
        # 处理输出量化器（只在 QuantizationMixin 模块中）
        if isinstance(module, QuantizationMixin) and hasattr(module, 'output_quantizers'):
            for idx, quantizer in enumerate(module.output_quantizers):
                if quantizer is not None and quantizer.is_initialized():
                    total_quantizers += 1
                    if modify_quantizer_to_power_of_2(quantizer, f"{name}.output_quantizer[{idx}]", verbose):
                        modified_quantizers += 1
        
        # 处理参数量化器（所有模块，包括非 QuantizationMixin）
        if hasattr(module, 'param_quantizers'):
            for param_name, quantizer in module.param_quantizers.items():
                if quantizer is not None and quantizer.is_initialized():
                    total_quantizers += 1
                    if modify_quantizer_to_power_of_2(quantizer, f"{name}.param_quantizer[{param_name}]", verbose):
                        modified_quantizers += 1
    
    print(f"\n总共处理 {total_quantizers} 个量化器")
    print(f"成功修改 {modified_quantizers} 个量化器为 Power-of-2 scale")
    print("="*80 + "\n")
    
    return {
        'total': total_quantizers,
        'modified': modified_quantizers
    }


def modify_quantizer_to_power_of_2(quantizer, quantizer_name: str, verbose: bool = True) -> bool:
    """
    修改单个量化器的scale为2的幂次方
    
    Args:
        quantizer: 量化器对象
        quantizer_name: 量化器名称（用于打印）
        verbose: 是否打印详细信息
        
    Returns:
        是否成功修改
    """
    try:
        # 获取原始参数（在任何修改之前）
        old_min = quantizer.get_min()
        old_max = quantizer.get_max()
        old_scale = quantizer.get_scale()
        old_offset = quantizer.get_offset()
        
        qmin = quantizer.qmin
        qmax = quantizer.qmax
        symmetric = quantizer.symmetric
        
        # 确保量化器使用scale-offset参数化
        # 这样我们可以直接修改scale参数
        quantizer._reparametrize_to_scale_offset()
        
        # 如果是张量，需要对每个元素分别处理
        if isinstance(old_scale, torch.Tensor):
            new_scale = torch.zeros_like(old_scale)
            
            # 展平处理
            old_scale_flat = old_scale.flatten()
            new_scale_flat = new_scale.flatten()
            
            for i in range(old_scale_flat.numel()):
                scale_val = old_scale_flat[i].item()
                # 找到最接近的2的幂次方scale
                new_scale_val, n = find_closest_power_of_2_scale(scale_val)
                new_scale_flat[i] = new_scale_val
            
            # 重塑回原始形状
            new_scale = new_scale_flat.view(old_scale.shape)
            
            if verbose:
                print(f"\n修改 {quantizer_name}:")
                print(f"  原始 scale 范围: [{old_scale.min().item():.8f}, {old_scale.max().item():.8f}]")
                print(f"  原始 min 范围: [{old_min.min().item():.6f}, {old_min.max().item():.6f}]")
                print(f"  原始 max 范围: [{old_max.min().item():.6f}, {old_max.max().item():.6f}]")
                if old_offset is not None:
                    print(f"  原始 offset 范围: [{old_offset.min().item():.2f}, {old_offset.max().item():.2f}]")
                print(f"  新的 scale 范围: [{new_scale.min().item():.8f}, {new_scale.max().item():.8f}]")
        else:
            # 标量情况
            # 找到最接近的2的幂次方scale
            new_scale_val, n = find_closest_power_of_2_scale(old_scale.item())
            new_scale = torch.tensor(new_scale_val, dtype=old_scale.dtype)
            
            if verbose:
                print(f"\n修改 {quantizer_name}:")
                print(f"  原始 scale: {old_scale.item():.8f}")
                print(f"  原始范围: [{old_min.item():.6f}, {old_max.item():.6f}]")
                if old_offset is not None:
                    print(f"  原始 offset: {old_offset.item():.2f}")
                print(f"  新的 scale: {new_scale_val:.8f} (1/2^{n})")
        
        # ！！！正确的做法！！！
        # 1. 保持浮点域min/max不变（校准得到的数据范围）
        # 2. 修改scale为power-of-2
        # 3. 对称量化（权重）：offset保持不变，重新计算qmin/qmax
        # 4. 非对称量化（feature）：重新计算offset、qmin和qmax
        
        if isinstance(new_scale, torch.Tensor):
            # Per-channel情况
            if symmetric or old_offset is None:
                # 对称量化：offset保持不变，Qmin=-Qmax
                if old_offset is not None:
                    new_qmin_float = old_min / new_scale - old_offset
                    new_qmax_float = old_max / new_scale - old_offset
                else:
                    new_qmin_float = old_min / new_scale
                    new_qmax_float = old_max / new_scale
                
                # 取绝对值较大的一个，保证对称，使用ceil确保覆盖
                max_abs = torch.max(torch.abs(new_qmin_float).max(), torch.abs(new_qmax_float).max())
                new_qmax_val = int(torch.ceil(max_abs).item())
                new_qmin_val = -new_qmax_val  # Qmin = -Qmax
                new_offset = old_offset  # 保持不变
            else:
                # 非对称量化：Qmin保持不变（通常为0），只修改Qmax
                orig_qmin = qmin
                new_qmin_val = orig_qmin  # Qmin不动
                
                # 关键：使用floor确保覆盖min端（向负方向扩展）
                # min_float = scale * (qmin + offset)
                # offset = min_float / scale - qmin
                new_offset_float = old_min / new_scale - orig_qmin
                new_offset = torch.floor(new_offset_float)  # floor确保min端不被截断
                
                # 计算qmax，使用ceil确保覆盖max端（向正方向扩展）
                # max_float = scale * (qmax + offset)
                # qmax = max_float / scale - offset
                new_qmax_float = old_max / new_scale - new_offset
                new_qmax_val = int(torch.ceil(new_qmax_float.max()).item())
        else:
            # Scalar情况
            old_min_val = old_min.item() if hasattr(old_min, 'item') else old_min
            old_max_val = old_max.item() if hasattr(old_max, 'item') else old_max
            new_scale_val = new_scale.item() if hasattr(new_scale, 'item') else new_scale
            
            if symmetric or old_offset is None:
                # 对称量化：offset保持不变，Qmin=-Qmax
                if old_offset is not None:
                    old_offset_val = old_offset.item() if hasattr(old_offset, 'item') else old_offset
                    new_qmin_float = old_min_val / new_scale_val - old_offset_val
                    new_qmax_float = old_max_val / new_scale_val - old_offset_val
                else:
                    new_qmin_float = old_min_val / new_scale_val
                    new_qmax_float = old_max_val / new_scale_val
                
                # 取绝对值较大的一个，保证对称，使用ceil确保覆盖
                import math
                max_abs = max(abs(new_qmin_float), abs(new_qmax_float))
                new_qmax_val = int(math.ceil(max_abs))
                new_qmin_val = -new_qmax_val  # Qmin = -Qmax
                new_offset = old_offset  # 保持不变
            else:
                # 非对称量化：Qmin保持不变（通常为0），只修改Qmax
                orig_qmin = qmin
                new_qmin_val = orig_qmin  # Qmin不动
                
                # 关键：使用floor确保覆盖min端（向负方向扩展）
                # min_float = scale * (qmin + offset)
                # offset = min_float / scale - qmin
                import math
                new_offset_val = math.floor(old_min_val / new_scale_val - orig_qmin)
                new_offset = torch.tensor(new_offset_val, dtype=old_scale.dtype)
                
                # 计算qmax，使用ceil确保覆盖max端（向正方向扩展）
                # max_float = scale * (qmax + offset)
                # qmax = max_float / scale - offset
                new_qmax_float = old_max_val / new_scale_val - new_offset_val
                new_qmax_val = int(math.ceil(new_qmax_float))
        
        # 修改scale、qmin、qmax和offset
        with torch.no_grad():
            quantizer.scale.copy_(new_scale)
            quantizer.qmin = new_qmin_val
            quantizer.qmax = new_qmax_val
            if new_offset is not None and quantizer.offset is not None:
                quantizer.offset.copy_(new_offset)
        
        if verbose:
            # 验证修改后的参数
            verify_scale = quantizer.get_scale()
            verify_min = quantizer.get_min()
            verify_max = quantizer.get_max()
            verify_offset = quantizer.get_offset()
            verify_qmin = quantizer.qmin
            verify_qmax = quantizer.qmax
            
            print(f"  新的Qmin: {verify_qmin}, Qmax: {verify_qmax}")
            
            if isinstance(verify_scale, torch.Tensor):
                all_po2 = all(verify_power_of_2_scale(s.item())[0] for s in verify_scale.flatten())
                print(f"  验证: {'✓ 所有scale都是2的幂次方' if all_po2 else '✗ 存在非2的幂次方scale'}")
                print(f"  新的Scale范围: [{verify_scale.min().item():.8f}, {verify_scale.max().item():.8f}]")
                print(f"  新的Min范围: [{verify_min.min().item():.6f}, {verify_min.max().item():.6f}]")
                print(f"  新的Max范围: [{verify_max.min().item():.6f}, {verify_max.max().item():.6f}]")
                if verify_offset is not None:
                    print(f"  新的Offset范围: [{verify_offset.min().item():.2f}, {verify_offset.max().item():.2f}]")
            else:
                is_po2, n = verify_power_of_2_scale(verify_scale.item())
                print(f"  验证: {'✓ scale是2的幂次方' if is_po2 else '✗ scale不是2的幂次方'}")
                if is_po2:
                    print(f"  新Scale = 1/2^{n} = {verify_scale.item():.8f}")
                print(f"  新的浮点范围: [{verify_min.item():.6f}, {verify_max.item():.6f}]")
                if verify_offset is not None:
                    print(f"  新的Offset: {verify_offset.item():.2f}")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"\n警告: 修改 {quantizer_name} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
        return False


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


def verify_model_power_of_2(model):
    """
    验证模型中所有量化器的scale是否都是2的幂次方
    
    Args:
        model: 量化模型
    """
    print("\n" + "="*80)
    print("验证 Power-of-2 量化")
    print("="*80)
    
    total_quantizers = 0
    power_of_2_quantizers = 0
    
    for name, module in model.named_modules():
        quantizers_to_check = []
        
        # 检查输入量化器（只在 QuantizationMixin 模块中）
        if isinstance(module, QuantizationMixin) and hasattr(module, 'input_quantizers'):
            for idx, q in enumerate(module.input_quantizers):
                if q is not None and q.is_initialized():
                    quantizers_to_check.append((q, f"{name}.input_quantizer[{idx}]"))
        
        # 检查输出量化器（只在 QuantizationMixin 模块中）
        if isinstance(module, QuantizationMixin) and hasattr(module, 'output_quantizers'):
            for idx, q in enumerate(module.output_quantizers):
                if q is not None and q.is_initialized():
                    quantizers_to_check.append((q, f"{name}.output_quantizer[{idx}]"))
        
        # 检查参数量化器（所有模块，包括非 QuantizationMixin）
        if hasattr(module, 'param_quantizers'):
            for param_name, q in module.param_quantizers.items():
                if q is not None and q.is_initialized():
                    quantizers_to_check.append((q, f"{name}.param_quantizer[{param_name}]"))
        
        # 处理收集到的量化器
        if quantizers_to_check:
            
            for quantizer, qname in quantizers_to_check:
                total_quantizers += 1
                scale = quantizer.get_scale()
                
                if isinstance(scale, torch.Tensor):
                    # 检查张量中的所有元素
                    all_power_of_2 = True
                    for s in scale.flatten():
                        is_po2, n = verify_power_of_2_scale(s.item())
                        if not is_po2:
                            all_power_of_2 = False
                            break
                    
                    if all_power_of_2:
                        power_of_2_quantizers += 1
                        print(f"✓ {qname}: 所有scale都是2的幂次方")
                    else:
                        print(f"✗ {qname}: 存在非2的幂次方scale")
                else:
                    is_po2, n = verify_power_of_2_scale(scale.item())
                    if is_po2:
                        power_of_2_quantizers += 1
                        print(f"✓ {qname}: scale = 1/2^{n}")
                    else:
                        print(f"✗ {qname}: scale = {scale.item():.8f} (非2的幂次方)")
    
    print(f"\n验证结果: {power_of_2_quantizers}/{total_quantizers} 个量化器使用 Power-of-2 scale")
    print("="*80 + "\n")
    
    return power_of_2_quantizers == total_quantizers

