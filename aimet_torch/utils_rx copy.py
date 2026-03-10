"""
AIMET 量化工具函数库
专注于量化相关的工具函数：Percentile 校准、Power-of-2 量化、量化器分析等
"""
from typing import Dict, Any


# ============================================================================
# Percentile 校准相关函数
# ============================================================================

def setup_percentile_calibration(sim_model, percentile: float = 99.0, num_bins: int = 2048, verbose: bool = True, 
                                only_params: bool = False) -> int:
    """
    将所有量化器的校准方法从 MinMax 改为 Percentile
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        percentile: 百分位数 (例如 99.0 表示排除最小1%和最大1%的极端值)
        num_bins: 直方图的 bin 数量
        verbose: 是否打印详细信息
        only_params: 如果为 True，只对参数量化器使用 Percentile，激活量化器保持 MinMax
    
    Returns:
        转换的量化器数量
    """
    from aimet_torch.v2.quantization.encoding_analyzer import PercentileEncodingAnalyzer
    
    if verbose:
        print(f"\n设置校准方法为 Percentile ({percentile}% 分位数)...")
        print(f"  这将排除最小 {(100-percentile)/2:.1f}% 和最大 {(100-percentile)/2:.1f}% 的极端数据")
        if only_params:
            print(f"  ⚠️  只对参数量化器使用 Percentile，激活量化器保持 MinMax")
    
    converted_count = 0
    
    # 转换 output_quantizers（如果 only_params=False）
    if not only_params:
        for name, module in sim_model.named_modules():
            if hasattr(module, 'output_quantizers'):
                for idx, quantizer in enumerate(module.output_quantizers):
                    if quantizer is not None and hasattr(quantizer, 'encoding_analyzer'):
                        original_shape = quantizer.encoding_analyzer.observer.shape
                        quantizer.encoding_analyzer = PercentileEncodingAnalyzer(
                            shape=original_shape,
                            num_bins=num_bins,
                            percentile=percentile
                        )
                        converted_count += 1
    
    # 转换 param_quantizers（总是转换）
    for name, module in sim_model.named_modules():
        # 转换 param_quantizers
        if hasattr(module, 'param_quantizers'):
            for param_name, quantizer in module.param_quantizers.items():
                if quantizer is not None and hasattr(quantizer, 'encoding_analyzer'):
                    original_shape = quantizer.encoding_analyzer.observer.shape
                    quantizer.encoding_analyzer = PercentileEncodingAnalyzer(
                        shape=original_shape,
                        num_bins=num_bins,
                        percentile=percentile
                    )
                    converted_count += 1
    
    if verbose:
        print(f"✅ 已将 {converted_count} 个量化器的校准方法改为 Percentile")
    
    return converted_count


def verify_percentile_calibration(sim_model, verbose: bool = True) -> Dict[str, int]:
    """
    验证 Percentile 校准设置是否成功
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        verbose: 是否打印详细信息
    
    Returns:
        统计字典 {'minmax': count, 'percentile': count, 'other': count}
    """
    minmax_count = 0
    percentile_count = 0
    other_count = 0
    
    for name, module in sim_model.named_modules():
        if hasattr(module, 'output_quantizers'):
            for quantizer in module.output_quantizers:
                if quantizer is not None and hasattr(quantizer, 'encoding_analyzer'):
                    analyzer_type = type(quantizer.encoding_analyzer).__name__
                    if 'MinMax' in analyzer_type:
                        minmax_count += 1
                    elif 'Percentile' in analyzer_type:
                        percentile_count += 1
                    else:
                        other_count += 1
        
        if hasattr(module, 'param_quantizers'):
            for quantizer in module.param_quantizers.values():
                if quantizer is not None and hasattr(quantizer, 'encoding_analyzer'):
                    analyzer_type = type(quantizer.encoding_analyzer).__name__
                    if 'MinMax' in analyzer_type:
                        minmax_count += 1
                    elif 'Percentile' in analyzer_type:
                        percentile_count += 1
                    else:
                        other_count += 1
    
    if verbose:
        print("\n验证校准方法:")
        print(f"  MinMaxEncodingAnalyzer:     {minmax_count} 个")
        print(f"  PercentileEncodingAnalyzer: {percentile_count} 个 ✅")
        print(f"  其他:                       {other_count} 个")
        
        if percentile_count > 0:
            print(f"\n✅ 替换成功！将使用 Percentile 校准方法")
        else:
            print(f"\n⚠️ 警告：未找到 PercentileEncodingAnalyzer，仍使用默认 MinMax 方法")
    
    return {
        'minmax': minmax_count,
        'percentile': percentile_count,
        'other': other_count
    }


# ============================================================================
# Power-of-2 量化相关函数
# ============================================================================

def apply_power_of_2_workflow(sim_model, verbose: bool = True) -> Dict[str, Any]:
    """
    完整的 Power-of-2 量化工作流
    
    包括：
    1. 打印校准后的量化器信息
    2. 收集修改前的量化器信息
    3. 应用 power-of-2 量化
    4. 验证修改结果
    5. 对比修改前后的差异
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        verbose: 是否打印详细信息
    
    Returns:
        结果字典，包含统计信息和对比结果
    """
    from aimet_torch.power_of_2_quantization import (
        print_quantizer_info,
        apply_power_of_2_quantization,
        verify_model_power_of_2
    )
    from aimet_torch.compare_before_after_po2 import collect_quantizer_info, compare_quantizers
    
    results = {}
    
    # 1. 打印校准后的量化器信息
    if verbose:
        print("\n" + "="*80)
        print("步骤 1: 校准后的量化器信息")
        print("="*80)
    
    original_info = print_quantizer_info(sim_model, "校准后的原始量化参数")
    results['original_info'] = original_info
    
    # 2. 收集修改前的量化器信息
    if verbose:
        print("\n收集修改前的量化器参数...")
    before_po2_info = collect_quantizer_info(sim_model)
    if verbose:
        print(f"✅ 收集了 {len(before_po2_info)} 个量化器的信息")
    results['before_count'] = len(before_po2_info)
    results['before_info'] = before_po2_info
    
    # 3. 应用 Power-of-2 量化
    if verbose:
        print("\n" + "="*80)
        print("步骤 2: 将 scale 修改为 2 的幂次方 (1/2^n)")
        print("="*80)
    
    stats = apply_power_of_2_quantization(sim_model, verbose=verbose)
    results['stats'] = stats
    
    # 4. 收集修改后的量化器信息
    if verbose:
        print("\n收集修改后的量化器参数...")
    after_po2_info = collect_quantizer_info(sim_model)
    if verbose:
        print(f"✅ 收集了 {len(after_po2_info)} 个量化器的信息")
    results['after_count'] = len(after_po2_info)
    results['after_info'] = after_po2_info
    
    # 5. 验证修改结果
    if verbose:
        print("\n" + "="*80)
        print("步骤 3: 验证修改后的量化参数")
        print("="*80)
    
    modified_info = print_quantizer_info(sim_model, "修改后的量化参数")
    results['modified_info'] = modified_info
    
    # 6. 验证所有 scale 是否都是 2 的幂次方
    all_power_of_2 = verify_model_power_of_2(sim_model)
    results['all_power_of_2'] = all_power_of_2
    
    if verbose:
        if all_power_of_2:
            print("\n✅ 所有量化器的 scale 已成功修改为 2 的幂次方!")
        else:
            print("\n⚠️ 警告: 部分量化器的 scale 未能修改为 2 的幂次方")
    
    # 7. 对比修改前后的参数
    if verbose:
        print("\n" + "="*80)
        print("步骤 4: 对比修改前后的参数")
        print("="*80)
    
    issues = compare_quantizers(before_po2_info, after_po2_info)
    results['issues'] = issues
    
    return results


# ============================================================================
# 量化器参数冻结相关函数
# ============================================================================

def freeze_quantizer_parameters(sim_model, verbose: bool = True) -> Dict[str, Any]:
    """
    冻结量化器参数（scale、offset 等），仅训练模型权重
    
    在 QAT 微调时，通常需要冻结量化器的参数（如 scale、offset），
    只训练模型的权重参数（weights、bias）
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        verbose: 是否打印详细信息
    
    Returns:
        结果字典，包含：
        - trainable_params: 可训练参数名称列表
        - frozen_params: 冻结参数名称列表
        - trainable_count: 可训练参数数量
        - frozen_count: 冻结参数数量
    """
    trainable_params = []
    frozen_params = []
    
    for name, param in sim_model.named_parameters():
        if param.requires_grad:
            should_freeze = False
            # 检查是否是量化器参数
            if 'quantizer' in name and any(kw in name for kw in ['min', 'max', 'scale', 'offset', 'encoding']):
                should_freeze = True
            
            if should_freeze:
                param.requires_grad = False
                frozen_params.append(name)
            else:
                trainable_params.append(name)
    
    if verbose:
        print(f"冻结量化器minmax参数:   {len(frozen_params)} 个")
    
    return {
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'trainable_count': len(trainable_params),
        'frozen_count': len(frozen_params)
    }


def set_train_mode_freeze_bn(model, verbose: bool = False) -> int:
    """
    设置模型为训练模式，但保持 BatchNorm 层为评估模式
    
    在 QAT 微调时，通常需要保持 BatchNorm 层的统计信息不变，
    避免量化后的统计信息与浮点模型差异过大
    
    Args:
        model: PyTorch 模型或 AIMET QuantizationSimModel.model
        verbose: 是否打印详细信息
    
    Returns:
        冻结的 BatchNorm 层数量
    """
    import torch.nn as nn
    
    # 设置整个模型为训练模式
    model.train()
    
    # 将所有 BatchNorm 层设置为评估模式
    bn_count = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            bn_count += 1
    
    if verbose:
        print(f"✅ 模型设置为训练模式，{bn_count} 个 BatchNorm 层保持评估模式")
    
    return bn_count

