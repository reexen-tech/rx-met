"""
AIMET 量化工具函数库
专注于量化相关的工具函数：Percentile 校准、Power-of-2 量化、量化器分析等
"""
from typing import Dict, Any, Iterable, Mapping, Set, Tuple
import json
from pathlib import Path


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
    # 注意：不转换 input_quantizers，因为它们可能接收特殊值（如常量、离散值）
    # 使用 Percentile 直方图可能导致 NaN 或索引越界
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


def setup_selective_percentile_calibration(
    sim_model, 
    percentile: float = 99.0, 
    num_bins: int = 2048, 
    exclude_modules: list = None,
    verbose: bool = True
) -> Dict[str, int]:
    """
    选择性地设置 Percentile 校准，可以排除特定模块
    
    适用场景：
    - GRU/LSTM 等循环结构对 Percentile 敏感，需要使用 MinMax
    - 其他层（Conv2d, Linear）使用 Percentile 效果更好
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        percentile: 百分位数
        num_bins: 直方图 bin 数量
        exclude_modules: 要排除的模块名称关键字列表 (例如 ['seq_t', 'gru', 'lstm'])
        verbose: 是否打印详细信息
    
    Returns:
        统计字典 {'converted': int, 'excluded': int}
    
    示例:
        # 对 GRU 相关模块使用 MinMax，其他使用 Percentile
        stats = setup_selective_percentile_calibration(
            sim.model,
            exclude_modules=['seq_t', 'gru'],  # 排除 GRU 相关模块
            verbose=True
        )
    """
    from aimet_torch.v2.quantization.encoding_analyzer import PercentileEncodingAnalyzer
    
    if exclude_modules is None:
        exclude_modules = []
    
    if verbose:
        print(f"\n设置选择性 Percentile 校准 ({percentile}% 分位数)...")
        print(f"  排除模块关键字: {exclude_modules if exclude_modules else '无 (全部使用 Percentile)'}")
    
    converted_count = 0
    excluded_quantizer_count = 0  # 统计被排除的量化器数量
    excluded_modules_list = []  # 记录被排除的顶层模块
    
    def should_exclude(module_name: str) -> bool:
        """检查模块名是否应该被排除"""
        for keyword in exclude_modules:
            if keyword.lower() in module_name.lower():
                return True
        return False
    
    # 遍历所有模块
    for name, module in sim_model.named_modules():
        # 检查是否应该排除
        if should_exclude(name):
            # 只记录顶层被排除的模块（避免输出太多子模块）
            if name and '.' not in name[name.find('seq_t'):].replace('seq_t', '', 1):
                # 这是一个 seq_t 顶层模块
                if name not in excluded_modules_list:
                    excluded_modules_list.append(name)
            
            # 统计被排除的量化器数量
            if hasattr(module, 'input_quantizers'):
                for quantizer in module.input_quantizers:
                    if quantizer is not None and hasattr(quantizer, 'encoding_analyzer'):
                        excluded_quantizer_count += 1
            if hasattr(module, 'output_quantizers'):
                for quantizer in module.output_quantizers:
                    if quantizer is not None and hasattr(quantizer, 'encoding_analyzer'):
                        excluded_quantizer_count += 1
            if hasattr(module, 'param_quantizers'):
                for quantizer in module.param_quantizers.values():
                    if quantizer is not None and hasattr(quantizer, 'encoding_analyzer'):
                        excluded_quantizer_count += 1
            continue
        
        # 转换 output_quantizers
        # 注意：不转换 input_quantizers，避免对特殊值（常量、离散值）使用 Percentile 导致错误
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
        print(f"\n✅ 已将 {converted_count} 个量化器改为 Percentile")
        print(f"⏭️  排除了 {excluded_quantizer_count} 个量化器（保持原始 Analyzer）")
        if excluded_modules_list:
            print(f"   排除的顶层模块: {', '.join(excluded_modules_list)}")
    
    return {
        'converted': converted_count,
        'excluded': excluded_quantizer_count,
        'excluded_modules': excluded_modules_list
    }


def verify_percentile_calibration(sim_model, verbose: bool = True) -> Dict[str, int]:
    """
    验证 Percentile 校准设置是否成功
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        verbose: 是否打印详细信息
    
    Returns:
        统计字典 {'minmax': count, 'percentile': count, 'other': count, 'other_types': dict}
    """
    minmax_count = 0
    percentile_count = 0
    other_count = 0
    other_types = {}  # 记录其他类型的 analyzer
    
    for name, module in sim_model.named_modules():
        # 检查 input_quantizers
        if hasattr(module, 'input_quantizers'):
            for quantizer in module.input_quantizers:
                if quantizer is not None and hasattr(quantizer, 'encoding_analyzer'):
                    analyzer_type = type(quantizer.encoding_analyzer).__name__
                    if 'MinMax' in analyzer_type:
                        minmax_count += 1
                    elif 'Percentile' in analyzer_type:
                        percentile_count += 1
                    else:
                        other_count += 1
                        # 记录具体类型
                        if analyzer_type not in other_types:
                            other_types[analyzer_type] = 0
                        other_types[analyzer_type] += 1
        
        # 检查 output_quantizers
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
                        # 记录具体类型
                        if analyzer_type not in other_types:
                            other_types[analyzer_type] = 0
                        other_types[analyzer_type] += 1
        
        # 检查 param_quantizers
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
                        # 记录具体类型
                        if analyzer_type not in other_types:
                            other_types[analyzer_type] = 0
                        other_types[analyzer_type] += 1
    
    if verbose:
        print("\n验证校准方法:")
        print(f"  MinMaxEncodingAnalyzer:     {minmax_count} 个")
        print(f"  PercentileEncodingAnalyzer: {percentile_count} 个 ✅")
        print(f"  其他:                       {other_count} 个")
        
        # 显示其他类型的详细信息
        if other_types:
            print(f"\n  其他类型详情:")
            for analyzer_type, count in other_types.items():
                print(f"    - {analyzer_type}: {count} 个")
        
        if percentile_count > 0:
            print(f"\n✅ 替换成功！将使用 Percentile 校准方法")
        else:
            print(f"\n⚠️ 警告：未找到 PercentileEncodingAnalyzer，仍使用默认 MinMax 方法")
    
    return {
        'minmax': minmax_count,
        'percentile': percentile_count,
        'other': other_count,
        'other_types': other_types
    }


# ============================================================================
# Power-of-2 量化相关函数
# ============================================================================

def apply_power_of_2_workflow(sim_model, method: str = "cover_range", tolerance: float = 0.02, 
                              align_bias_scale: bool = False, verbose: bool = True,
                              bias_bitwidth: int = 32) -> Dict[str, Any]:
    """
    完整的 Power-of-2 量化工作流
    
    包括：
    1. 打印校准后的量化器信息
    2. 收集修改前的量化器信息
    3. 应用 power-of-2 量化
    4. 验证修改结果
    5. 对比修改前后的差异
    6. (可选) Bias Scale 对齐到 Sx * Sw
    
    Args:
        sim_model: Torch 侧传 ``QuantizationSimModel.model``；ONNX 侧传
            ``aimet_onnx.QuantizationSimModel``（含 ``qc_quantize_op_dict``）
        method: Power-of-2 选择策略
            - "round": 全部四舍五入
            - "cover_range": 智能策略（默认）
                - 如果 real_max - real_min 接近 2^n（相对误差 < tolerance）：使用四舍五入
                - 否则：增大 scale 使得覆盖原范围
        tolerance: 判断 real_range 是否接近 2^n 的相对容差（默认 2%）
            - 仅在 method="cover_range" 时生效
            - 例如：real_range=2.007884 与 2.0 的相对误差 0.39% < 2%，被视为 2^n
        align_bias_scale: 是否将卷积层的 bias scale 对齐到 Sx * Sw（默认 False）
            - True: 执行对齐，使得硬件可以直接计算 (Qx*Qw + Qb)
            - False: 跳过对齐
        verbose: 是否打印详细信息
        bias_bitwidth: 对齐后的有符号 bias 位宽；PyTorch Conv/Linear 与 ONNX 均使用，默认 32。不可表示时拒绝对齐
    
    Returns:
        结果字典，包含统计信息和对比结果
    """
    if hasattr(sim_model, "qc_quantize_op_dict"):
        from aimet_onnx.rx_ptq.po2 import (
            apply_power_of_2_workflow as apply_onnx_power_of_2_workflow,
        )

        return apply_onnx_power_of_2_workflow(
            sim_model,
            method=method,
            tolerance=tolerance,
            align_bias_scale=align_bias_scale,
            verbose=verbose,
            bias_bitwidth=bias_bitwidth,
        )

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
    
    original_info = print_quantizer_info(sim_model, "校准后的原始量化参数", verbose=verbose)
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
    
    stats = apply_power_of_2_quantization(sim_model, method=method, tolerance=tolerance, verbose=verbose)
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
    
    modified_info = print_quantizer_info(sim_model, "修改后的量化参数", verbose=verbose)
    results['modified_info'] = modified_info
    
    # 6. 验证所有 scale 是否都是 2 的幂次方
    all_power_of_2 = verify_model_power_of_2(sim_model,verbose=verbose)
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
    else:
        issues = []
    results['issues'] = issues
    
    # 8. (可选) Bias Scale 对齐到 Sx * Sw
    if align_bias_scale:
        if verbose:
            print("\n" + "="*80)
            print("步骤 5: Bias Scale 对齐（Sb = Sx * Sw）")
            print("="*80)
            print("📝 将卷积/全连接层的 bias scale 对齐到输入和权重的 scale 乘积")
            print("📝 目标：使得硬件可以直接计算 Sy*Qy = Sx*Sw*(Qx*Qw + Qb)\n")
        
        bias_alignment_stats = _align_conv_bias_scale(sim_model, verbose=verbose, bias_bitwidth=bias_bitwidth)
        results['bias_alignment'] = bias_alignment_stats

    from aimet_torch.native_recurrent import named_native_recurrent
    for name, module, op in named_native_recurrent(sim_model):
        if module.use_quantization:
            module.enable_pot2(method=method, tolerance=tolerance)
            key = f"{op.package}_pot2_count"
            results[key] = results.get(key, 0) + 1
            if verbose:
                print(f"  [{op.module_type.__name__}] {name}: POT2 enabled")

    return results


# ============================================================================
# 量化器参数冻结相关函数
# ============================================================================

def freeze_quantizer_parameters(sim_model, verbose: bool = True, freeze_bn_affine: bool = True) -> Dict[str, Any]:
    """
    冻结量化器参数（scale、offset 等），仅训练模型权重
    
    在 QAT 微调时，通常需要冻结量化器的参数（如 scale、offset），
    只训练模型的权重参数（weights、bias）
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        verbose: 是否打印详细信息
        freeze_bn_affine: 是否冻结 BatchNorm 的 affine 参数（weight、bias）
    
    Returns:
        结果字典，包含：
        - trainable_params: 可训练参数名称列表
        - frozen_params: 冻结参数名称列表
        - frozen_quantizer_params: 冻结的量化器参数列表
        - frozen_bn_params: 冻结的 BN affine 参数列表
        - trainable_count: 可训练参数数量
        - frozen_count: 冻结参数数量
    """
    trainable_params = []
    frozen_params = []
    frozen_quantizer_params = []
    frozen_bn_params = []
    
    # 常见 BN 模块名称模式
    bn_name_patterns = ['_bn', 'bn_', 'pre_bn', 'rnn2d_bn', 'batchnorm', 'batch_norm']
    
    for name, param in sim_model.named_parameters():
        if param.requires_grad:
            should_freeze = False
            freeze_reason = None
            
            # 检查是否是量化器参数
            if 'quantizer' in name and any(kw in name for kw in ['min', 'max', 'scale', 'offset', 'encoding']):
                should_freeze = True
                freeze_reason = 'quantizer'
            
            # 检查是否是 BatchNorm 的 affine 参数（weight 或 bias）
            if freeze_bn_affine and not should_freeze:
                # 检查参数名称是否包含 BN 模块名称模式
                name_lower = name.lower()
                is_bn_param = (any(pattern in name_lower for pattern in bn_name_patterns)
                               or 'bn' in name_lower.split('.'))
                # 检查是否是 weight 或 bias 参数
                is_affine_param = name.endswith('.weight') or name.endswith('.bias')
                
                if is_bn_param and is_affine_param:
                    should_freeze = True
                    freeze_reason = 'bn_affine'
            
            if should_freeze:
                param.requires_grad = False
                frozen_params.append(name)
                if freeze_reason == 'quantizer':
                    frozen_quantizer_params.append(name)
                elif freeze_reason == 'bn_affine':
                    frozen_bn_params.append(name)
            else:
                trainable_params.append(name)
    
    if verbose:
        print(f"冻结量化器参数:         {len(frozen_quantizer_params)} 个")
        if freeze_bn_affine:
            print(f"冻结 BN affine 参数:    {len(frozen_bn_params)} 个")
            if frozen_bn_params:
                print("  冻结的 BN 参数示例:")
                for param_name in frozen_bn_params[:5]:  # 只显示前5个
                    print(f"    - {param_name}")
        print(f"总冻结参数:             {len(frozen_params)} 个")
        print(f"可训练参数:             {len(trainable_params)} 个")
    
    return {
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'frozen_quantizer_params': frozen_quantizer_params,
        'frozen_bn_params': frozen_bn_params,
        'trainable_count': len(trainable_params),
        'frozen_count': len(frozen_params)
    }


def set_train_mode_freeze_bn(model, verbose: bool = False) -> int:
    """
    设置模型为训练模式，但保持 BatchNorm 层为评估模式
    
    在 QAT 微调时，通常需要保持 BatchNorm 层的统计信息不变，
    避免量化后的统计信息与浮点模型差异过大
    
    支持的 BatchNorm 类型：
    - nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d (标准 PyTorch BN)
    - QuantizableBatchNorm2d (自定义可量化 BN)
    - prepare_model 展开后的 BN（通过模块名称识别，如 pre_bn, rnn2d_bn）
    
    Args:
        model: PyTorch 模型或 AIMET QuantizationSimModel.model
        verbose: 是否打印详细信息
    
    Returns:
        冻结的 BatchNorm 层数量
    """
    import torch.nn as nn
    
    # 动态导入自定义 BatchNorm（避免循环导入）
    try:
        from aimet_torch.quantizable_batchnorm import QuantizableBatchNorm2d
        custom_bn_types = (QuantizableBatchNorm2d,)
    except ImportError:
        custom_bn_types = ()
    
    # 设置整个模型为训练模式
    model.train()
    
    # 策略 1: 通过类型识别标准 BN（适用于未展开的 BN）
    bn_count = 0
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d) + custom_bn_types
    frozen_modules = set()  # 记录已冻结的模块，避免重复计数
    
    for module in model.modules():
        if isinstance(module, bn_types) and module not in frozen_modules:
            module.eval()
            frozen_modules.add(module)
            bn_count += 1
    
    # 策略 2: 通过名称识别展开后的 BN（适用于 prepare_model 后）
    # 常见 BN 模块名称模式（以 _bn 或 bn_ 结尾/开头的模块）
    bn_name_patterns = ['_bn', 'bn_', 'pre_bn', 'rnn2d_bn', 'batchnorm']
    
    for name, module in model.named_modules():
        # 检查模块名称是否匹配 BN 模式
        name_lower = name.lower()
        is_bn_by_name = (any(pattern in name_lower for pattern in bn_name_patterns)
                               or 'bn' in name_lower.split('.'))
        
        if is_bn_by_name and module not in frozen_modules:
            # 将该模块及其所有子模块设置为 eval 模式
            module.eval()
            frozen_modules.add(module)
            bn_count += 1
            
            # 如果有子模块，也冻结它们
            for child in module.modules():
                if child != module and child not in frozen_modules:
                    child.eval()
                    frozen_modules.add(child)
    
    if verbose:
        print(f"✅ 模型设置为训练模式，{bn_count} 个 BatchNorm 层保持评估模式")
    
    return bn_count


# ============================================================================
# 混合精度位宽配置相关函数
# ============================================================================

def load_mixed_precision_config(config_file: str) -> Dict[str, Any]:
    """
    加载混合精度位宽配置文件
    
    Args:
        config_file: JSON 配置文件路径
    
    Returns:
        配置字典
    """
    from aimet_common.mixed_precision import load_mixed_precision_config as load_config
    return load_config(config_file)


_LAYER_TYPE_CONFIG_JSON_ALIASES = frozenset(
    {
        "QuantizedSign",
    }
)


def _is_layer_type_config_comment(key: str) -> bool:
    from aimet_common.mixed_precision import is_config_comment
    return is_config_comment(key)


def known_layer_type_config_keys() -> Set[str]:
    """Registered ``Quantized*`` class names that JSON 2 may list, plus aliases."""
    names: Set[str] = set(_LAYER_TYPE_CONFIG_JSON_ALIASES)
    try:
        import aimet_torch.v2.nn.true_quant  # noqa: F401
        import aimet_torch.v2.nn.modules.custom  # noqa: F401
        from aimet_torch.quantizable_batchnorm import QuantizableBatchNorm2d  # noqa: F401
        from aimet_torch.v2.nn import QuantizationMixin

        names.update(qcls.__name__ for qcls in QuantizationMixin.cls_to_qcls.values())
    except ImportError:
        pass
    return names


def classify_unused_layer_type_config_keys(
    layer_type_config: Mapping[str, Any],
    present_layer_types: Iterable[str],
) -> Tuple[list, list]:
    """Split JSON 2 type keys that are not in the model into (ignored, typos).

    * Present in ``sim.model``: applied later, not returned.
    * Registered ``Quantized*`` (or a documented alias) but absent here: ignore.
    * Anything else: treat as a misspelled type name.
    """
    present = set(present_layer_types)
    known = known_layer_type_config_keys()
    ignored = []
    unknown = []
    for layer_type in layer_type_config:
        if _is_layer_type_config_comment(layer_type) or layer_type in present:
            continue
        if layer_type in known:
            ignored.append(layer_type)
        else:
            unknown.append(layer_type)
    return sorted(ignored), sorted(unknown)



def _set_weight_granularity(module, per_channel, name):
    import copy
    import torch.nn as nn
    from aimet_torch.v2.quantization.affine import QuantizeDequantize

    quantizers = getattr(module, 'param_quantizers', {})
    quantizer = quantizers['weight'] if 'weight' in quantizers else None
    if quantizer is None:
        return
    shape = ()
    if per_channel:
        transposed = isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d))
        supported = isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)) or transposed
        if not supported or (transposed and module.groups != 1):
            raise ValueError(f'{name}: unsupported per-channel weight configuration')
        axis = 1 if transposed else 0
        shape = tuple(size if index == axis else 1 for index, size in enumerate(module.weight.shape))
    if quantizer.shape == shape:
        return
    if type(quantizer) is not QuantizeDequantize or quantizer.block_size is not None:
        raise ValueError(f'{name}: unsupported weight quantizer granularity change')
    if quantizer.is_initialized():
        raise ValueError(f'{name}: configure per_channel_quantization before calibration')
    analyzer = copy.deepcopy(quantizer.encoding_analyzer)
    analyzer.observer.shape = shape
    analyzer.observer.reset_stats()
    parameter = next(quantizer.parameters())
    replacement = QuantizeDequantize(
        shape, qmin=quantizer.qmin, qmax=quantizer.qmax, symmetric=quantizer.symmetric,
        encoding_analyzer=analyzer, zero_point_shift=quantizer.zero_point_shift,
    ).to(device=parameter.device, dtype=parameter.dtype)
    replacement.train(quantizer.training)
    replacement.requires_grad_(parameter.requires_grad)
    module.param_quantizers['weight'] = replacement


def apply_mixed_precision_bitwidth(sim_model, config_file: str, verbose: bool = True) -> Dict[str, int]:
    """
    根据 JSON 配置文件应用混合精度位宽设置
    
    支持的配置项：
    - weight_bitwidth: 权重位宽
    - bias_bitwidth: 偏置位宽
    - input_bitwidth: 输入位宽
    - input_bitwidths: 按输入下标指定输入位宽，例如 [8, 16]
    - output_bitwidth: 输出位宽
    - param_bitwidth: 通用参数位宽，作用于所有非 weight/bias 的 param_quantizers 键
                      （如 Snake2d 的 alpha、LayerNorm 的 gamma/beta 等）
    - weight_symmetric: 权重对称量化 (True/False)
    - output_symmetric: 输出对称量化 (True/False)
    - input_symmetric: 输入对称量化 (True/False)
    - param_symmetric: 通用参数对称量化 (True/False)，与 param_bitwidth 配合使用
    - per_channel_quantization: 逐通道量化 (True/False)
    - disable_quantization: 禁用量化 (True/False)

    Torch 侧扫 ``sim.model`` 的 ``Quantized*`` 模块。JSON 2 是通用类型清单：
    当前模型没有、但已注册的 ``Quantized*`` 直接忽略（与 ONNX 一致）；未注册
    且模型里也没有的 key 视为拼写错误并报错。ONNX 侧传入带
    ``qc_quantize_op_dict`` 的 sim 时复用同一份 JSON，对不上的类型和
    ``GRU_config`` 同样忽略。
    
    支持模式匹配（在 layer_name_config 中）：
    - 使用 "*" 通配符匹配层名称
    - 例如：
      "*.seq_t.cells.0.weight_ih": {...}  匹配所有GRU的weight_ih层
      "enc_seqs.*.seq_t.*": {...}         匹配所有encoder的GRU内部层
    
    配置优先级：
    1. layer_name_config - 精确匹配（优先级最高）
    2. layer_type_config（按类型匹配）
    3. layer_name_config - 模式匹配
    4. default_bitwidth（默认值）
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        config_file: JSON 配置文件路径
        verbose: 是否打印详细信息
    
    Returns:
        统计字典，包含各类型层的设置数量
    """
    if hasattr(sim_model, "qc_quantize_op_dict"):
        from aimet_onnx.rx_ptq.mixed_precision import (
            apply_onnx_mixed_precision_bitwidth,
        )

        return apply_onnx_mixed_precision_bitwidth(
            sim_model, config_file, verbose=verbose
        )

    from aimet_common.mixed_precision import (
        QuantizerConfigPlan, default_layer_config, is_config_comment, select_layer_config,
    )
    
    # 加载配置
    config = load_mixed_precision_config(config_file)
    
    default_bitwidth = config.get('default_bitwidth', {})
    default_config = config.get('default_config', {})  # 读取 default_config 字段
    layer_type_config = config.get('layer_type_config', {})
    layer_name_config = config.get('layer_name_config', {})
    
    # 过滤掉注释字段
    layer_name_config = {k: v for k, v in layer_name_config.items() 
                        if not is_config_comment(k)}
    
    # 检查默认配置是否要求禁用量化
    default_disable_quantization = default_config.get('disable_quantization', False)
    
    if verbose:
        print("\n" + "="*70)
        print("🔧 应用混合精度位宽配置")
        print("="*70)
        print(f"📄 配置文件: {config_file}")
        print(f"📝 默认位宽: weight={default_bitwidth.get('weight', 8)}-bit, output={default_bitwidth.get('output', 8)}-bit")
        if default_disable_quantization:
            print(f"📝 默认策略: 禁用所有未匹配层的量化")
        print(f"📝 类型配置数量: {len(layer_type_config)} 个层类型")
        print(f"📝 名称配置数量: {len(layer_name_config)} 个具体层")
        print("="*70)
    
    present_layer_types = {type(module).__name__ for _, module in sim_model.named_modules()}
    ignored_layer_types, unknown_layer_types = classify_unused_layer_type_config_keys(
        layer_type_config, present_layer_types
    )
    if unknown_layer_types:
        unknown_msg = ", ".join(repr(name) for name in unknown_layer_types)
        raise ValueError(
            "Unknown layer_type_config key(s): "
            f"{unknown_msg}. "
            "Not a registered Quantized* type and not present in sim.model; "
            "check spelling. Types that exist but are absent from this model are ignored."
        )
    if ignored_layer_types and verbose:
        print(f"  忽略当前模型没有的类型: {ignored_layer_types}")

    # 统计信息
    stats = {
        'conv_count': 0,
        'linear_count': 0,
        'activation_count': 0,
        'add_count': 0,
        'multiply_count': 0,
        'subtract_count': 0,
        'disabled_count': 0,
        'name_matched_count': 0,
        'type_matched_count': 0,
        'default_count': 0
    }

    from aimet_torch.native_recurrent import named_native_recurrent
    recurrent_modules = list(named_native_recurrent(sim_model))
    recurrent_names = {name for name, _, _ in recurrent_modules}

    domain_plan = QuantizerConfigPlan()
    for name, module in sim_model.named_modules():
        if any(name == native or name.startswith(native + '.') for native in recurrent_names):
            continue
        settings, _, _ = select_layer_config(name, type(module).__name__, layer_type_config, layer_name_config)
        if settings is None and default_disable_quantization:
            continue
        settings = settings or {}
        if settings.get('disable_quantization', False):
            continue
        bindings = [(role if role in ('weight', 'bias') else 'param', None, q)
                    for role, q in getattr(module, 'param_quantizers', {}).items()]
        bindings += [(role, index, q) for role in ('input', 'output')
                     for index, q in enumerate(getattr(module, f'{role}_quantizers', []))]
        for role, index, quantizer in bindings:
            domain_plan.request(quantizer, 'signed', default_config.get('signed'),
                                f'default_config.{name}.{role}', fallback=True)
            signs = settings.get('input_signeds') if role == 'input' else None
            signed = ((signs[index] if index < len(signs) else None) if signs is not None
                      else settings.get(f'{role}_signed'))
            domain_plan.request(quantizer, 'signed', signed, f'{name}.{role}[{index}]')
    for quantizer, settings in domain_plan.entries():
        if 'signed' in settings and hasattr(quantizer, 'signed'):
            quantizer.signed = settings['signed']

    # 遍历所有模块
    for name, module in sim_model.named_modules():
        module_type = type(module).__name__

        # Native operators parse their own stage configuration.
        if name in recurrent_names:
            continue

        # 确定使用哪个配置
        # 优先级：精确名称 > 类型 > 模式匹配 > 默认
        #   类型 > 模式：避免宽泛通配符（如 *.act*）意外覆盖子模块的类型配置；
        #   模式匹配保留在类型之后，专门用于无类型配置的父 Module（如 Snake2d 包装层的 alpha）。
        layer_config = None
        match_type = None
        matched_pattern = None

        layer_config, match_type, matched_pattern = select_layer_config(
            name, module_type, layer_type_config, layer_name_config
        )
        if match_type in ('name', 'pattern'):
            stats['name_matched_count'] += 1
        elif match_type == 'type':
            stats['type_matched_count'] += 1

        # 4. 对于未匹配的模块，检查是否需要禁用量化
        if layer_config is None:
            has_quantizers = (
                (hasattr(module, 'param_quantizers') and module.param_quantizers) or
                (hasattr(module, 'output_quantizers') and len(module.output_quantizers) > 0 and any(q is not None for q in module.output_quantizers)) or
                (hasattr(module, 'input_quantizers') and len(module.input_quantizers) > 0 and any(q is not None for q in module.input_quantizers))
            )
            if default_disable_quantization and has_quantizers:
                _disable_quantization(module, name, verbose=False)
                stats['disabled_count'] += 1
                continue
            elif has_quantizers:
                layer_config = default_layer_config(default_bitwidth)
                match_type = 'default'
                stats['default_count'] += 1

        # 应用配置
        if layer_config:
            if layer_config.get('disable_quantization', False):
                _disable_quantization(module, name, verbose)
                stats['disabled_count'] += 1
                continue

            if 'per_channel_quantization' in layer_config:
                _set_weight_granularity(module, layer_config['per_channel_quantization'], name)

            # 设置权重量化参数
            if hasattr(module, 'param_quantizers') and 'weight' in module.param_quantizers:
                weight_quantizer = module.param_quantizers['weight']
                if weight_quantizer is not None:
                    changes = []
                    weight_bw = layer_config.get('weight_bitwidth')
                    if weight_bw:
                        weight_quantizer.bitwidth = weight_bw
                        changes.append(f"{weight_bw}-bit")
                    weight_sym = layer_config.get('weight_symmetric')
                    if weight_sym is not None:
                        weight_quantizer.symmetric = weight_sym
                        changes.append(f"{'sym' if weight_sym else 'asym'}")
                    if changes and verbose:
                        label = f"pattern: {matched_pattern}" if match_type == 'pattern' else match_type
                        print(f"  ✅ [{label}] {name}.weight: {', '.join(changes)}")
                    if 'Conv' in module_type:
                        stats['conv_count'] += 1
                    elif 'Linear' in module_type:
                        stats['linear_count'] += 1
                    elif 'Add' in module_type:
                        stats['add_count'] += 1
                    elif 'Multiply' in module_type:
                        stats['multiply_count'] += 1
                    elif 'Subtract' in module_type:
                        stats['subtract_count'] += 1
            
            # 设置 bias 量化参数（新增）
            if hasattr(module, 'param_quantizers') and 'bias' in module.param_quantizers:
                bias_quantizer = module.param_quantizers['bias']
                if bias_quantizer is not None:
                    changes = []
                    
                    # 位宽
                    bias_bw = layer_config.get('bias_bitwidth')
                    if bias_bw:
                        bias_quantizer.bitwidth = bias_bw
                        changes.append(f"{bias_bw}-bit")
                    
                    # bias 通常使用对称量化
                    bias_sym = layer_config.get('bias_symmetric')
                    if bias_sym is not None:
                        bias_quantizer.symmetric = bias_sym
                        changes.append(f"{'sym' if bias_sym else 'asym'}")
                    
                    if changes and verbose:
                        if match_type == 'pattern':
                            print(f"  ✅ [pattern: {matched_pattern}] {name}.bias: {', '.join(changes)}")
                        else:
                            print(f"  ✅ [{match_type}] {name}.bias: {', '.join(changes)}")
            
            # 设置通用参数量化器（param_bitwidth / param_symmetric）
            # 作用于所有非 weight/bias 的 param_quantizers 键，如 Snake2d 的 alpha
            param_bw = layer_config.get('param_bitwidth')
            param_sym = layer_config.get('param_symmetric')
            if (param_bw is not None or param_sym is not None) and hasattr(module, 'param_quantizers'):
                for param_key, param_quantizer in module.param_quantizers.items():
                    if param_key in ('weight', 'bias'):
                        continue  # weight/bias 已由专用字段处理
                    if param_quantizer is not None:
                        changes = []
                        if param_bw is not None:
                            param_quantizer.bitwidth = param_bw
                            changes.append(f"{param_bw}-bit")
                        if param_sym is not None:
                            param_quantizer.symmetric = param_sym
                            changes.append(f"{'sym' if param_sym else 'asym'}")
                        if changes and verbose:
                            prefix = f"pattern: {matched_pattern}" if match_type == 'pattern' else match_type
                            print(f"  ✅ [{prefix}] {name}.{param_key}: {', '.join(changes)}")

            # 设置输入量化参数
            if hasattr(module, 'input_quantizers') and len(module.input_quantizers) > 0:
                input_bws = layer_config.get('input_bitwidths')
                default_input_bw = layer_config.get('input_bitwidth')
                input_sym = layer_config.get('input_symmetric')
                for idx, input_quantizer in enumerate(module.input_quantizers):
                    if input_bws is not None:
                        input_bw = input_bws[idx] if idx < len(input_bws) else None
                    else:
                        input_bw = default_input_bw

                    # 若 slot 为 None 但配置了当前输入位宽，且是第 0 个输入（主输入），则创建新 quantizer
                    # （如 QuantizedVar / QuantizedSnake2d 因无 ONNX 映射，无法由 JSON config 自动创建）
                    # 仅对 idx==0 补建缺失 quantizer（QuantizedVar 等）；
                    # Mul/Add/Sub 的第二路 input 由 tutorial.json is_input_quantized 启用
                    # 标量输入槽（构造期已判定为“无 shape”并移除量化器）不得重建，否则会破坏
                    # 训练/reload 的量化器集合一致性，并使导出 encodings 重新包含标量路参数。
                    _is_scalar_slot = idx in getattr(module, "_scalar_input_slots", set())
                    if (input_quantizer is None and idx == 0 and input_bw is not None
                            and match_type != 'default' and not _is_scalar_slot):
                        try:
                            from aimet_torch.v2.quantization.affine import QuantizeDequantize
                            import torch.nn as _nn
                            sym = input_sym if input_sym is not None else True
                            signed = layer_config.get('input_signed', default_config.get('signed', True))
                            signs = layer_config.get('input_signeds')
                            if signs is not None and idx < len(signs) and signs[idx] is not None:
                                signed = signs[idx]
                            new_q = QuantizeDequantize(shape=(), bitwidth=input_bw, symmetric=sym, signed=signed)
                            q_device = _infer_quant_module_device(module)
                            if q_device is not None:
                                new_q = new_q.to(q_device)
                            # nn.ModuleList 不支持直接用索引赋值 None 元素，需重建整个列表
                            slots = list(module.input_quantizers)
                            slots[idx] = new_q
                            module.input_quantizers = _nn.ModuleList(slots)
                            input_quantizer = new_q
                            if verbose:
                                label = f"pattern: {matched_pattern}" if match_type == 'pattern' else match_type
                                print(f"  ➕ [{label}] {name}.input[{idx}]: 创建新 quantizer {input_bw}-bit {'sym' if sym else 'asym'}")
                        except Exception as _e:
                            if verbose:
                                print(f"  ⚠️  {name}.input[{idx}]: 创建 quantizer 失败 - {_e}")

                    if input_quantizer is not None:
                        changes = []

                        if input_bw:
                            input_quantizer.bitwidth = input_bw
                            changes.append(f"{input_bw}-bit")

                        if input_sym is not None:
                            input_quantizer.symmetric = input_sym
                            changes.append(f"{'sym' if input_sym else 'asym'}")

                        if verbose and changes:
                            label = f"pattern: {matched_pattern}" if match_type == 'pattern' else match_type
                            if len(module.input_quantizers) > 1:
                                print(f"  ✅ [{label}] {name}.input[{idx}]: {', '.join(changes)}")
                            else:
                                print(f"  ✅ [{label}] {name}.input: {', '.join(changes)}")
            
            # 设置输出量化参数
            if hasattr(module, 'output_quantizers') and len(module.output_quantizers) > 0:
                output_quantizer = module.output_quantizers[0]
                output_bw = layer_config.get('output_bitwidth')
                output_sym = layer_config.get('output_symmetric')
                if output_quantizer is None and output_bw is not None:
                    try:
                        from aimet_torch.v2.quantization.affine import QuantizeDequantize
                        import torch.nn as _nn
                        sym = output_sym if output_sym is not None else True
                        signed = layer_config.get('output_signed', default_config.get('signed', True))
                        new_q = QuantizeDequantize(shape=(), bitwidth=output_bw, symmetric=sym, signed=signed)
                        slots = list(module.output_quantizers)
                        slots[0] = new_q
                        module.output_quantizers = _nn.ModuleList(slots)
                        output_quantizer = new_q
                        if verbose:
                            label = f"pattern: {matched_pattern}" if match_type == 'pattern' else match_type
                            print(f"  ➕ [{label}] {name}.output[0]: 创建新 quantizer {output_bw}-bit {'sym' if sym else 'asym'}")
                    except Exception as _e:
                        if verbose:
                            print(f"  ⚠️  {name}.output[0]: 创建 quantizer 失败 - {_e}")
                if output_quantizer is not None:
                    changes = []
                    
                    # 位宽
                    if output_bw:
                        output_quantizer.bitwidth = output_bw
                        changes.append(f"{output_bw}-bit")
                    
                    # 对称性
                    if output_sym is not None:
                        output_quantizer.symmetric = output_sym
                        changes.append(f"{'sym' if output_sym else 'asym'}")
                    
                    if verbose and changes:
                        # 只在没有设置权重时打印，避免重复
                        if not (hasattr(module, 'param_quantizers') and 'weight' in module.param_quantizers):
                            print(f"  ✅ [{match_type}] {name}.output: {', '.join(changes)}")
                    
                    # 统计不同类型的层（只统计有输出量化器的层）
                    if 'Tanh' in module_type or 'Sigmoid' in module_type or 'ReLU' in module_type:
                        stats['activation_count'] += 1
                    elif 'Add' in module_type:
                        stats['add_count'] += 1
                    elif 'Multiply' in module_type:
                        stats['multiply_count'] += 1
                    elif 'Subtract' in module_type:
                        stats['subtract_count'] += 1
    
    for name, module, op in recurrent_modules:
        module.set_module_name(name)
        module.load_bitwidth_config(config_file, verbose=verbose)
        key = op.package
        stats[f"{key}_count"] = stats.get(f"{key}_count", 0) + 1
        for state in ("enabled", "disabled"):
            stats.setdefault(f"{key}_{state}", 0)
        state = "enabled" if module.use_quantization else "disabled"
        stats[f"{key}_{state}"] += 1

    # 打印统计
    if verbose:
        print("="*70)
        print(f"📊 位宽设置统计:")
        print(f"  • Conv 层:             {stats['conv_count']} 个")
        print(f"  • Linear 层:           {stats['linear_count']} 个")
        print(f"  • Add 层:              {stats['add_count']} 个")
        print(f"  • Multiply 层:         {stats['multiply_count']} 个")
        print(f"  • Subtract 层:         {stats['subtract_count']} 个")
        print(f"  • 激活函数层:          {stats['activation_count']} 个")
        for op in dict((op.package, op) for _, _, op in recurrent_modules).values():
            key = op.package
            print(f"  • {op.module_type.__name__}: {stats[f'{key}_count']} 个 "
                  f"(启用量化: {stats[f'{key}_enabled']}, 未启用: {stats[f'{key}_disabled']})")
        print(f"  • 禁用量化层:          {stats['disabled_count']} 个")
        print(f"  • 按名称匹配:          {stats['name_matched_count']} 个")
        print(f"  • 按类型匹配:          {stats['type_matched_count']} 个")
        print(f"  • 使用默认值:          {stats['default_count']} 个")
        print("="*70)

    return stats

def _infer_quant_module_device(module) -> "torch.device | None":
    """从已有 quantizer / 参数 / buffer 推断模块设备。"""
    import torch
    for qlist in (
        getattr(module, "input_quantizers", []),
        getattr(module, "output_quantizers", []),
    ):
        for q in qlist:
            if q is not None:
                try:
                    return next(q.parameters()).device
                except StopIteration:
                    pass
    for p in module.parameters(recurse=False):
        return p.device
    for b in module.buffers(recurse=False):
        return b.device
    return None


def _disable_quantization(module, name: str, verbose: bool):
    """
    禁用模块的量化（辅助函数）
    
    Args:
        module: 要禁用量化的模块
        name: 模块名称
        verbose: 是否打印详细信息
    """
    # 移除输出量化器
    if hasattr(module, 'output_quantizers') and len(module.output_quantizers) > 0:
        for idx in range(len(module.output_quantizers)):
            if module.output_quantizers[idx] is not None:
                if verbose:
                    print(f"  🚫 {name}: 禁用输出量化器 [{idx}]")
                module.output_quantizers[idx] = None
    
    # 移除输入量化器
    if hasattr(module, 'input_quantizers') and len(module.input_quantizers) > 0:
        for idx in range(len(module.input_quantizers)):
            if module.input_quantizers[idx] is not None:
                if verbose:
                    print(f"  🚫 {name}: 禁用输入量化器 [{idx}]")
                module.input_quantizers[idx] = None
    
    # 移除参数量化器
    if hasattr(module, 'param_quantizers'):
        for param_name in list(module.param_quantizers.keys()):
            if module.param_quantizers[param_name] is not None:
                if verbose:
                    print(f"  🚫 {name}: 禁用参数量化器 [{param_name}]")
                module.param_quantizers[param_name] = None



def check_bn_params_status(model, verbose: bool = True) -> Dict[str, Any]:
    """
    检查 BatchNorm 参数的状态（是否可训练、是否存在）
    
    这个函数用于诊断 BatchNorm affine 参数的状态，帮助排查 QAT 训练问题。
    
    Args:
        model: PyTorch 模型或 AIMET QuantizationSimModel.model
        verbose: 是否打印详细信息
    
    Returns:
        结果字典，包含：
        - bn_modules: BatchNorm 模块数量
        - trainable_affine_params: 可训练的 affine 参数列表
        - frozen_affine_params: 冻结的 affine 参数列表
        - total_affine_params: affine 参数总数
    """
    import torch.nn as nn
    
    # 常见 BN 模块名称模式
    bn_name_patterns = ['_bn', 'bn_', 'pre_bn', 'rnn2d_bn', 'batchnorm', 'batch_norm']
    
    trainable_affine_params = []
    frozen_affine_params = []
    bn_modules = []
    
    # 策略 1: 通过类型识别 BatchNorm 模块
    try:
        from aimet_torch.quantizable_batchnorm import QuantizableBatchNorm2d
        custom_bn_types = (QuantizableBatchNorm2d,)
    except ImportError:
        custom_bn_types = ()
    
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d) + custom_bn_types
    
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            bn_modules.append((name, module))
    
    # 策略 2: 检查参数状态
    for name, param in model.named_parameters():
        # 检查是否是 affine 参数
        is_affine_param = name.endswith('.weight') or name.endswith('.bias')
        if not is_affine_param:
            continue
        
        # 检查是否属于 BatchNorm
        name_lower = name.lower()
        is_bn_param = (any(pattern in name_lower for pattern in bn_name_patterns)
                               or 'bn' in name_lower.split('.'))
        
        # 检查是否在识别的 BN 模块中
        for bn_name, bn_module in bn_modules:
            if name.startswith(bn_name + '.'):
                is_bn_param = True
                break
        
        if is_bn_param:
            if param.requires_grad:
                trainable_affine_params.append(name)
            else:
                frozen_affine_params.append(name)
    
    if verbose:
        print("\n" + "="*70)
        print("BatchNorm 参数状态检查")
        print("="*70)
        print(f"发现 BatchNorm 模块:     {len(bn_modules)} 个")
        
        if bn_modules:
            print("\nBatchNorm 模块列表:")
            for bn_name, bn_module in bn_modules:
                affine_status = "affine=True" if hasattr(bn_module, 'weight') and bn_module.weight is not None else "affine=False"
                training_status = "train" if bn_module.training else "eval"
                print(f"  - {bn_name}: {type(bn_module).__name__} ({affine_status}, {training_status})")
        
        print(f"\n可训练的 affine 参数:   {len(trainable_affine_params)} 个")
        if trainable_affine_params:
            print("  ⚠️  警告：以下 BN affine 参数仍可训练（可能导致 QAT 不稳定）:")
            for param_name in trainable_affine_params:
                print(f"      - {param_name}")
        
        print(f"\n冻结的 affine 参数:     {len(frozen_affine_params)} 个")
        if frozen_affine_params and len(frozen_affine_params) <= 10:
            print("  ✅ 已冻结的参数:")
            for param_name in frozen_affine_params:
                print(f"      - {param_name}")
        
        if trainable_affine_params:
            print("\n💡 建议：在 QAT 训练前调用 freeze_quantizer_parameters(sim.model, freeze_bn_affine=True)")
        else:
            print("\n✅ 所有 BatchNorm affine 参数已正确冻结")
        
        print("="*70 + "\n")
    
    return {
        'bn_modules': bn_modules,
        'bn_module_count': len(bn_modules),
        'trainable_affine_params': trainable_affine_params,
        'frozen_affine_params': frozen_affine_params,
        'total_affine_params': len(trainable_affine_params) + len(frozen_affine_params)
    }


def _align_conv_bias_scale(model, verbose: bool = True, bias_bitwidth: int = 32) -> Dict[str, Any]:
    """Align Conv/Linear bias to the accumulator grid, rejecting unrepresentable values.

    The requested signed bias width is applied to the actual PyTorch quantizer.
    All candidate bias grids are checked before any bias quantizer is replaced.
    """
    import copy
    import torch
    import torch.nn as nn
    from aimet_torch.v2.nn import QuantizationMixin
    from aimet_common.power_of_2 import validate_bias_bitwidth, validate_bias_values

    validate_bias_bitwidth(bias_bitwidth)
    stats = {'total_conv': 0, 'modified_conv': 0, 'skipped': [], 'scale_changes': []}
    conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d,
                  nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)
    pending = []
    for name, module in model.named_modules():
        if not isinstance(module, conv_types):
            continue
        stats['total_conv'] += 1
        if module.bias is None or not isinstance(module, QuantizationMixin):
            stats['skipped'].append((name, 'no bias or not quantized'))
            continue
        weight_q = module.param_quantizers['weight']
        bias_q = module.param_quantizers['bias']
        input_q = module.input_quantizers[0] if module.input_quantizers else None
        if any(q is None or not q.is_initialized() for q in (weight_q, bias_q, input_q)):
            stats['skipped'].append((name, 'input/weight/bias quantizer not initialized'))
            continue

        sx = input_q.get_scale().detach()
        sw = weight_q.get_scale().detach()
        if sx.numel() != 1 or sw.numel() not in (1, module.bias.numel()):
            raise ValueError(f"{name}: bias alignment requires scalar input and output-channel weight scales")
        target = (sx.reshape(()) * sw.reshape(-1)).to(module.bias.device)
        validate_bias_values(module.bias.detach().double().cpu().numpy(),
                             target.double().cpu().numpy(), bias_bitwidth, name=name)

        aligned = copy.deepcopy(bias_q)
        aligned.bitwidth = bias_bitwidth
        aligned.signed = True
        aligned.symmetric = True
        aligned._reparametrize_to_scale_offset()
        if target.numel() == 1:
            target = target.reshape(()).expand_as(aligned.scale)
        elif target.numel() == aligned.scale.numel():
            target = target.reshape_as(aligned.scale)
        else:
            raise ValueError(f"{name}: bias quantizer shape cannot represent the output-channel accumulator grid")
        with torch.no_grad():
            aligned.scale.copy_(target)
        if not torch.equal(aligned.get_scale(), target):
            raise ValueError(f"{name}: bias accumulator scale is below the quantizer's representable scale range")
        pending.append((name, module, aligned, sx, sw, bias_q.get_scale().detach().clone(), target))

    for name, module, aligned, sx, sw, previous, target in pending:
        module.param_quantizers['bias'] = aligned
        stats['modified_conv'] += 1
        stats['scale_changes'].append({
            'layer': name, 'Sx': sx.cpu().tolist(), 'Sw': sw.cpu().tolist(),
            'Sb_old': previous.cpu().tolist(), 'Sb_new': target.cpu().tolist(),
            'bias_bitwidth': bias_bitwidth, 'is_per_channel': target.numel() > 1,
        })
        if verbose:
            print(f"{name}: bias aligned to Sx*Sw, signed {bias_bitwidth}-bit; range checked")
    return stats
