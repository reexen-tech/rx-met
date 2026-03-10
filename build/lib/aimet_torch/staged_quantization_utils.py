"""
分阶段量化工具函数
支持按算子类型进行分阶段量化，包括保存和加载量化参数
"""
import torch
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


def save_quantizer_encodings(
    sim_model,
    save_path: str,
    layer_types: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    保存指定层类型的量化参数到 JSON 文件
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        save_path: 保存路径
        layer_types: 要保存的层类型列表，例如 ["QuantizedConv2d", "QuantizedGRU"]
                    如果为 None，则保存所有量化层
        verbose: 是否打印详细信息
    
    Returns:
        保存的量化参数字典
    """
    encodings_dict = {}
    saved_count = 0
    
    for module_name, module in sim_model.named_modules():
        module_type = type(module).__name__
        
        # 检查是否匹配指定的层类型
        if layer_types is not None:
            if not any(layer_type in module_type for layer_type in layer_types):
                continue
        
        # 保存 param_quantizers (权重量化器)
        if hasattr(module, 'param_quantizers'):
            for param_name, quantizer in module.param_quantizers.items():
                if quantizer is not None and quantizer.is_initialized():
                    encoding = quantizer.get_encodings()
                    if encoding is not None:
                        key = f"{module_name}.param_quantizers.{param_name}"
                        encodings_dict[key] = {
                            'module_name': module_name,
                            'module_type': module_type,
                            'quantizer_type': 'param',
                            'param_name': param_name,
                            'scale': _tensor_to_list(encoding.scale),
                            'offset': _tensor_to_list(encoding.offset),
                            'qmin': int(encoding.qmin),
                            'qmax': int(encoding.qmax),
                            'symmetric': bool(encoding.symmetry),
                            'min': _tensor_to_list(encoding.min),
                            'max': _tensor_to_list(encoding.max),
                            'shape': list(encoding.scale.shape) if isinstance(encoding.scale, torch.Tensor) else [],
                        }
                        saved_count += 1
        
        # 保存 output_quantizers (输出量化器)
        if hasattr(module, 'output_quantizers'):
            for idx, quantizer in enumerate(module.output_quantizers):
                if quantizer is not None and quantizer.is_initialized():
                    encoding = quantizer.get_encodings()
                    if encoding is not None:
                        key = f"{module_name}.output_quantizers.{idx}"
                        encodings_dict[key] = {
                            'module_name': module_name,
                            'module_type': module_type,
                            'quantizer_type': 'output',
                            'quantizer_idx': idx,
                            'scale': _tensor_to_list(encoding.scale),
                            'offset': _tensor_to_list(encoding.offset),
                            'qmin': int(encoding.qmin),
                            'qmax': int(encoding.qmax),
                            'symmetric': bool(encoding.symmetry),
                            'min': _tensor_to_list(encoding.min),
                            'max': _tensor_to_list(encoding.max),
                            'shape': list(encoding.scale.shape) if isinstance(encoding.scale, torch.Tensor) else [],
                        }
                        saved_count += 1
    
    # 保存到文件
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(encodings_dict, f, indent=2, ensure_ascii=False)
    
    if verbose:
        layer_type_str = ', '.join(layer_types) if layer_types else 'all'
        print(f"\n✅ 保存量化参数:")
        print(f"   文件路径: {save_path}")
        print(f"   层类型: {layer_type_str}")
        print(f"   保存数量: {saved_count} 个量化器")
    
    return encodings_dict


def load_quantizer_encodings(
    sim_model,
    load_path: str,
    layer_types: Optional[List[str]] = None,
    exclude_layer_types: Optional[List[str]] = None,
    skip_if_not_found: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    从 JSON 文件加载量化参数并应用到模型
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        load_path: 加载路径
        layer_types: 要加载的层类型列表，例如 ["QuantizedConv2d"]
                    如果为 None，则加载所有匹配的量化器
        exclude_layer_types: 要排除的层类型列表，例如 ["QuantizedGRU"]
                           优先级高于 layer_types
        skip_if_not_found: 如果量化器不存在，是否跳过
        verbose: 是否打印详细信息
    
    Returns:
        加载统计信息
    
    Examples:
        # 1. 加载所有保存的量化参数
        load_quantizer_encodings(sim.model, "stage1.json")
        
        # 2. 只加载 Conv2d
        load_quantizer_encodings(sim.model, "stage1.json", layer_types=["QuantizedConv2d"])
        
        # 3. 加载除了 GRU 之外的所有参数
        load_quantizer_encodings(sim.model, "stage1.json", exclude_layer_types=["QuantizedGRU"])
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"编码文件不存在: {load_path}")
    
    with open(load_path, 'r', encoding='utf-8') as f:
        encodings_dict = json.load(f)
    
    loaded_count = 0
    skipped_count = 0
    loaded_types = set()
    
    # 尝试导入 QuantGRU
    try:
        from quant_gru import QuantGRU
    except ImportError:
        try:
            import sys
            from pathlib import Path
            quant_gru_path = Path(__file__).parent.parent.parent.parent / "quant-gru-pytorch" / "pytorch"
            if quant_gru_path.exists():
                sys.path.insert(0, str(quant_gru_path))
            from quant_gru import QuantGRU
        except ImportError:
            QuantGRU = None
    
    # 先处理 QuantGRU 模块（如果 encodings_dict 包含 activation_encodings）
    if QuantGRU is not None and "activation_encodings" in encodings_dict:
        if verbose:
            print("\n📥 尝试从 AIMET encodings 加载 QuantGRU 量化参数...")
        
        for module_name, module in sim_model.named_modules():
            if isinstance(module, QuantGRU):
                target_name = getattr(module, 'aimet_onnx_name', None)
                if not target_name:
                    # 尝试从 encodings 文件中查找对应的键
                    # 通常 encodings 中的键就是模块路径名称
                    if module_name in encodings_dict.get("activation_encodings", {}):
                        target_name = module_name
                        # 设置 aimet_onnx_name 以便后续使用
                        module.aimet_onnx_name = target_name
                    else:
                        # 如果找不到，使用模块路径名称作为后备
                        target_name = module_name
                        if verbose:
                            print(f"  ⚠️ 警告：{module_name} 的 aimet_onnx_name 未设置，使用模块路径名称作为后备")
                
                if module.load_quant_params_from_aimet_format(
                    encodings_dict,
                    module_name=target_name,
                    verbose=verbose
                ):
                    loaded_count += 1
                    loaded_types.add("QuantGRU")
                    if verbose:
                        print(f"  ✅ {module_name}: 成功加载 QuantGRU 量化参数")
    
    # 检查 encodings 文件格式：AIMET 标准格式（activation_encodings + param_encodings）
    if "activation_encodings" in encodings_dict or "param_encodings" in encodings_dict:
        if verbose:
            print("\n📝 检测到 AIMET 标准格式 encodings")
            print("   QuantGRU 参数已加载完成，正在加载 CONV 等 AIMET 量化器...")
        module_map = {name: m for name, m in sim_model.named_modules()}
        device = _get_model_device(sim_model)
        act = encodings_dict.get("activation_encodings", {})
        param_enc = encodings_dict.get("param_encodings", {})
        gru_module_names = _collect_gru_module_names_from_activation_encodings(act)
        # 1) 非 GRU 的 activation_encodings：优先用 is_GRU 标记，无则用名字启发式
        for mod_name, enc in act.items():
            if _is_gru_activation_by_encoding(enc):
                continue
            if isinstance(enc, dict) and enc.get("is_GRU") is None and _is_gru_activation_key(mod_name):
                continue
            module = module_map.get(mod_name)
            if module is None:
                if verbose:
                    print(f"  ⚠️ 模块不存在，跳过: {mod_name}")
                skipped_count += 1
                continue
            if not isinstance(enc, dict):
                continue
            # input: 对应 input_quantizers[0], [1], ...
            inp_enc = enc.get("input")
            if isinstance(inp_enc, list) and len(inp_enc) > 0 and hasattr(module, 'input_quantizers'):
                for idx, item in enumerate(inp_enc):
                    if idx >= len(module.input_quantizers) or module.input_quantizers[idx] is None:
                        break
                    q = module.input_quantizers[idx]
                    rmin = item.get("real_min")
                    rmax = item.get("real_max")
                    if rmin is not None and rmax is not None and _apply_aimet_encoding_to_quantizer(q, rmin, rmax, device, item, verbose):
                        loaded_count += 1
                        loaded_types.add(type(module).__name__)
            elif isinstance(inp_enc, dict):
                rmin, rmax = inp_enc.get("real_min"), inp_enc.get("real_max")
                if hasattr(module, 'input_quantizers') and len(module.input_quantizers) > 0 and module.input_quantizers[0] is not None and rmin is not None and rmax is not None:
                    if _apply_aimet_encoding_to_quantizer(module.input_quantizers[0], rmin, rmax, device, inp_enc, verbose):
                        loaded_count += 1
                        loaded_types.add(type(module).__name__)
            # output: 对应 output_quantizers[0], [1], ...
            out_enc = enc.get("output")
            if isinstance(out_enc, list) and len(out_enc) > 0 and hasattr(module, 'output_quantizers'):
                for idx, item in enumerate(out_enc):
                    if idx >= len(module.output_quantizers) or module.output_quantizers[idx] is None:
                        break
                    q = module.output_quantizers[idx]
                    rmin = item.get("real_min")
                    rmax = item.get("real_max")
                    if rmin is not None and rmax is not None and _apply_aimet_encoding_to_quantizer(q, rmin, rmax, device, item, verbose):
                        loaded_count += 1
                        loaded_types.add(type(module).__name__)
            elif isinstance(out_enc, dict):
                rmin, rmax = out_enc.get("real_min"), out_enc.get("real_max")
                if hasattr(module, 'output_quantizers') and len(module.output_quantizers) > 0 and module.output_quantizers[0] is not None and rmin is not None and rmax is not None:
                    if _apply_aimet_encoding_to_quantizer(module.output_quantizers[0], rmin, rmax, device, out_enc, verbose):
                        loaded_count += 1
                        loaded_types.add(type(module).__name__)
        # 2) 非 GRU 的 param_encodings：通用解析 key 为 "module_name.param_name"（与 param_quantizers 匹配）
        for key, enc in param_enc.items():
            if not isinstance(enc, dict):
                continue
            # 标准做法：param key 形如 "module_name.param_name"，GRU 模块由 is_GRU 已收集
            if any(key.startswith(g + ".") for g in gru_module_names):
                continue
            rmin = enc.get("real_min")
            rmax = enc.get("real_max")
            if rmin is None or rmax is None:
                continue
            parsed = _parse_param_encodings_key(key, module_map)
            if parsed is None:
                skipped_count += 1
                continue
            mod_name, param_name = parsed
            module = module_map[mod_name]
            if param_name not in module.param_quantizers:
                skipped_count += 1
                continue
            q = module.param_quantizers[param_name]
            if q is None:
                skipped_count += 1
                continue
            if _apply_aimet_encoding_to_quantizer(q, rmin, rmax, device, enc, verbose):
                loaded_count += 1
                loaded_types.add(type(module).__name__)
        if verbose:
            print(f"\n✅ AIMET 格式加载完成: 共加载 {loaded_count} 个量化器")
        return {
            'loaded': loaded_count,
            'skipped': skipped_count,
            'loaded_types': list(loaded_types)
        }
    
    # 创建模块映射
    module_map = {name: module for name, module in sim_model.named_modules()}
    
    for key, encoding_info in encodings_dict.items():
        module_name = encoding_info['module_name']
        module_type = encoding_info.get('module_type', '')
        
        # 优先检查排除列表
        if exclude_layer_types is not None:
            if any(layer_type in module_type for layer_type in exclude_layer_types):
                skipped_count += 1
                continue
        
        # 检查层类型是否匹配
        if layer_types is not None:
            if not any(layer_type in module_type for layer_type in layer_types):
                skipped_count += 1
                continue
        
        # 检查模块是否存在
        if module_name not in module_map:
            if skip_if_not_found:
                skipped_count += 1
                continue
            else:
                raise KeyError(f"模块不存在: {module_name}")
        
        module = module_map[module_name]
        
        # 加载 param_quantizers
        if encoding_info['quantizer_type'] == 'param':
            param_name = encoding_info['param_name']
            if hasattr(module, 'param_quantizers') and param_name in module.param_quantizers:
                quantizer = module.param_quantizers[param_name]
                if quantizer is not None:
                    _apply_encoding_to_quantizer(quantizer, encoding_info)
                    loaded_count += 1
                    loaded_types.add(module_type)
                    if verbose:
                        print(f"  ✅ {module_name}.param_quantizers.{param_name}")
                else:
                    if not skip_if_not_found:
                        raise ValueError(f"量化器为 None: {key}")
                    skipped_count += 1
            else:
                if not skip_if_not_found:
                    raise ValueError(f"参数量化器不存在: {key}")
                skipped_count += 1
        
        # 加载 output_quantizers
        elif encoding_info['quantizer_type'] == 'output':
            quantizer_idx = encoding_info['quantizer_idx']
            if hasattr(module, 'output_quantizers') and quantizer_idx < len(module.output_quantizers):
                quantizer = module.output_quantizers[quantizer_idx]
                if quantizer is not None:
                    _apply_encoding_to_quantizer(quantizer, encoding_info)
                    loaded_count += 1
                    loaded_types.add(module_type)
                    if verbose:
                        print(f"  ✅ {module_name}.output_quantizers[{quantizer_idx}]")
                else:
                    if not skip_if_not_found:
                        raise ValueError(f"量化器为 None: {key}")
                    skipped_count += 1
            else:
                if not skip_if_not_found:
                    raise ValueError(f"输出量化器不存在: {key}")
                skipped_count += 1
    
    if verbose:
        print(f"\n✅ 加载量化参数:")
        print(f"   文件路径: {load_path}")
        if layer_types:
            print(f"   指定层类型: {', '.join(layer_types)}")
        if exclude_layer_types:
            print(f"   排除层类型: {', '.join(exclude_layer_types)}")
        if not layer_types and not exclude_layer_types:
            print(f"   加载模式: 全部加载")
        print(f"   加载数量: {loaded_count} 个量化器")
        print(f"   涉及层类型: {', '.join(sorted(loaded_types))}")
        if skipped_count > 0:
            print(f"   跳过数量: {skipped_count} 个")
    
    return {
        'loaded': loaded_count,
        'skipped': skipped_count,
        'loaded_types': list(loaded_types)
    }


def disable_quantization_by_layer_types(
    sim_model,
    layer_types: List[str],
    verbose: bool = True
) -> Dict[str, int]:
    """
    禁用指定层类型的所有量化器
    
    Args:
        sim_model: AIMET QuantizationSimModel.model
        layer_types: 要禁用的层类型列表，例如 ["QuantizedGRU", "QuantizedLinear"]
        verbose: 是否打印详细信息
    
    Returns:
        禁用统计信息
    """
    disabled_modules = []
    disabled_param_count = 0
    disabled_output_count = 0
    
    for module_name, module in sim_model.named_modules():
        module_type = type(module).__name__
        
        # 检查是否匹配指定的层类型
        if not any(layer_type in module_type for layer_type in layer_types):
            continue
        
        # 禁用 param_quantizers
        if hasattr(module, 'param_quantizers'):
            for param_name, quantizer in module.param_quantizers.items():
                if quantizer is not None:
                    module.param_quantizers[param_name] = None
                    disabled_param_count += 1
        
        # 禁用 output_quantizers
        if hasattr(module, 'output_quantizers'):
            for idx in range(len(module.output_quantizers)):
                if module.output_quantizers[idx] is not None:
                    module.output_quantizers[idx] = None
                    disabled_output_count += 1
        
        # 禁用 input_quantizers
        if hasattr(module, 'input_quantizers'):
            for idx in range(len(module.input_quantizers)):
                if module.input_quantizers[idx] is not None:
                    module.input_quantizers[idx] = None
        
        disabled_modules.append(module_name)
    
    if verbose:
        layer_type_str = ', '.join(layer_types)
        print(f"\n🚫 禁用量化:")
        print(f"   层类型: {layer_type_str}")
        print(f"   涉及模块: {len(disabled_modules)} 个")
        print(f"   参数量化器: {disabled_param_count} 个")
        print(f"   输出量化器: {disabled_output_count} 个")
    
    return {
        'disabled_modules': len(disabled_modules),
        'disabled_param_quantizers': disabled_param_count,
        'disabled_output_quantizers': disabled_output_count
    }


def _tensor_to_list(value):
    """将张量或标量转换为列表"""
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    elif isinstance(value, (int, float)):
        return value
    else:
        return value


def _is_gru_activation_by_encoding(enc: dict) -> bool:
    """根据 encodings 条目的 is_GRU 标记判断是否为 GRU（标准做法）。"""
    if isinstance(enc, dict) and "is_GRU" in enc:
        return enc.get("is_GRU") is True
    return False


def _is_gru_activation_key(key: str) -> bool:
    """启发式：key 含 seq_t/seq_f 时视为 GRU（仅当 enc 中无 is_GRU 时作后备）。"""
    return "seq_t" in key or "seq_f" in key


def _collect_gru_module_names_from_activation_encodings(act: dict) -> set:
    """从 activation_encodings 中收集 is_GRU 为 True 的模块名（用于 param 跳过）。"""
    out = set()
    for mod_name, enc in act.items():
        if isinstance(enc, dict) and enc.get("is_GRU") is True:
            out.add(mod_name)
    return out


def _parse_param_encodings_key(key: str, module_map: dict):
    """
    通用解析 param_encodings 的 key，得到 (module_name, param_name) 或 None。
    约定：key 形如 "module_name.param_name"，其中 param_name 必须在 module.param_quantizers 中存在。
    按最长匹配优先，避免 "a.b.c" 被拆成 "a.b" + "c" 而 "a.b.c" 才是模块名。
    """
    if not key or "." not in key:
        return None
    # 按模块名长度从长到短尝试，以便匹配 "generator.enc_seqs.0.conv_f" 而非 "generator"
    candidates = sorted(module_map.keys(), key=len, reverse=True)
    for mod_name in candidates:
        prefix = mod_name + "."
        if key.startswith(prefix):
            param_name = key[len(prefix):]
            module = module_map.get(mod_name)
            if module is not None and hasattr(module, 'param_quantizers') and param_name in module.param_quantizers:
                return (mod_name, param_name)
    return None


def _encoding_compatible_with_quantizer(enc_entry: dict, quantizer, verbose: bool) -> tuple:
    """
    校验 encodings 条目与量化器是否兼容（bitwidth、symmetric）。
    enc_entry 可为单条 dict（含 bitwidth, is_symmetric）或 list 的首元素。
    Returns: (ok: bool, msg: str)
    """
    if not isinstance(enc_entry, dict):
        return True, ""
    entry = enc_entry.get("input", enc_entry) if isinstance(enc_entry.get("input"), list) else enc_entry
    if isinstance(entry, list) and len(entry) > 0:
        entry = entry[0]
    if not isinstance(entry, dict):
        return True, ""
    file_bw = entry.get("bitwidth")
    file_sym = entry.get("is_symmetric")
    if file_sym is not None and isinstance(file_sym, str):
        file_sym = file_sym.lower() == "true"
    if file_bw is None and file_sym is None:
        return True, ""
    try:
        q_bw = None
        if hasattr(quantizer, 'qmin') and hasattr(quantizer, 'qmax'):
            num_steps = int(quantizer.qmax) - int(quantizer.qmin) + 1
            q_bw = int(round((num_steps - 1).bit_length())) if num_steps > 1 else 0
        q_sym = getattr(quantizer, 'symmetric', getattr(quantizer, '_symmetric', None))
        if file_bw is not None and q_bw is not None and file_bw != q_bw:
            return False, f"bitwidth 不一致: file={file_bw}, quantizer≈{q_bw}"
        if file_sym is not None and q_sym is not None and bool(file_sym) != bool(q_sym):
            return False, f"symmetric 不一致: file={file_sym}, quantizer={q_sym}"
    except Exception as e:
        if verbose:
            print(f"  [校验兼容性时忽略]: {e}")
    return True, ""


def _get_model_device(sim_model):
    """从模型获取 device，优先 parameters，其次 buffers，再次从任意量化器取。"""
    try:
        return next(sim_model.parameters()).device
    except StopIteration:
        pass
    try:
        return next(sim_model.buffers()).device
    except StopIteration:
        pass
    for _name, mod in sim_model.named_modules():
        if hasattr(mod, 'param_quantizers'):
            for q in mod.param_quantizers.values() if hasattr(mod.param_quantizers, 'values') else []:
                if q is not None and hasattr(q, 'parameters'):
                    for p in q.parameters():
                        return p.device
    return None


def _apply_aimet_encoding_to_quantizer(quantizer, real_min, real_max, device=None, enc_entry=None, verbose=False):
    """
    将 AIMET 导出格式的 real_min/real_max 应用到量化器（调用 set_range）。
    real_min, real_max 可为标量或列表，会转为 tensor。
    若 enc_entry 提供且与量化器不兼容（bitwidth/symmetric），则跳过并返回 False。
    """
    if not hasattr(quantizer, 'set_range'):
        return False
    if real_min is None or real_max is None:
        return False
    if enc_entry is not None:
        ok, msg = _encoding_compatible_with_quantizer(enc_entry, quantizer, verbose)
        if not ok:
            if verbose:
                print(f"  ⚠️ 跳过（编码不兼容）: {msg}")
            return False
    min_t = torch.tensor(real_min, dtype=torch.float32)
    max_t = torch.tensor(real_max, dtype=torch.float32)
    if device is not None:
        min_t = min_t.to(device)
        max_t = max_t.to(device)
    try:
        if hasattr(quantizer, 'min') and isinstance(quantizer.min, torch.Tensor) and quantizer.min.numel() > 1 and min_t.numel() == 1:
            min_t = min_t.expand(quantizer.min.shape).contiguous()
            max_t = max_t.expand(quantizer.max.shape).contiguous()
        quantizer.set_range(min_t, max_t)
        return True
    except Exception as e:
        if verbose:
            print(f"  ⚠️ set_range 失败: {e}")
        return False


def _apply_encoding_to_quantizer(quantizer, encoding_info: Dict):
    """将编码信息应用到量化器"""
    # 转换为张量
    scale_data = encoding_info['scale']
    offset_data = encoding_info['offset']
    min_data = encoding_info['min']
    max_data = encoding_info['max']
    
    # 处理标量或列表
    scale = torch.tensor(scale_data, dtype=torch.float32) if not isinstance(scale_data, torch.Tensor) else scale_data
    offset = torch.tensor(offset_data, dtype=torch.float32) if not isinstance(offset_data, torch.Tensor) else offset_data
    min_val = torch.tensor(min_data, dtype=torch.float32) if not isinstance(min_data, torch.Tensor) else min_data
    max_val = torch.tensor(max_data, dtype=torch.float32) if not isinstance(max_data, torch.Tensor) else max_data
    
    # 如果 shape 不为空，reshape
    if encoding_info.get('shape'):
        shape = tuple(encoding_info['shape'])
        if scale.numel() == 1 and shape:
            scale = scale.expand(shape)
        elif scale.shape != shape:
            scale = scale.view(shape)
        if offset.numel() == 1 and shape:
            offset = offset.expand(shape)
        elif offset.shape != shape:
            offset = offset.view(shape)
        if min_val.numel() == 1 and shape:
            min_val = min_val.expand(shape)
        elif min_val.shape != shape:
            min_val = min_val.view(shape)
        if max_val.numel() == 1 and shape:
            max_val = max_val.expand(shape)
        elif max_val.shape != shape:
            max_val = max_val.view(shape)
    
    # 确保在正确的设备上
    device = next((p.device for p in quantizer.parameters()), torch.device('cpu'))
    scale = scale.to(device)
    offset = offset.to(device)
    min_val = min_val.to(device)
    max_val = max_val.to(device)
    
    # 使用 set_range 方法设置编码
    try:
        quantizer.set_range(min_val, max_val)
    except Exception as e:
        print(f"    ⚠️  set_range 失败: {e}")
        # 如果 set_range 失败，尝试其他方法
        pass


def _has_active_quantizers(module) -> bool:
    """检查模块是否有任何非 None 的 input/output quantizer（即量化未被禁用）"""
    if hasattr(module, 'input_quantizers'):
        if any(q is not None for q in module.input_quantizers):
            return True
    if hasattr(module, 'output_quantizers'):
        if any(q is not None for q in module.output_quantizers):
            return True
    return False


def _find_next_call_module_users(node) -> list:
    """
    从 node.users 中找到紧邻的 call_module 后继节点。

    - call_function / call_method 节点（view/reshape/permute/contiguous 等）视为透明，
      递归进入其 users 继续查找。
    - 遇到 call_module 节点立即返回，不穿透。
    - 其他类型节点（output/placeholder 等）直接忽略。
    """
    result = []
    for user in node.users:
        if user.op == 'call_module':
            result.append(user)
        elif user.op in ('call_function', 'call_method'):
            # 透明算子：继续递归查找其后继
            result.extend(_find_next_call_module_users(user))
    return result


def sync_adjacent_quantizers(sim_model, verbose: bool = True) -> Dict[str, int]:
    """
    通过 FX Graph 遍历，将已初始化的 output_quantizers[0] 传播给
    相邻且未初始化的 input_quantizers[0]。

    传播规则：
    1. src_q (output_q) 已初始化 且 dst_q (input_q) 未初始化 → 传播
    2. call_function/call_method 节点（view/reshape/permute 等）→ 透明跳过
    3. 量化被禁用的 call_module（所有 quantizer 均为 None）→ 停止，不修改，不继续

    Args:
        sim_model: AIMET QuantizationSimModel.model（FX GraphModule）
        verbose:   是否打印详细日志

    Returns:
        {'synced': int, 'skipped_disabled': int}
    """
    if not hasattr(sim_model, 'graph'):
        if verbose:
            print("  ⚠️  sim_model 没有 .graph 属性，跳过 sync_adjacent_quantizers")
        return {'synced': 0, 'skipped_disabled': 0}

    # 构建 node.target → module 映射
    module_map = dict(sim_model.named_modules())

    synced_count = 0
    skipped_disabled = 0

    for node in sim_model.graph.nodes:
        if node.op != 'call_module':
            continue

        src_module = module_map.get(node.target)
        if src_module is None:
            continue

        # 检查 output_q[0] 是否存在且已初始化
        if not (hasattr(src_module, 'output_quantizers') and
                len(src_module.output_quantizers) > 0 and
                src_module.output_quantizers[0] is not None and
                src_module.output_quantizers[0].is_initialized()):
            continue

        src_q = src_module.output_quantizers[0]
        src_encoding = src_q.get_encodings()
        if src_encoding is None:
            continue

        # 找相邻 call_module 后继节点（透明跳过 call_function/call_method）
        next_nodes = _find_next_call_module_users(node)

        for next_node in next_nodes:
            dst_module = module_map.get(next_node.target)
            if dst_module is None:
                continue

            # 后继模块量化被禁用 → STOP
            if not _has_active_quantizers(dst_module):
                skipped_disabled += 1
                if verbose:
                    print(f"  ⏹  {node.target} → {next_node.target}: "
                          f"后继模块量化已禁用，停止传播")
                continue

            # input_q[0] 存在但未初始化 → 传播
            if not (hasattr(dst_module, 'input_quantizers') and
                    len(dst_module.input_quantizers) > 0 and
                    dst_module.input_quantizers[0] is not None):
                continue

            dst_q = dst_module.input_quantizers[0]
            if dst_q.is_initialized():
                continue  # 已初始化，不覆盖

            try:
                device = next(dst_q.parameters()).device
                dst_q.set_range(
                    src_encoding.min.to(device),
                    src_encoding.max.to(device)
                )
                synced_count += 1
                if verbose:
                    min_v = src_encoding.min.item() if src_encoding.min.numel() == 1 else src_encoding.min.tolist()
                    max_v = src_encoding.max.item() if src_encoding.max.numel() == 1 else src_encoding.max.tolist()
                    print(f"  🔗 {node.target}.output_q → {next_node.target}.input_q  "
                          f"[{min_v:.4f}, {max_v:.4f}]")
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  {node.target} → {next_node.target}: 传播失败 - {e}")

    if verbose:
        print(f"\n✅ sync_adjacent_quantizers 完成: "
              f"传播 {synced_count} 个, 跳过 disabled {skipped_disabled} 个")

    return {'synced': synced_count, 'skipped_disabled': skipped_disabled}

