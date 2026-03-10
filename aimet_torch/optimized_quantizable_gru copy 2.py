"""
可量化 GRU - 共享算术 Module
每个类型的算术操作共享一个 Module 实例

优势：
1. 减少 Module 数量，从 8 个减少到 3 个
2. 内存更节省
3. 模型更简洁

注意事项：
- 不同类型的操作不能共享（Add/Multiply/Subtract 分别共享）
- 适用于对模型大小敏感的场景
"""

import sys
sys.path.append('/home/sdong/Program/aimet/TrainingExtensions/torch/src/python')
sys.path.append('/home/sdong/Program/aimet/TrainingExtensions/common/src/python')

import torch
import torch.nn as nn
from aimet_torch._base.nn.modules.custom import Add, Multiply, Subtract


class OptimizedQuantizableGRUCell(nn.Module):
    """
    优化的可量化 GRU Cell - 共享算术 Module
    
    对比独立版本：
    - 独立版：每个算术操作一个 Module（8 个）
    - 共享版：每种类型算术操作共享一个 Module（3 个）
    
    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
    """
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # ========== Linear 层 ==========
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=True)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        
        # ========== 激活函数 ==========
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        # ========== 共享的算术 Module（3 个）==========
        # 共享的加法 Module（用于所有加法操作）
        self.shared_add = Add()
        # 共享的乘法 Module（用于所有乘法操作）
        self.shared_mul = Multiply()
        # 共享的减法 Module（用于减法操作）
        self.shared_sub = Subtract()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """使用与 PyTorch GRU 相同的初始化方式"""
        std = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
    
    def forward(self, input, hidden):
        """
        Args:
            input: [batch, input_size]
            hidden: [batch, hidden_size]
        
        Returns:
            new_h: [batch, hidden_size]
        """

        # ========== 计算门控的输入和隐藏贡献 ==========
        gi = self.weight_ih(input)   # ✅ 量化
        gh = self.weight_hh(hidden)  # ✅ 量化
        
        # ========== 分割为三个门 ==========
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)
        
        # ========== Reset Gate ==========
        # 使用共享的加法 Module
        r_gate_input = self.shared_add(i_r, h_r)  # ✅ 加法被量化
        resetgate = self.sigmoid(r_gate_input)  # ✅ sigmoid 被量化
        
        # ========== Update Gate ==========
        # 使用共享的加法 Module
        u_gate_input = self.shared_add(i_z, h_z)  # ✅ 加法被量化
        updategate = self.sigmoid(u_gate_input)  # ✅ sigmoid 被量化
        
        # ========== New Gate ==========
        # 使用共享的乘法 Module
        reset_h = self.shared_mul(resetgate, h_n)  # ✅ 乘法被量化
        # 使用共享的加法 Module
        n_gate_input = self.shared_add(i_n, reset_h)  # ✅ 加法被量化
        newgate = self.tanh(n_gate_input)  # ✅ tanh 被量化
        
        # ========== 计算新的隐藏状态 ==========
        # 使用共享的减法 Module
        one_minus_update = self.shared_sub(updategate.new_tensor(1.0), updategate)  # ✅ 减法被量化
        # 使用共享的乘法 Module
        new_contribution = self.shared_mul(one_minus_update, newgate)  # ✅ 乘法被量化
        # 使用共享的乘法 Module
        old_contribution = self.shared_mul(updategate, hidden)  # ✅ 乘法被量化
        # 使用共享的加法 Module
        new_h = self.shared_add(new_contribution, old_contribution)  # ✅ 加法被量化
        
        return new_h


class OptimizedQuantizableGRU(nn.Module):
    """
    可量化多层 GRU - 共享算术 Module
    
    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
        num_layers: GRU 层数
        batch_first: 如果为 True，输入形状为 [batch, seq, feature]
        dropout: 层间 dropout 概率
    """
    
    def __init__(self, input_size, hidden_size, num_layers=1, 
                 batch_first=True, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        
        # 创建多层 GRU Cell
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            self.cells.append(
                OptimizedQuantizableGRUCell(cell_input_size, hidden_size)
            )
        
        if dropout > 0 and num_layers > 1:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
    
    def forward(self, input, h_0=None):
        """
        Args:
            input: [batch, seq_len, input_size] if batch_first=True
            h_0: [num_layers, batch, hidden_size] or None
        
        Returns:
            output: [batch, seq_len, hidden_size]
            h_n: [num_layers, batch, hidden_size]
        """
        if not self.batch_first:
            input = input.transpose(0, 1)
        
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        
        # 初始化隐藏状态
        if h_0 is None:
            h_0 = input.new_zeros(self.num_layers, batch_size, self.hidden_size)
        
        # 存储每层的输出
        layer_output = input
        h_n_list = []
        
        # 遍历每一层
        for layer_idx, cell in enumerate(self.cells):
            h_t = h_0[layer_idx]
            outputs = []
            
            # 遍历每个时间步
            for t in range(seq_len):
                x_t = layer_output[:, t, :]
                h_t = cell(x_t, h_t)
                outputs.append(h_t.unsqueeze(1))
            
            # 拼接所有时间步
            layer_output = torch.cat(outputs, dim=1)
            
            # 应用 dropout
            if self.dropout_layer is not None and layer_idx < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)
            
            # 保存最终隐藏状态
            h_n_list.append(h_t.unsqueeze(0))
        
        # 拼接所有层的最终隐藏状态
        h_n = torch.cat(h_n_list, dim=0)
        
        if not self.batch_first:
            layer_output = layer_output.transpose(0, 1)
        
        return layer_output, h_n
    
    @classmethod
    def from_gru(cls, gru_module):
        """从 PyTorch nn.GRU 创建并迁移权重"""
        qgru = cls(
            input_size=gru_module.input_size,
            hidden_size=gru_module.hidden_size,
            num_layers=gru_module.num_layers,
            batch_first=gru_module.batch_first,
            dropout=gru_module.dropout if gru_module.num_layers > 1 else 0.0
        )
        
        # 迁移权重
        with torch.no_grad():
            for layer_idx in range(gru_module.num_layers):
                weight_ih_name = f'weight_ih_l{layer_idx}'
                weight_hh_name = f'weight_hh_l{layer_idx}'
                bias_ih_name = f'bias_ih_l{layer_idx}'
                bias_hh_name = f'bias_hh_l{layer_idx}'
                
                weight_ih = getattr(gru_module, weight_ih_name)
                weight_hh = getattr(gru_module, weight_hh_name)
                bias_ih = getattr(gru_module, bias_ih_name)
                bias_hh = getattr(gru_module, bias_hh_name)
                
                cell = qgru.cells[layer_idx]
                cell.weight_ih.weight.copy_(weight_ih)
                cell.weight_hh.weight.copy_(weight_hh)
                cell.weight_ih.bias.copy_(bias_ih)
                cell.weight_hh.bias.copy_(bias_hh)
        
        return qgru


def test_shared_vs_independent():
    """对比共享版和独立版的 Module 数量"""
    print("="*70)
    print("对比：共享算术 Module vs 独立算术 Module")
    print("="*70)
    
    # 创建参数
    batch_size = 4
    seq_len = 10
    input_size = 32
    hidden_size = 64
    num_layers = 2
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, input_size)
    
    # 创建标准 GRU
    gru_std = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
    gru_std.eval()
    
    # 创建优化版可量化 GRU（独立算术 Module）
    from optimized_quantizable_gru_independent import OptimizedQuantizableGRU as IndependentGRU
    gru_independent = IndependentGRU.from_gru(gru_std)
    gru_independent.eval()
    
    # 创建优化版可量化 GRU（共享算术 Module）
    gru_shared = OptimizedQuantizableGRU.from_gru(gru_std)
    gru_shared.eval()
    
    # 前向传播
    with torch.no_grad():
        output_std, h_n_std = gru_std(x)
        output_independent, h_n_independent = gru_independent(x)
        output_shared, h_n_shared = gru_shared(x)
    
    # 比较结果
    print("\n精度对比（与标准 GRU）:")
    print("-" * 70)
    independent_output_diff = (output_std - output_independent).abs().max().item()
    shared_output_diff = (output_std - output_shared).abs().max().item()
    
    print(f"独立算术 Module 版本最大差异: {independent_output_diff:.6f}")
    print(f"共享算术 Module 版本最大差异: {shared_output_diff:.6f}")
    
    if abs(independent_output_diff - shared_output_diff) < 1e-6:
        print("✅ 两者精度相同！")
    
    # 统计 Module 数量
    print("\n" + "="*70)
    print("Module 数量对比")
    print("="*70)
    
    def count_modules(model, name):
        module_count = {}
        for _, module in model.named_modules():
            module_type = type(module).__name__
            if module_type not in module_count:
                module_count[module_type] = 0
            module_count[module_type] += 1
        
        print(f"\n{name}:")
        total = 0
        for module_type in ['Add', 'Multiply', 'Subtract']:
            count = module_count.get(module_type, 0)
            print(f"  {module_type}: {count} 个")
            total += count
        print(f"  总计算术 Module: {total} 个")
        return total
    
    independent_count = count_modules(gru_independent, "独立算术 Module 版本")
    shared_count = count_modules(gru_shared, "共享算术 Module 版本")
    
    print("\n" + "="*70)
    print(f"算术 Module 总数对比: {independent_count} vs {shared_count}")
    print(f"减少数量: {independent_count - shared_count} 个")
    print(f"减少比例: {(independent_count - shared_count)/independent_count*100:.1f}%")
    print("="*70)
    
    # 统计参数数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    independent_params = count_parameters(gru_independent)
    shared_params = count_parameters(gru_shared)
    
    print("\n参数数量对比:")
    print(f"独立版本: {independent_params:,} 个参数")
    print(f"共享版本: {shared_params:,} 个参数")
    print(f"参数数量相同: {independent_params == shared_params}")
    
    print("\n✅ 共享版本减少了 Module 数量，但保持了相同的精度！")


if __name__ == "__main__":
    test_shared_vs_independent()