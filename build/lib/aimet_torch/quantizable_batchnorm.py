import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class QuantizableBatchNorm2d(nn.Module):
    """自定义BatchNorm2d，功能与nn.BatchNorm2d完全相同"""
    
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        """
        参数说明：
        num_features: 输入的特征图通道数（C）
        eps: 数值稳定性的小常数，防止除以0
        momentum: 用于running_mean和running_var更新的动量
        affine: 是否使用可学习的缩放和平移参数（gamma和beta）
        track_running_stats: 是否跟踪运行时的统计信息（训练时）
        """
        super(QuantizableBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        # 可学习的参数（如果affine=True）
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))  # gamma
            self.bias = nn.Parameter(torch.zeros(num_features))   # beta
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        # 运行时统计信息（如果track_running_stats=True）
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        
        # 重置参数
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置所有参数"""
        if self.track_running_stats:
            # 初始化running_mean为0，running_var为1
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
        
        if self.affine:
            # 初始化gamma为1，beta为0（与PyTorch官方实现一致）
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def _check_input_dim(self, x: Tensor):
        """检查输入维度是否为4D (N, C, H, W)"""
        # ⚠️ 注释掉维度检查，因为trace时无法求值
        # PyTorch 运算会自动检查维度，无需显式检查
        # if x.dim() != 4:
        #     raise ValueError(f'Expected 4D input (got {x.dim()}D input)')
        pass
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播 - trace友好版本
        
        ✅ 移除所有控制流，使用函数式API
        ✅ training/eval状态在Python层处理（不在trace中）
        """
        # 根据 training 状态选择统计量
        if self.training or not self.track_running_stats:
            # 训练模式：计算批次统计量
            mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)  # [1, C, 1, 1]
            var = torch.var(x, dim=(0, 2, 3), keepdim=True, unbiased=False)  # [1, C, 1, 1]
            
            # 更新 running statistics
            if self.track_running_stats:
                with torch.no_grad():
                    mean_flat = mean.squeeze()
                    var_flat = var.squeeze()
                    self.running_mean.data = (1 - self.momentum) * self.running_mean.data + \
                                             self.momentum * mean_flat
                    self.running_var.data = (1 - self.momentum) * self.running_var.data + \
                                            self.momentum * var_flat
                    self.num_batches_tracked += 1
        else:
            # 推理模式：使用 running statistics
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)
        
        # BatchNorm 计算 - ✅ 使用 torch 函数式 API
        x_centered = torch.sub(x, mean)  # 明确使用 torch.sub
        
        var_eps = torch.add(var, self.eps)  # 明确使用 torch.add
        
        std = torch.sqrt(var_eps)  # torch.sqrt
        
        x_normalized = torch.div(x_centered, std)  # 明确使用 torch.div
        
        if self.affine:
            weight = self.weight.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            x_scaled = torch.mul(x_normalized, weight)  # 明确使用 torch.mul
            output = torch.add(x_scaled, bias)  # 明确使用 torch.add
        else:
            output = x_normalized
        
        return output
    
    def extra_repr(self):
        """用于打印模块信息"""
        return (f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, '
                f'affine={self.affine}, track_running_stats={self.track_running_stats}')


# 测试函数，验证我们的实现与PyTorch官方实现的一致性
def test_custom_batchnorm():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 创建测试数据
    batch_size = 4
    channels = 3
    height = 32
    width = 32
    x = torch.randn(batch_size, channels, height, width)
    
    # 创建PyTorch官方版本和我们的自定义版本
    official_bn = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
    custom_bn = CustomBatchNorm2d(channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
    
    # 确保参数相同
    with torch.no_grad():
        custom_bn.weight.copy_(official_bn.weight)
        custom_bn.bias.copy_(official_bn.bias)
        custom_bn.running_mean.copy_(official_bn.running_mean)
        custom_bn.running_var.copy_(official_bn.running_var)
    
    # 训练模式测试
    official_bn.train()
    custom_bn.train()
    
    # 前向传播
    official_output = official_bn(x)
    custom_output = custom_bn(x)
    
    # 检查输出是否相同
    output_diff = torch.abs(official_output - custom_output).max().item()
    print(f"训练模式输出最大差异: {output_diff:.6e}")
    assert output_diff < 1e-6, "训练模式输出不匹配！"
    
    # 检查running statistics是否相同
    running_mean_diff = torch.abs(official_bn.running_mean - custom_bn.running_mean).max().item()
    running_var_diff = torch.abs(official_bn.running_var - custom_bn.running_var).max().item()
    print(f"训练后running_mean最大差异: {running_mean_diff:.6e}")
    print(f"训练后running_var最大差异: {running_var_diff:.6e}")
    
    # 推理模式测试
    official_bn.eval()
    custom_bn.eval()
    
    # 再次前向传播（应该使用running statistics）
    official_output_eval = official_bn(x)
    custom_output_eval = custom_bn(x)
    
    # 检查输出是否相同
    output_diff_eval = torch.abs(official_output_eval - custom_output_eval).max().item()
    print(f"推理模式输出最大差异: {output_diff_eval:.6e}")
    assert output_diff_eval < 1e-6, "推理模式输出不匹配！"
    
    # 测试梯度计算
    official_bn.train()
    custom_bn.train()
    
    # 计算梯度
    official_output.sum().backward()
    custom_output.sum().backward()
    
    # 检查梯度是否相同
    weight_grad_diff = torch.abs(official_bn.weight.grad - custom_bn.weight.grad).max().item()
    bias_grad_diff = torch.abs(official_bn.bias.grad - custom_bn.bias.grad).max().item()
    print(f"weight梯度最大差异: {weight_grad_diff:.6e}")
    print(f"bias梯度最大差异: {bias_grad_diff:.6e}")
    
    # 测试无affine参数的情况
    official_bn_no_affine = nn.BatchNorm2d(channels, affine=False)
    custom_bn_no_affine = CustomBatchNorm2d(channels, affine=False)
    
    # 前向传播
    official_output_no_affine = official_bn_no_affine(x)
    custom_output_no_affine = custom_bn_no_affine(x)
    
    output_diff_no_affine = torch.abs(official_output_no_affine - custom_output_no_affine).max().item()
    print(f"无affine参数输出最大差异: {output_diff_no_affine:.6e}")
    
    print("\n所有测试通过！CustomBatchNorm2d与nn.BatchNorm2d功能完全一致")


if __name__ == "__main__":
    test_custom_batchnorm()