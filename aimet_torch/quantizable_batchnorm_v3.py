"""
可量化 BatchNorm2d V3 - 使用函数式 API（保留层级名称）

关键改进：
- ❌ 不使用 AIMET 自定义模块（Mean, Var 等）
- ✅ 使用 PyTorch 函数式 API（torch.mean, torch.var 等）
- ✅ 让 model_preparer 自动转换函数式操作为模块
- ✅ 保留模块层级名称（如 pre_bn.module_mean）
"""

import torch
import torch.nn as nn


class QuantizableBatchNorm2d(nn.Module):
    """
    可量化的 BatchNorm2d - 使用函数式 API 版本（保留层级）
    
    🎯 设计原则（模仿 CLN 模块）：
    - ✅ 使用 torch 函数式 API（torch.mean, torch.var, torch.sqrt 等）
    - ✅ 不注册子模块，让 model_preparer 自动转换
    - ✅ 放在 stateless_modules_to_preserve 中会保留层级名称
    - ✅ 转换后的模块会带有父模块前缀（如 pre_bn.module_mean）
    
    使用方法：
    1. 在 stateless_modules_to_preserve 中添加此类
    2. AIMET 会自动将函数式操作转换为模块
    3. 保留层级名称（pre_bn.module_mean 而不是 module_mean）
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, 
                 affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        # 可学习参数
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        # Running statistics
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置参数"""
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def forward(self, input):
        """
        前向传播 - 使用 torch 函数式 API
        
        ✅ 关键：使用 torch.xxx（函数式 API），而不是 self.xxx_op（子模块）
        ✅ model_preparer 会自动将这些操作转换为模块，并保留父模块前缀
        """
        # 训练模式：计算批次统计量
        if self.training:
            # ✅ 使用 torch 函数式 API
            batch_mean = torch.mean(input, dim=(0, 2, 3), keepdim=True)  # [1, C, 1, 1]
            batch_var = torch.var(input, dim=(0, 2, 3), keepdim=True, unbiased=False)  # [1, C, 1, 1]
            
            # 更新 running statistics
            if self.track_running_stats:
                with torch.no_grad():
                    batch_mean_flat = batch_mean.squeeze()
                    batch_var_flat = batch_var.squeeze()
                    
                    self.running_mean = (1 - self.momentum) * self.running_mean + \
                                       self.momentum * batch_mean_flat
                    self.running_var = (1 - self.momentum) * self.running_var + \
                                      self.momentum * batch_var_flat
                    self.num_batches_tracked += 1
            
            mean = batch_mean
            var = batch_var
        
        # 推理模式：使用 running statistics
        else:
            if not self.track_running_stats:
                raise RuntimeError('Must have track_running_stats=True for eval mode')
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)
        
        # BatchNorm 计算 - ✅ 使用 torch 函数式 API
        x_centered = input - mean  # torch.sub
        
        var_eps = var + self.eps  # torch.add
        
        std = torch.sqrt(var_eps)  # torch.sqrt
        
        x_normalized = x_centered / std  # torch.div
        
        if self.affine:
            weight = self.weight.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            x_scaled = x_normalized * weight  # torch.mul
            output = x_scaled + bias  # torch.add
        else:
            output = x_normalized
        
        return output
    
    def extra_repr(self):
        return (
            f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, '
            f'affine={self.affine}, track_running_stats={self.track_running_stats}'
        )


class QuantizableBatchNorm1d(nn.Module):
    """
    可量化的 BatchNorm1d - 使用函数式 API 版本（保留层级）
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, 
                 affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        # 可学习参数
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        # Running statistics
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置参数"""
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def forward(self, input):
        """前向传播 - 使用 torch 函数式 API"""
        # 处理 2D 和 3D 输入
        if input.dim() == 2:
            mean_dim = 0
        elif input.dim() == 3:
            mean_dim = (0, 2)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {input.dim()}D")
        
        # 训练模式
        if self.training:
            # ✅ 使用 torch 函数式 API
            batch_mean = torch.mean(input, dim=mean_dim, keepdim=True)
            batch_var = torch.var(input, dim=mean_dim, keepdim=True, unbiased=False)
            
            # 更新 running statistics
            if self.track_running_stats:
                with torch.no_grad():
                    batch_mean_flat = batch_mean.squeeze()
                    batch_var_flat = batch_var.squeeze()
                    
                    self.running_mean = (1 - self.momentum) * self.running_mean + \
                                       self.momentum * batch_mean_flat
                    self.running_var = (1 - self.momentum) * self.running_var + \
                                      self.momentum * batch_var_flat
                    self.num_batches_tracked += 1
            
            mean = batch_mean
            var = batch_var
        
        # 推理模式
        else:
            if not self.track_running_stats:
                raise RuntimeError('Must have track_running_stats=True for eval mode')
            if input.dim() == 2:
                mean = self.running_mean.view(1, -1)
                var = self.running_var.view(1, -1)
            else:
                mean = self.running_mean.view(1, -1, 1)
                var = self.running_var.view(1, -1, 1)
        
        # BatchNorm 计算 - ✅ 使用 torch 函数式 API
        x_centered = input - mean
        
        var_eps = var + self.eps
        
        std = torch.sqrt(var_eps)
        
        x_normalized = x_centered / std
        
        if self.affine:
            if input.dim() == 2:
                weight = self.weight.view(1, -1)
                bias = self.bias.view(1, -1)
            else:
                weight = self.weight.view(1, -1, 1)
                bias = self.bias.view(1, -1, 1)
            
            x_scaled = x_normalized * weight
            output = x_scaled + bias
        else:
            output = x_normalized
        
        return output
    
    def extra_repr(self):
        return (
            f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, '
            f'affine={self.affine}, track_running_stats={self.track_running_stats}'
        )

