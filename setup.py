# -*- coding: utf-8 -*-
"""
AIMET RX - AI Model Efficiency Toolkit (Redistributable Package)

This package includes aimet_common, aimet_onnx, and aimet_torch modules.
"""

from setuptools import setup, find_packages
import os

# 读取版本号
version = "1.0.2"

# 读取依赖
def parse_requirements(filename):
    """解析 requirements.txt 文件"""
    if not os.path.exists(filename):
        return []
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="aimet-rx",
    version=version,
    description="AI Model Efficiency Toolkit - Redistributable Package",
    long_description=open('README.md', encoding='utf-8').read() if os.path.exists('README.md') else '',
    long_description_content_type="text/markdown",
    author="Qualcomm Innovation Center",
    author_email="",
    url="https://github.com/quic/aimet",
    license="BSD-3-Clause",
    
    # 包配置
    packages=find_packages(include=['aimet_common', 'aimet_common.*', 
                                     'aimet_onnx', 'aimet_onnx.*',
                                     'aimet_torch', 'aimet_torch.*',
                                     'export_onnx_and_encodings', 'export_onnx_and_encodings.*']),
    
    # 包含所有Python文件
    package_data={
        'aimet_common': ['*.json', '*.so', '*.pyd', '*.dll', 'bin/*'],
        'aimet_onnx': ['*.json', '*.so', '*.pyd', '*.dll'],
        'aimet_torch': ['*.json', '*.so', '*.pyd', '*.dll'],
        'export_onnx_and_encodings': ['*.json', '*.pt', 'data/*'],
    },
    
    # 依赖项
    install_requires=parse_requirements('requirements.txt'),
    
    # Python版本要求
    python_requires='>=3.8',
    
    # 分类信息
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    
    # 关键词
    keywords='ai machine-learning model-optimization quantization compression',
    
    # 确保包含所有文件
    include_package_data=True,
    zip_safe=False,
)













