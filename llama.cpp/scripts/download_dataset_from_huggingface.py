#!/usr/bin/env python3
"""
从 Hugging Face 下载数据集的脚本
直接下载原始数据文件
"""

import argparse
import sys
import os
import time
from pathlib import Path


def extract_available_splits(data_files):
    """从文件名中提取可用的 split 列表"""
    splits = set()
    # 常见的 split 名称
    split_patterns = ['train', 'test', 'validation', 'val', 'dev', 'eval', 'evaluation']
    
    for f in data_files:
        filename = Path(f).name.lower()
        # 匹配 split-*.parquet 或 split.parquet 格式
        for pattern in split_patterns:
            if filename.startswith(pattern + '-') or filename.startswith(pattern + '.'):
                # 标准化 split 名称
                if pattern == 'val':
                    splits.add('validation')
                elif pattern == 'dev':
                    splits.add('validation')
                else:
                    splits.add(pattern)
                break
    
    return sorted(list(splits))


def download_raw_files(dataset_name, files_dir, use_mirror=True, split=None):
    """直接下载原始数据文件（不使用 datasets 缓存）
    
    参数:
        dataset_name: 数据集名称
        files_dir: 保存目录
        use_mirror: 是否使用镜像源
        split: 要下载的数据集分割类型 (train/test/validation)，None 表示下载所有
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        
        # 设置镜像源（默认使用镜像源）
        if use_mirror:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
            endpoint = "https://hf-mirror.com"
        else:
            endpoint = os.getenv("HF_HUB_ENDPOINT") or os.getenv("HF_ENDPOINT") or "https://huggingface.co"
        
        print(f"✓ 使用端点: {endpoint}")
        print(f"正在下载原始文件到: {files_dir}")
        
        save_path = Path(files_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        api = HfApi(endpoint=endpoint)
        
        # 列出文件
        print("\n正在列出数据集文件...")
        try:
            files = api.list_repo_files(repo_id=dataset_name, repo_type="dataset")
            print(f"找到 {len(files)} 个文件")
        except Exception as e:
            print(f"⚠️  无法列出文件列表: {e}")
            if split:
                print(f"\n✗ 错误: 指定了 --split {split}，但无法列出文件来确定可用的分割")
                print("   无法仅下载指定的分割，脚本已退出")
                print("   提示: 如果不指定 --split，将下载整个数据集")
                sys.exit(1)
            print("  使用 snapshot_download 下载整个仓库...")
            downloaded_path = snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                local_dir=str(save_path),
                endpoint=endpoint
            )
            print(f"\n✓ 数据集已下载到: {downloaded_path}")
            # 清理缓存目录
            cache_dir = save_path / ".cache"
            if cache_dir.exists():
                import shutil
                try:
                    shutil.rmtree(cache_dir)
                    print(f"✓ 已清理缓存目录: {cache_dir}")
                except Exception as e:
                    print(f"⚠️  无法清理缓存目录: {e}")
            return
        
        # 下载所有文件（排除一些明显不需要的文件）
        # 排除 .lock 文件和其他系统文件
        excluded_extensions = ['.lock']
        excluded_patterns = ['.git/', '.gitignore', '.gitattributes']
        data_files = []
        for f in files:
            # 排除特定扩展名
            if any(f.endswith(ext) for ext in excluded_extensions):
                continue
            # 排除特定模式
            if any(pattern in f for pattern in excluded_patterns):
                continue
            data_files.append(f)
        
        # 如果指定了 split，过滤出对应 split 的文件
        if split:
            # Hugging Face 数据集文件命名通常为: split-00000-of-00001.parquet
            # 也支持其他命名方式，如 split.parquet, split.json 等
            split_lower = split.lower()
            filtered_files = []
            for f in data_files:
                filename = Path(f).name.lower()
                # 检查文件名是否以 split 开头（支持 train, test, validation, val, dev 等）
                if filename.startswith(split_lower + '-') or filename.startswith(split_lower + '.'):
                    filtered_files.append(f)
                # 也支持 val 和 dev 作为 validation 的简写
                elif split_lower == 'validation' and (
                    filename.startswith('val-') or filename.startswith('val.') or
                    filename.startswith('dev-') or filename.startswith('dev.')
                ):
                    filtered_files.append(f)
            
            if filtered_files:
                data_files = filtered_files
                print(f"✓ 已过滤出 {split} 分割的文件: {len(data_files)} 个")
            else:
                # 提取可用的 split 列表
                available_splits = extract_available_splits(data_files)
                
                print(f"\n✗ 错误: 未找到指定的分割 '{split}'")
                if available_splits:
                    print(f"   可用的分割: {', '.join(available_splits)}")
                else:
                    print(f"   无法从文件名中识别出分割信息")
                    print(f"   数据文件示例: {data_files[:5] if len(data_files) <= 5 else data_files[:5] + ['...']}")
                print(f"\n   提示: 如果不指定 --split，将下载所有数据文件")
                print(f"   或者使用 --split 参数指定上述可用的分割之一")
                sys.exit(1)
        
        if not data_files:
            if split:
                print(f"\n✗ 错误: 指定了 --split {split}，但未找到任何数据文件")
                print("   无法下载指定的分割，脚本已退出")
                print("   提示: 如果不指定 --split，将下载整个数据集")
                sys.exit(1)
            print("使用 snapshot_download 下载整个仓库...")
            snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                local_dir=str(save_path),
                endpoint=endpoint
            )
            # 清理缓存目录
            cache_dir = save_path / ".cache"
            if cache_dir.exists():
                import shutil
                try:
                    shutil.rmtree(cache_dir)
                    print(f"✓ 已清理缓存目录: {cache_dir}")
                except Exception as e:
                    print(f"⚠️  无法清理缓存目录: {e}")
            return
        
        print(f"\n开始下载 {len(data_files)} 个文件...")
        start_time = time.time()
        
        downloaded_files = []
        for file_path in data_files:
            try:
                print(f"  下载: {file_path}")
                downloaded_path = hf_hub_download(
                    repo_id=dataset_name,
                    filename=file_path,
                    repo_type="dataset",
                    local_dir=str(save_path),
                    endpoint=endpoint
                )
                downloaded_files.append(downloaded_path)
                file_size = Path(downloaded_path).stat().st_size / (1024 * 1024)  # MB
                print(f"  ✓ 完成 ({file_size:.2f} MB)")
            except Exception as e:
                print(f"  ✗ 失败: {e}")
                continue
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ 下载完成！耗时: {elapsed_time:.2f} 秒，成功: {len(downloaded_files)} 个文件")
        
        # 清理 huggingface_hub 创建的缓存目录
        cache_dir = save_path / ".cache"
        if cache_dir.exists():
            import shutil
            try:
                shutil.rmtree(cache_dir)
                print(f"✓ 已清理缓存目录: {cache_dir}")
            except Exception as e:
                print(f"⚠️  无法清理缓存目录: {e}")
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="从 Hugging Face 下载数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载数据集到当前目录下的数据集名称文件夹（如 ./competition_math）
  python download_dataset_from_huggingface.py --dataset qwedsacf/competition_math
  
  # 指定输出目录，会在该目录下创建数据集名称文件夹（如 ./my_data/competition_math）
  python download_dataset_from_huggingface.py --dataset qwedsacf/competition_math --output-dir ./my_data
  
  # 只下载测试集
  python download_dataset_from_huggingface.py --dataset qwedsacf/competition_math --split test
  
  # 只下载训练集
  python download_dataset_from_huggingface.py --dataset qwedsacf/competition_math --split train
  
  # 只下载验证集
  python download_dataset_from_huggingface.py --dataset qwedsacf/competition_math --split validation
  
  # 不使用镜像源（使用官方源）
  python download_dataset_from_huggingface.py --dataset qwedsacf/competition_math --no-mirror
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="数据集名称，格式: username/dataset_name"
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="不使用镜像源，使用官方 Hugging Face 源（默认使用镜像源）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认: 使用数据集名称）"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "validation", "val", "dev"],
        default=None,
        help="要下载的数据集分割类型 (train/test/validation/val/dev)，不指定则下载所有分割"
    )
    
    args = parser.parse_args()
    
    # 标准化 split 名称：val 和 dev 都映射为 validation
    if args.split:
        if args.split in ['val', 'dev']:
            args.split = 'validation'
    
    # 从数据集名称中提取名称部分（去掉用户名）
    dataset_dir = args.dataset.split('/')[-1]
    
    # 确定最终输出目录
    if args.output_dir is None:
        # 如果没有指定目录，使用当前目录下的数据集名称文件夹
        final_output_dir = f"./{dataset_dir}"
    else:
        # 如果指定了目录，在该目录下创建数据集名称的文件夹
        final_output_dir = str(Path(args.output_dir) / dataset_dir)
    
    # 设置镜像源（默认使用，除非指定 --no-mirror）
    use_mirror = not args.no_mirror
    
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
    
    # 直接下载原始文件
    download_raw_files(args.dataset, final_output_dir, use_mirror, args.split)


if __name__ == "__main__":
    main()
