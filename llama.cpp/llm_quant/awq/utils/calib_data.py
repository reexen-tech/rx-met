import torch
from datasets import load_dataset


def get_calib_dataset(
    data="pileval",
    tokenizer=None,
    n_samples=512,
    block_size=512,
    exact_blocks=False,
):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    n_tokens = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        n_tokens += sample.numel()
        if exact_blocks and n_tokens >= n_samples * block_size:
            break
        if not exact_blocks and n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    blocks = [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]
    return blocks[:n_samples] if exact_blocks else blocks
