#!/usr/bin/env python3
import json
import random
from pathlib import Path

# 配置
project_root = Path("")
input_json = project_root / "transforms.json"
output_dir = project_root / "nerf_data"
output_dir.mkdir(parents=True, exist_ok=True)

# 固定随机种子
random.seed(20211202)

# 读取原始 transforms
with open(input_json, 'r') as f:
    meta = json.load(f)
frames = meta.get('frames', [])
if not frames:
    raise ValueError("transforms.json 中没有 frames 字段")

# 随机打乱
random.shuffle(frames)

# 按比例切分：80% train, 10% val, 10% test
N = len(frames)
# 保证至少1张用于 val/test
n_val = max(1, int(N * 0.1))
n_test = max(1, int(N * 0.1))
# 剩余用作训练
n_train = N - n_val - n_test

train_frames = frames[:n_train]
val_frames   = frames[n_train:n_train + n_val]
test_frames  = frames[n_train + n_val:]

# 输出文件函数
def write_split(frames_subset, split_name):
    out_meta = {k: v for k, v in meta.items() if k != 'frames'}
    out_meta['frames'] = frames_subset
    path = output_dir / f"transforms_{split_name}.json"
    with open(path, 'w') as f:
        json.dump(out_meta, f, indent=2)
    print(f"Wrote {path} ({len(frames_subset)} frames)")

# 依次写入\write_split(train_frames, 'train')
write_split(train_frames, 'train')
write_split(val_frames, 'val')
write_split(test_frames, 'test')

print(f"Split done: total={N}, train={len(train_frames)}, val={len(val_frames)}, test={len(test_frames)}")
