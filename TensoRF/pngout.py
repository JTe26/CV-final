import json
import os

# 1. 读
with open('transforms.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 处理：递归地把所有 file_path 去掉 .png
def strip_png(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == 'file_path' and isinstance(v, str) and v.endswith('.png'):
                obj[k] = v[:-4]
            else:
                strip_png(v)
    elif isinstance(obj, list):
        for e in obj:
            strip_png(e)

strip_png(data)

# 3. 写回（覆盖原文件）
with open('transforms.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Done: all .png suffix removed from file_path.")
