import glob
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# 1. 定位事件文件
evt_files = glob.glob("/root/project3/output/events.out.tfevents.*")
if not evt_files:
    raise FileNotFoundError("找不到任何 TensorBoard 事件文件，请确认路径。")
evt_path = evt_files[0]
print(f"使用事件文件: {evt_path}")

# 2. 加载事件文件
ea = event_accumulator.EventAccumulator(evt_path, size_guidance={event_accumulator.SCALARS: 0})
ea.Reload()

# 3. 列出所有可用标量标签，帮助确认正确tag
tags = ea.Tags().get('scalars', [])
print("可用的标量标签：")
for t in tags:
    print("  ", t)

# 3. 列出可用标签
tags = ea.Tags().get('scalars', [])
print("可用的标量标签：", tags)

# 4. 直接用 train/loss_viewpoint … 这两个
train_view_l1   = ea.Scalars('train/loss_viewpoint - l1_loss')
train_view_psnr = ea.Scalars('train/loss_viewpoint - psnr')

# 绘制 L1 Loss
plt.figure(figsize=(6,4))
steps = [s.step  for s in train_view_l1]
vals  = [s.value for s in train_view_l1]
plt.plot(steps, vals, label='train eval L1 loss')
plt.xlabel('Iteration'); plt.ylabel('L1 Loss')
plt.title('Viewpoint L1 Loss')
plt.legend(); plt.tight_layout()
plt.savefig('eval_l1_loss.png'); plt.close()

# 绘制 PSNR
plt.figure(figsize=(6,4))
steps = [s.step  for s in train_view_psnr]
vals  = [s.value for s in train_view_psnr]
plt.plot(steps, vals, label='train eval PSNR', color='orange')
plt.xlabel('Iteration'); plt.ylabel('PSNR (dB)')
plt.title('Viewpoint PSNR')
plt.legend(); plt.tight_layout()
plt.savefig('eval_psnr.png'); plt.close()