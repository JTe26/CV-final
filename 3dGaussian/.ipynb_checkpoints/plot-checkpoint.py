import glob
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# 1. 定位事件文件（根据实际目录修改）
evt_files = glob.glob("/root/project3/output/events.out.tfevents.*")
evt_path = evt_files[0]  # 取第一个

# 2. 加载并解析
ea = event_accumulator.EventAccumulator(
    evt_path,
    size_guidance={event_accumulator.SCALARS: 0}
)
ea.Reload()

# 3. 提取标量
train_loss = ea.Scalars('train_loss_patches/total_loss')
test_l1    = ea.Scalars('test/loss_viewpoint - l1_loss')
test_psnr  = ea.Scalars('test/loss_viewpoint - psnr')

# --- 绘制 Loss 图 ---
# 准备数据
steps_train = [s.step  for s in train_loss]
vals_train  = [s.value for s in train_loss]
steps_test  = [s.step  for s in test_l1]
vals_test   = [s.value for s in test_l1]

plt.figure(figsize=(6,4))
plt.plot(steps_train, vals_train, label='train total_loss')
plt.plot(steps_test,  vals_test,  label='test L1 loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Train vs Test Loss')
plt.legend()
plt.tight_layout()
plt.savefig('gaussian_loss.png', dpi=300)
plt.close()

# --- 绘制 PSNR 图 ---
steps_psnr = [s.step  for s in test_psnr]
vals_psnr  = [s.value for s in test_psnr]

plt.figure(figsize=(6,4))
plt.plot(steps_psnr, vals_psnr, label='test PSNR', color='orange')
plt.xlabel('Iteration')
plt.ylabel('PSNR (dB)')
plt.title('Test PSNR')
plt.legend()
plt.tight_layout()
plt.savefig('gaussian_psnr.png', dpi=300)
plt.close()

print("Saved gaussian_loss.png and gaussian_psnr.png")
