import glob
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# 1. 自动查找 TensorBoard 事件文件
event_files = glob.glob("log/**/*.tfevents.*", recursive=True)
if not event_files:
    raise FileNotFoundError("在 logs 目录下未找到任何 events.out.tfevents 文件，请确认日志路径是否正确。")
evt_path = event_files[0]

# 2. 加载事件文件
ea = event_accumulator.EventAccumulator(
    evt_path,
    size_guidance={event_accumulator.SCALARS: 0}
)
ea.Reload()

# 3. 提取 PSNR 标量
train_psnr = ea.Scalars('train/psnr')
test_psnr  = ea.Scalars('test/psnr')

steps_train = [s.step for s in train_psnr]
vals_train  = [s.value for s in train_psnr]
steps_test  = [s.step for s in test_psnr]
vals_test   = [s.value for s in test_psnr]

# 4. 绘制 PSNR 曲线并保存
plt.figure()
plt.plot(steps_train, vals_train, label='train/psnr')
plt.xlabel('Iteration')
plt.ylabel('PSNR (dB)')
plt.title('Training vs Test PSNR')
plt.legend()
plt.tight_layout()
plt.savefig('psnr_curves.png', dpi=300)
plt.show()
