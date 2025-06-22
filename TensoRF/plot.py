import glob
import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# 1. 自动查找 TensorBoard 事件文件
event_files = glob.glob("log/**/*.tfevents.*", recursive=True)
if not event_files:
    raise FileNotFoundError("在 logs 目录下未找到任何 events.out.tfevents 文件，请确认日志路径是否正确。")

evt_path = event_files[0]
print(f"使用事件文件: {evt_path}")

# 2. 加载事件文件
ea = event_accumulator.EventAccumulator(
    evt_path,
    size_guidance={
        event_accumulator.SCALARS: 0,  # load all scalar data
    }
)
ea.Reload()

# 3. 提取训练和测试损失标量
# 训练 loss：train/total_loss
# 测试 loss：test/mse
train_loss = ea.Scalars('train/total_loss')


# 4. 准备绘图数据
steps_train = [s.step for s in train_loss]
vals_train  = [s.value for s in train_loss]


# 5. 绘制损失曲线
plt.figure()
plt.plot(steps_train, vals_train, label='train/total_loss')

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss vs. Test Loss')
plt.legend()
plt.tight_layout()
plt.savefig('loss_curves.png', dpi=300)
plt.show()
