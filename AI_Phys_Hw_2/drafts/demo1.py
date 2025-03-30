"""
Müller-Brown势能神经网络回归模型
"""

# 导入必要库
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# %% [1] 定义Müller-Brown势能函数
def muller_brown(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    计算Müller-Brown势能值（自动广播机制）

    参数：
        x1: x1坐标值数组，形状(N,)
        x2: x2坐标值数组，形状(N,)

    返回：
        U: 势能值数组，形状(N,)
    """
    s = 0.05
    A = np.array([-200, -100, -170, 15], dtype=np.float32)  # 振幅系数
    alpha = np.array([-1, -1, -6.5, 0.7], dtype=np.float32)  # 二次项参数α
    beta = np.array([0, 0, 11, 0.6], dtype=np.float32)  # 二次项参数β
    gamma = np.array([-10, -10, -6.5, 0.7], dtype=np.float32)  # 二次项参数γ
    a = np.array([1, 0, -0.5, -1], dtype=np.float32)  # 中心坐标a
    b = np.array([0, 0.5, 1.5, 1], dtype=np.float32)  # 中心坐标b

    # 将输入转换为二维数组便于广播计算
    x1 = np.asarray(x1, dtype=np.float32)
    x2 = np.asarray(x2, dtype=np.float32)
    x1 = x1.reshape(-1, 1)  # 形状变为(N,1)
    x2 = x2.reshape(-1, 1)  # 形状变为(N,1)

    # 计算每个高斯项的指数部分
    dx1 = x1 - a  # 广播计算，形状(N,4)
    dx2 = x2 - b  # 广播计算，形状(N,4)

    # 指数项计算（逐元素运算）
    exponents = alpha * dx1**2 + beta * dx1 * dx2 + gamma * dx2**2  # 形状(N,4)

    # 求和并应用缩放因子
    U = s * np.sum(A * np.exp(exponents), axis=1)  # 形状(N,)

    # 应用势能截断
    return np.minimum(U, 9.0)


# %% [2] 数据准备
# 加载原始训练数据（强制转换为float32）
data = np.loadtxt("train_data.txt").astype(np.float32)  # 形状(500,3)
x1_orig = data[:, 0]  # 第1列为x1，形状(500,)
x2_orig = data[:, 1]  # 第2列为x2，形状(500,)
U_orig = data[:, 2]  # 第3列为U，形状(500,)

# 生成额外数据（定义域内均匀采样）
n_samples = 5000  # 新增样本量
# 在x1的定义域[-1.5, 1]内生成随机数（显式指定float32）
x1_new = np.random.uniform(low=-1.5, high=1.0, size=n_samples).astype(np.float32)
# 在x2的定义域[-0.5, 2]内生成随机数
x2_new = np.random.uniform(low=-0.5, high=2.0, size=n_samples).astype(np.float32)
# 计算势能值（自动处理广播）
U_new = muller_brown(x1_new, x2_new)  # 形状(5000,)

# 合并原始数据与新生成数据（保持float32类型）
X = np.hstack(
    [
        np.concatenate([x1_orig, x1_new])[:, np.newaxis],  # 列向量 (5500,1)
        np.concatenate([x2_orig, x2_new])[:, np.newaxis],  # 列向量 (5500,1)
    ]
)  # 合并后形状(5500,2)
y = np.concatenate([U_orig, U_new])  # 形状(5500,)

# %% [3] 数据预处理
# 划分训练集与验证集（8:2比例）
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42  # 固定随机种子保证可复现性
)

# 计算标准化参数（基于训练集）
mean = X_train.mean(axis=0).astype(np.float32)  # 各特征均值，形状(2,)
std = X_train.std(axis=0).astype(np.float32)  # 各特征标准差，形状(2,)


# %% [4] 神经网络模型定义
class MB_Net(nn.Module):
    """Müller-Brown势能回归神经网络"""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        """
        初始化模型

        参数：
            mean: 标准化均值，形状(2,)
            std:  标准化标准差，形状(2,)
        """
        super().__init__()

        # 注册标准化参数为buffer（固定为float32）
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32)  # 必须显式指定类型
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32)  # 防止自动类型推断错误
        )

        # 网络层定义
        self.fc1 = nn.Linear(2, 128)  # 输入层→隐藏层1
        self.fc2 = nn.Linear(128, 64)  # 隐藏层1→隐藏层2
        self.fc3 = nn.Linear(64, 1)  # 隐藏层2→输出层
        self.dropout = nn.Dropout(0.1)  # 随机失活层（防止过拟合）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数：
            x: 输入张量，形状(B,2)，B为批量大小

        返回：
            预测势能值，形状(B,1)
        """
        # 输入标准化（自动广播）
        x = (x - self.mean) / self.std

        # 隐藏层1（ReLU激活）
        x = F.relu(self.fc1(x))  # 输出形状(B,128)

        # 随机失活
        x = self.dropout(x)

        # 隐藏层2（ReLU激活）
        x = F.relu(self.fc2(x))  # 输出形状(B,64)

        # 输出层（线性激活）
        return self.fc3(x)  # 输出形状(B,1)


# %% [5] 初始化训练组件
# 实例化模型（传入标准化参数）
model = MB_Net(mean, std)

# 损失函数：均方误差（MSE）
criterion = nn.MSELoss()

# 优化器：Adam（初始学习率0.001）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器（验证损失稳定时自动降低学习率）
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",  # 监控验证损失最小值
    factor=0.5,  # 学习率衰减因子
    patience=5,  # 等待5个epoch无改善
    verbose=True,  # 打印调整信息
)

# %% [6] 数据加载器
# 将numpy数组转换为torch张量（显式指定float32类型）
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# 创建训练集DataLoader（批大小64，打乱顺序）
train_loader = DataLoader(
    dataset=TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True
)

# 创建验证集DataLoader（批大小64）
val_loader = DataLoader(
    dataset=TensorDataset(X_val_tensor, y_val_tensor), batch_size=64
)

# %% [7] 训练循环
best_val_loss = float("inf")  # 最佳验证损失初始值
patience = 10  # 早停等待周期
counter = 0  # 无改善计数器

# 训练轮次（最大1000轮）
for epoch in range(1000):
    # === 训练阶段 ===
    model.train()  # 训练模式（启用Dropout）
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(X_batch).squeeze()  # 输出形状(B,)

        # 计算损失
        loss = criterion(outputs, y_batch)

        # 反向传播
        loss.backward()

        # 参数更新
        optimizer.step()

        # 累计损失
        train_loss += loss.item()

    # 计算平均训练损失
    train_loss /= len(train_loader)

    # === 验证阶段 ===
    model.eval()  # 评估模式（禁用Dropout）
    val_loss = 0.0

    with torch.no_grad():  # 禁用梯度计算
        for X_batch, y_batch in val_loader:
            # 前向传播
            outputs = model(X_batch).squeeze()

            # 计算损失
            val_loss += criterion(outputs, y_batch).item()

    # 计算平均验证损失
    val_loss /= len(val_loader)

    # 更新学习率
    scheduler.step(val_loss)

    # === 早停机制 ===
    if val_loss < best_val_loss:
        # 保存最佳模型
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        counter = 0  # 重置计数器
    else:
        counter += 1
        # 达到耐心值则提前终止
        if counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # 打印训练进度（每10轮）
    if (epoch + 1) % 10 == 0:
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d} | "
            f"Train Loss: {train_loss:.4e} | "
            f"Val Loss: {val_loss:.4e} | "
            f"LR: {lr:.2e}"
        )

# %% [8] 加载最佳模型并评估
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# 计算训练集MAE
with torch.no_grad():
    # 预测整个训练集
    train_preds = model(X_train_tensor).squeeze()
    # 计算MAE
    mae = F.l1_loss(train_preds, y_train_tensor)

print(f"\nFinal Training MAE: {mae.item():.4f}")

# %% [9] 保存最终模型（包含标准化参数）
torch.save(model.state_dict(), "model.pth")
