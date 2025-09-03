import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 1. 加载数据
data_files = {
    'rough_slope': 'sim_data_rough_slope.npz',
    'slope': 'sim_data_slope.npz',
    'step': 'sim_data_step.npz'
}

datasets = {}
for name, file in data_files.items():
    datasets[name] = np.load(file)

# 2. 绘制 linear_vel 和 pri_latents（每个维度单独一图，不同曲线用不同颜色）
dim_labels = ['X', 'Y', 'Z']  # 维度标签
curve_colors = {
    'linear_vel': 'blue',    # linear_vel 用蓝色
    'pri_latents': 'orange'  # pri_latents 用橙色
}

plt.figure(figsize=(18, 12))  # 调整画布大小

# 遍历三种地形和三个维度
for i, terrain in enumerate(datasets.keys(), 1):
    data = datasets[terrain]
    linear_vel = data['linear_vel']
    pri_latents = data['pri_latents']
    timesteps = data['timesteps']

    for dim in range(3):  # 遍历X/Y/Z三个维度
        plt.subplot(3, 3, (i-1)*3 + dim + 1)  # 3行3列的子图布局

        # 绘制当前维度的 linear_vel（蓝色实线）
        plt.plot(timesteps, linear_vel[:, dim],
                color=curve_colors['linear_vel'], linestyle='-',
                label=f'linear_vel ({dim_labels[dim]})')

        # 绘制当前维度的 pri_latents（橙色虚线）
        plt.plot(timesteps, pri_latents[:, dim],
                color=curve_colors['pri_latents'], linestyle='--',
                label=f'pri_latents ({dim_labels[dim]})')

        plt.xlabel('Timesteps')
        plt.ylabel('Value')
        plt.title(f'Terrain: {terrain} | Dim: {dim_labels[dim]}')
        plt.legend()

plt.tight_layout()
plt.savefig('linear_vel_vs_pri_latents_by_dim.png')
plt.show()

# 3. 绘制 hist_latents 的 t-SNE 图（保持不变）
plt.figure(figsize=(10, 8))
all_hist_latents = []
terrain_labels = []

for terrain, data in datasets.items():
    hist_latents = data['hist_latents']
    all_hist_latents.append(hist_latents)
    terrain_labels.extend([terrain] * len(hist_latents))

all_hist_latents = np.vstack(all_hist_latents)
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(all_hist_latents)

terrain_colors = {
    'rough_slope': 'red',
    'slope': 'green',
    'step': 'blue'
}

for terrain in datasets.keys():
    mask = np.array(terrain_labels) == terrain
    plt.scatter(
        tsne_results[mask, 0], tsne_results[mask, 1],
        color=terrain_colors[terrain], label=terrain, alpha=0.6
    )

plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Visualization of hist_latents (by Terrain)')
plt.legend()
plt.savefig('tsne_hist_latents.png')
plt.show()