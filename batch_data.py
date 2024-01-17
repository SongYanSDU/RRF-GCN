import numpy as np


def get_balanced_batch(data, labels, N, M):
    # 确保每个类别中至少有M/N个样本
    samples_per_class = M // N
    batch_data = []
    batch_labels = []
    np.random.seed(1388)

    for class_idx in range(N):
        # 找到当前类别的所有样本
        class_data = data[labels == class_idx]
        class_labels = labels[labels == class_idx]

        # 从当前类别中随机选择samples_per_class个样本
        selected_indices = np.random.choice(len(class_data), samples_per_class, replace=False)
        selected_data = class_data[selected_indices]
        selected_labels = class_labels[selected_indices]

        # 将选中的样本添加到batch中
        batch_data.append(selected_data)
        batch_labels.append(selected_labels)

    # 将所有类别的样本合并成一个batch
    batch_data = np.vstack(batch_data)
    batch_labels = np.hstack(batch_labels)

    return batch_data, batch_labels

# 示例用法
# 假设 data 和 labels 已经定义，并且N（类别数）和M（batch大小）已给定
# balanced_data, balanced_labels = get_balanced_batch(data, labels, N, M)
