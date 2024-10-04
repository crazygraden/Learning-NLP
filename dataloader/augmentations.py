import numpy as np
import torch
# import skfuzzy as fuzz
from sklearn.neighbors import NearestNeighbors

def fuzz_gaussmf(x, mean, sigma):
    return np.exp(-((x - mean)**2.) / (2 * sigma**2.))
    
def DataTransform(sample, config):
    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)
    return weak_aug, strong_aug

def DataTransform_fuzzy(sample, config):
    # Convert tensor to NumPy array
    sample_numpy = sample.numpy()
    # Initialize lists to store weak and strong augmented samples
    weak_aug_list = []
    strong_aug_list = []

    # Apply augmentation to each sample
    for i in range(sample_numpy.shape[0]):
        weak_aug_sample = type1fuzzy(sample_numpy[i])
        strong_aug_sample = intervalt2f(sample_numpy[i], config.augmentation.jitter_ratio)
        # strong_aug_sample = type1fuzzy(sample_numpy[i])
        weak_aug_list.append(weak_aug_sample)
        strong_aug_list.append(strong_aug_sample)

    # Convert augmented samples back to PyTorch tensors
    weak_aug = torch.tensor(weak_aug_list).numpy()
    strong_aug = torch.tensor(strong_aug_list).numpy()
    weak_aug = scaling(weak_aug, config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(strong_aug, max_segments=config.augmentation.max_seg),
                        config.augmentation.jitter_ratio)
    # print(weak_aug)
    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])
    # x = x.cpu().numpy()
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.array_split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate([np.random.permutation(split) for split in splits]).ravel()
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)
def add_gaussian_noise(time_series, mean=0.0, stddev=1.0):
     # Gaussian noise generation
    noise = np.random.normal(mean, stddev, len(time_series))

    # Adding noise to the original time series
    noisy_series = time_series + noise

    return noisy_series


def muti_SMOTE(X, y, k=3):
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    unique_classes = np.unique(y)  # 获取唯一类别
    class_counts = {cls: np.sum(y == cls) for cls in unique_classes}  # 每个类别的样本数
    # 找到最多的类别
    max_class_count = max(class_counts.values())
    # print(max_class_count)
    X_balanced = X.copy()
    y_balanced = y.copy()
    # 针对每一个类别执行 SMOTE，除了最多的类
    for cls in unique_classes:
        if class_counts[cls] < max_class_count:
            # 当前类别是少数类，执行SMOTE
            X_minority = X[y == cls]
            N_per_sample = (max_class_count - class_counts[cls]) // len(X_minority)
            # 初始化 KNN
            k_neighbors = min(k, len(X_minority) - 1)
            knn = NearestNeighbors(n_neighbors=k_neighbors)
            knn.fit(X_minority)

            synthetic_samples = []
            synthetic_labels = []
            for minority_sample in X_minority:
                # 找到最近邻
                distances, indices = knn.kneighbors(minority_sample.reshape(1, -1), n_neighbors=k_neighbors)
                # 最近邻和最远邻
                # nearest_neighbor_index = indices[0][np.argmin(distances)]  # 最近邻
                # farthest_neighbor_index = indices[0][np.argmin(distances)]  # 最远邻
                # nearest_neighbor = X_minority[nearest_neighbor_index]
                # farthest_neighbor = X_minority[farthest_neighbor_index]
                fuzzy1 = fuzz_gaussmf(np.argmin(distances), minority_sample, np.std(minority_sample))
                fuzzy2 = fuzz_gaussmf(np.argmin(distances), minority_sample, np.std(minority_sample))
                # 创建合成样本
                for _ in range(N_per_sample):
                    # neighbor_index = np.random.choice(indices[0])
                    # neighbor = X_minority[neighbor_index]
                    difference = fuzzy2 - fuzzy1
                    alpha = np.random.random()

                    # 生成合成样本，保持维度一致
                    synthetic_sample = minority_sample + alpha * difference
                    synthetic_samples.append(synthetic_sample.reshape(1, -1))  # 保证合成样本是2D的
                    synthetic_labels.append(cls)
            # 合并生成的样本
            synthetic_samples = np.vstack(synthetic_samples)  # 使用 vstack 保证维度正确
            synthetic_labels = np.array(synthetic_labels)
            X_balanced = np.concatenate((X_balanced, synthetic_samples), axis=0)
            y_balanced = np.concatenate((y_balanced, synthetic_labels), axis=0)
    return X_balanced, y_balanced

def type1fuzzy(x):
    min_x = np.min(x)
    max_x = np.max(x)
    x = (x - min_x) / (max_x - min_x)
    mu = np.mean(x, axis=None)
    sigma = np.std(x)
    aug_x = fuzz_gaussmf(x, mu, sigma)
    result = np.zeros_like(x)
    # 使用模糊规则进行去模糊化
    for i in range(len(x)):
        result[i] = aug_x[i] * x[i]
    # print(type(aug_x))
    # print(type(result))
    # return result
    return aug_x
def intervalt2f(x, jitter_ratio):
    # jitter_ratio = 0.3
    min_x = np.min(x)
    max_x = np.max(x)
    x = (x - min_x) / (max_x - min_x)
    result = np.zeros_like(x)
    # ========================= 1 =================================
    mu1 = np.mean(x, axis=None)
    sigma1 = np.std(x)
    mu2 = np.mean(add_gaussian_noise(x, mean=0.2, stddev=0.05), axis=None)
    sigma2 = np.std(add_gaussian_noise(x, mean=0.2, stddev=0.05))
    upper = fuzz_gaussmf(x, mu1, sigma1)
    lower = fuzz_gaussmf(x, mu2, sigma2)
    ## 使用模糊规则进行去模糊化
    for i in range(len(x)):
        upper[i] = upper[i] * x[i]
        lower[i] = lower[i] * x[i]
        result[i] = (upper[i] + lower[i]) / 2
    return result
    # # ========================= 2 =================================
    # mu2 = np.mean(add_gaussian_noise(x, mean=0.2, stddev=0.05), axis=None)
    # sigma2 = np.std(add_gaussian_noise(x, mean=0.2, stddev=0.05))
    # mu_min = min(mu1, mu2)
    # mu_max = max(mu1, mu2)
    # aug_x = fuzz.gauss2mf(x[0], mu_min, sigma1, mu_max, sigma2)
    # # 使用模糊规则进行去模糊化
    # for i in range(len(x)):
    #     result[i] = aug_x[i] * x[i]
    # # print(type(aug_x))
    # # print(type(result))
    # # return result
    # return aug_x
