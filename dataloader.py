import scipy.io as iso
import numpy as np
import torch
from torch.utils.data import DataLoader
import random

batch_size = 30

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, images, valence_labels, dominance_labels, arousal_labels):
        self.images = images
        self.valence_labels = valence_labels
        self.dominance_labels = dominance_labels
        self.arousal_labels = arousal_labels

    def __getitem__(self, index):
        img = self.images[index]
        valence_labels = self.valence_labels[index]
        dominance_labels = self.dominance_labels[index]
        arousal_labels = self.arousal_labels[index]
        return img, valence_labels, dominance_labels, arousal_labels

    def __len__(self):
        return len(self.images)

def load_data(input_file="s01"): #读取数据
    dataset_dir = "E:/Data/DEAP/1D_DE_with-two classes/"
    data_file = iso.loadmat(dataset_dir + "DE_"+input_file + ".mat")
    data = data_file["data"]

    valence_labels = data_file["valence_labels"]
    valence_labels = np.squeeze(valence_labels)
    dominance_labels = data_file["dominance_labels"]
    dominance_labels = np.squeeze(dominance_labels)
    arousal_labels = data_file["arousal_labels"]
    arousal_labels = np.squeeze(arousal_labels)
    return data, valence_labels, dominance_labels, arousal_labels

def dataloader_maker(dataset_name):
    data, valence_labels, dominance_labels, arousal_labels = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(
    dataset=MyDataset(data, valence_labels, dominance_labels, arousal_labels),
    batch_size=batch_size,
    drop_last=True,
    shuffle=True,
    num_workers=8)
    return dataloader


def disorganize_data(dataset_name):  # 打乱数据

    data, valence_labels, dominance_labels, arousal_labels = load_data(dataset_name)
    n = len(data)
    d_data = torch.Tensor(n, 128)
    d_valence_labels = torch.LongTensor(n)
    d_dominance_labels = torch.LongTensor(n)
    d_arousal_labels = torch.LongTensor(n)

    random_index = torch.randperm(n)
    for i, index in enumerate(random_index):
        d_data[i] = torch.tensor(data[index])
        d_valence_labels[i] = torch.tensor(valence_labels[index])
        d_dominance_labels[i] = torch.tensor(dominance_labels[index])
        d_arousal_labels[i] = torch.tensor(arousal_labels[index])
    return d_data, d_valence_labels, d_dominance_labels, d_arousal_labels

def generate_balanced_samples(sample_num, dataset_name): # 平衡样本组，使0/1标签数量平衡
    # sample_num指的是一个0/1标签的数量，也就是说实际上一个样本组有2*sample_num个样本
    data, valence_labels, dominance_labels, arousal_labels = load_data(dataset_name)

    temp_data, temp_v_label, temp_d_label, temp_a_label = [], [], [], []
    counter = 2 * [sample_num]  # 计数器[n,n]，收集2种样本各n个
    i = 0
    while True:
        if len(temp_data) == sample_num * 2:  #目的是采集n*2个样本
            break
        if counter[int(valence_labels[i])] > 0:
            temp_data.append(data[i])
            temp_v_label.append(valence_labels[i])
            temp_d_label.append(dominance_labels[i])
            temp_a_label.append(arousal_labels[i])
            counter[int(valence_labels[i])] -= 1
        i += 1

    assert (len(temp_data) == sample_num * 2)
    data_balanced = torch.from_numpy(np.array(temp_data))
    v_label_selected = torch.from_numpy(np.array(temp_v_label))
    d_label_selected = torch.from_numpy(np.array(temp_d_label))
    a_label_selected = torch.from_numpy(np.array(temp_a_label))

    return data_balanced, v_label_selected, d_label_selected, a_label_selected


"""
G1: a pair of pic comes from same domain ,same class
G3: a pair of pic comes from same domain, different classes

G2: a pair of pic comes from different domain,same class
G4: a pair of pic comes from different domain, different classes
"""

def generate_groups2(source_name, target_name):
    # 加载源域和目标域的数据
    source_data, source_v, source_d, source_a = load_data(source_name)
    target_data, target_v, target_d, target_a = load_data(target_name)

    # 初始化组和标签列表
    G1, G2, G3, G4, G5, G6 = [], [], [], [], [], []
    Lv1, Lv2, Lv3, Lv4, Lv5, Lv6 = [], [], [], [], [], []
    Ld1, Ld2, Ld3, Ld4, Ld5, Ld6 = [], [], [], [], [], []
    La1, La2, La3, La4, La5, La6 = [], [], [], [], [], []

    # 遍历所有数据对
    for i in range(len(source_data)):
        for j in range(len(target_data)):
            # 第一组：两个数据都来自源域
            G1.append([source_data[i], source_data[j]])
            Lv1.append([source_v[i], source_v[j]])
            Ld1.append([source_d[i], source_d[j]])
            La1.append([source_a[i], source_a[j]])

            # 第二组：第一个数据来自源域，第二个来自目标域
            G2.append([source_data[i], target_data[j]])
            Lv2.append([source_v[i], target_v[j]])
            Ld2.append([source_d[i], target_d[j]])
            La2.append([source_a[i], target_a[j]])

            # 第三组：两个数据都来自目标域
            G3.append([target_data[i], target_data[j]])
            Lv3.append([target_v[i], target_v[j]])
            Ld3.append([target_d[i], target_d[j]])
            La3.append([target_a[i], target_a[j]])

            # 第四组：第一个数据来自目标域，第二个来自源域
            G4.append([target_data[i], source_data[j]])
            Lv4.append([target_v[i], source_v[j]])
            Ld4.append([target_d[i], source_d[j]])
            La4.append([target_a[i], source_a[j]])

            # 第五组：数据1来自源域，数据2来自目标域，但数据1和数据2标签不同
            if source_v[i] != target_v[j]:
                G5.append([source_data[i], target_data[j]])
                Lv5.append([source_v[i], target_v[j]])
                Ld5.append([source_d[i], target_d[j]])
                La5.append([source_a[i], target_a[j]])

            # 第六组：数据1来自目标域，数据2来自源域，但数据1和数据2标签不同
            if target_v[i] != source_v[j]:
                G6.append([target_data[i], source_data[j]])
                Lv6.append([target_v[i], source_v[j]])
                Ld6.append([target_d[i], source_d[j]])
                La6.append([target_a[i], source_a[j]])
    # 打乱每组数据的顺序
    for group in [G1, G2, G3, G4, G5, G6]:
        random.shuffle(group)
    for labels in [Lv1, Lv2, Lv3, Lv4, Lv5, Lv6, Ld1, Ld2, Ld3, Ld4, Ld5, Ld6, La1, La2, La3, La4, La5, La6]:
        random.shuffle(labels)

    return (G1, G2, G3, G4, G5, G6), (Lv1, Lv2, Lv3, Lv4, Lv5, Lv6), (Ld1, Ld2, Ld3, Ld4, Ld5, Ld6), (La1, La2, La3, La4, La5, La6)


def generate_groups(data_source, label_source, data_target, label_target, seed=1):
    torch.manual_seed(1 + seed)
    torch.cuda.manual_seed(1 + seed)
    n = data_target.shape[0]   # 样本数量

    # 打乱
    classes = torch.unique(label_target)  #提取label0和1
    classes = classes[torch.randperm(len(classes))]

    class_num = classes.shape[0]
    shot = n//class_num

    def s_idxs(c):
        idx = torch.nonzero(label_source.eq(int(c))) # 返回非零数据的位置
        return idx[torch.randperm(len(idx))][:shot*2].squeeze()

    def t_idxs(c):
        return torch.nonzero(label_target.eq(int(c)))[:shot].squeeze()

    source_idxs = list(map(s_idxs, classes))
    target_idxs = list(map(t_idxs, classes))

    source_matrix = torch.stack(source_idxs)
    target_matrix = torch.stack(target_idxs)

    G1, G2, G3, G4, G5, G6 = [],[],[],[],[],[]
    L1, L2, L3, L4, L5, L6 = [],[],[],[],[],[]


    for i in range(2):
        for j in range(shot):
            G1.append((data_source[source_matrix[i][j*2]], data_source[source_matrix[i][j*2+1]]))
            L1.append((label_source[source_matrix[i][j*2]], label_source[source_matrix[i][j*2+1]]))
            G2.append((data_source[source_matrix[i][j]], data_target[target_matrix[i][j]]))
            L2.append((label_source[source_matrix[i][j]], label_target[target_matrix[i][j]]))

            G3.append((data_source[source_matrix[i % 2][j]], data_source[source_matrix[(i+1) % 2][j]]))
            L3.append((label_source[source_matrix[i % 2][j]], label_source[source_matrix[(i + 1) % 2][j]]))
            G4.append((data_source[source_matrix[i % 2][j]], data_target[target_matrix[(i+ 1) % 2][j]]))
            L4.append((label_source[source_matrix[i % 2][j]], label_target[target_matrix[(i + 1) % 2][j]]))

    for i in range(class_num):
        for j in range(shot):
            G5.append((data_target[target_matrix[i][j]], data_target[target_matrix[i][int((j+1)%shot)]]))
            L5.append((label_target[target_matrix[i][j]], label_target[target_matrix[i][int((j+1)%shot)]]))
            if i == 0:
                G6.append((data_target[target_matrix[i][j]], data_target[target_matrix[(i+1) % 2][j]]))
                L6.append((label_target[target_matrix[i][j]], label_target[target_matrix[(i+1) % 2][j]]))
            else:
                G6.append((data_target[target_matrix[i][j]], data_target[target_matrix[(i + 1) % 2][int((j + 1) % shot)]]))
                L6.append((label_target[target_matrix[i][j]], label_target[target_matrix[(i + 1) % 2][int((j + 1) % shot)]]))

    groups=[G1,G2,G3,G4,G5,G6]
    groups_y=[L1,L2,L3,L4,L5,L6]

    #group_lengths = [(len(g), len(gy)) for g, gy in zip(groups, groups_y)]
    #print(group_lengths)

    for g in groups:
        assert(len(g)==n)
    # print("over")
    return groups,groups_y


def disorganize_groups(X_s,Y_s,X_t,Y_t,seed=1):
    return generate_groups(X_s,Y_s,X_t,Y_t,seed=seed)

