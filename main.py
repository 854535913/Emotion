import argparse
import torch
import dataloader
from Models import valance_model
from Models import muti_model
import numpy as np
import os
from sklearn import metrics

parser=argparse.ArgumentParser() #用于控制参数
parser.add_argument('--n_epoches_1',type=int,default=1)     #5 120 50
parser.add_argument('--n_epoches_2', type=int, default=1)
parser.add_argument('--n_epoches_3', type=int, default=1)
parser.add_argument('--original', type = str, default= 's01')      #源域
parser.add_argument('--target', type = str, default= 's04')        #目标域
parser.add_argument('--n_target_samples', type=int, default=120 )  #每组样本数
parser.add_argument('--batch_size', type=int, default=64)
opt = vars(parser.parse_args())

use_cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)

# 设置超参数
batch_size = 30
dropout = 0.5
output_size = 64
hidden_size = 32
embed_dim = 32
bidirectional = True
attention_size = 16
sequence_length = 128
o_data = opt['original']
t_data = opt['target']

train_dataloader = dataloader.dataloader_maker(o_data)
test_dataloader = dataloader.dataloader_maker(t_data)

classifier = muti_model.Classifier()
encoder = muti_model.Encoder()
discriminator_v = muti_model.DCD(input_features=128)
discriminator_a = muti_model.DCD(input_features=128)
discriminator_d = muti_model.DCD(input_features=128)

classifier.to(device) #用于GPU运算
encoder.to(device)
discriminator_v.to(device)
discriminator_a.to(device)
discriminator_d.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()))


# 主程序
if __name__=='__main__':

    #   --------------- step 1: 使用源域数据进行训练，得到源特征提取器和源分类器 ---------------
    print("-----step1-----")
    accuracy_all = []
    for epoch in range(opt['n_epoches_1']): #定义训练周期数
        for data, valence_labels, dominance_labels, arousal_labels in train_dataloader:  #遍历数据集
            data = data.to(device)
            valence_labels = (valence_labels.long()).to(device)
            dominance_labels = (dominance_labels.long()).to(device)
            arousal_labels = (arousal_labels.long()).to(device)

            optimizer.zero_grad()  #清除累积的梯度
            pred_v, pred_d, pred_a = classifier(encoder(data)) #将数据传递给encoder和classifier来获得预测结果
            loss = loss_fn(pred_v, valence_labels) + loss_fn(pred_d, dominance_labels) + loss_fn(pred_a, arousal_labels)
            loss.backward()  #损失进行反向传播，计算梯度
            optimizer.step()

        acc_v = acc_d = acc_a = 0
        for data, valence_labels, dominance_labels, arousal_labels in test_dataloader:  # 遍历测试数据集
            data = data.to(device)   #data:[30, 128]
            valence_labels = (valence_labels.long()).to(device)
            dominance_labels = (dominance_labels.long()).to(device)
            arousal_labels = (arousal_labels.long()).to(device)

            test_pred_v, test_pred_d, test_pred_a= classifier(encoder(data))  #y_test_pred:[30, 2],2是两个类别的预测概率

            acc_v += (torch.max(test_pred_v, 1)[1] == valence_labels).float().mean().item()
            acc_d += (torch.max(test_pred_d, 1)[1] == dominance_labels).float().mean().item()
            acc_a += (torch.max(test_pred_a, 1)[1] == arousal_labels).float().mean().item()

        acc_temp_v = round(acc_v / float(len(test_dataloader)), 3)
        acc_temp_d = round(acc_d / float(len(test_dataloader)), 3)
        acc_temp_a = round(acc_a / float(len(test_dataloader)), 3)

        accuracy_all.append(acc_temp_v)

        print("Epoch %d/%d   accuracy: %.3f %.3f %.3f" % (epoch + 1, opt['n_epoches_1'], acc_temp_v, acc_temp_d, acc_temp_a))

    #   --------------- step 2: 分组对比学习，冻结源分类器模型，训练鉴别器模型 ---------------
    print("-----step2-----")

    data_source, v_label_source, d_label_source, a_label_source = dataloader.disorganize_data(o_data)  # 从源域读取打乱的数据
    data_target, v_label_target, d_label_target, a_label_target = dataloader.generate_balanced_samples(opt['n_target_samples'], t_data) #从目标域读取数据，取0和1样本各120个

    # data_source:[2400,128] v_label_source:[2400] data_target:[240,128] v_label_target:[240]

    optimizer_D_v = torch.optim.Adam(discriminator_v.parameters(), lr=0.001)
    optimizer_D_a = torch.optim.Adam(discriminator_a.parameters(), lr=0.001)
    optimizer_D_d = torch.optim.Adam(discriminator_d.parameters(), lr=0.001)

    for epoch in range(opt['n_epoches_2']):
        G, Lv, La, Ld = dataloader.generate_groups2(o_data, t_data)

        total_sample_size = 3 * len(G[1])   # 样本总数:1440
        index_list = torch.randperm(total_sample_size)   # 创建随机索引:[41,50,1,...] (0-total_sample_size)
        mini_batch_size = 30  # 小批量训练

        loss_mean = []
        pair_x = []
        pair_y = []
        group_num_v = []
        group_num_a = []
        group_num_d = []

        for index in range(total_sample_size):
            group_num = index_list[index] // len(G[1]) #group_num记录当前在哪个组
            x, y = G[group_num][index_list[index] - len(G[1]) * group_num] #选择一对样本
            #print('x1.shape, x2.shape', x1.shape, x2.shape)
            #x1的维度是[128]，对应1s的微分熵数据(128hz)

            pair_x.append(x)
            pair_y.append(y)

            lv1, lv2 = Lv[group_num][index_list[index] - len(G[1]) * group_num]
            if lv1 == lv2:
                group_num_v.append(group_num)
            else: group_num_v.append(group_num+3)

            la1, la2 = La[group_num][index_list[index] - len(G[1]) * group_num]
            if la1 == la2:
                group_num_a.append(group_num)
            else: group_num_a.append(group_num+3)

            ld1, ld2 = Ld[group_num][index_list[index] - len(G[1]) * group_num]
            if ld1 == ld2:
                group_num_d.append(group_num)
            else: group_num_d.append(group_num+3)

            if (index + 1) % mini_batch_size == 0:  #当累积的样本数量达到预定的小批量大小mini_batch_size时开始训练

                pair_x = np.stack(pair_x)    #pair_x:(30, 128)
                pair_y = np.stack(pair_y)
                group_num_v = torch.LongTensor(group_num_v)      #鉴别器靠group_num来学习
                group_num_a = torch.LongTensor(group_num_a)
                group_num_d = torch.LongTensor(group_num_d)

                pair_x = torch.tensor(pair_x).to(device)
                pair_y = torch.tensor(pair_y).to(device)
                group_num_v = group_num_v.to(device)
                group_num_a = group_num_a.to(device)
                group_num_d = group_num_d.to(device)

                optimizer_D_v.zero_grad()
                optimizer_D_a.zero_grad()
                optimizer_D_d.zero_grad()
                X_cat = torch.cat([encoder(pair_x), encoder(pair_y)], 1)

                y_pred_v = discriminator_v(X_cat.detach())
                loss_v = loss_fn(y_pred_v, group_num_v)
                y_pred_a = discriminator_a(X_cat.detach())
                loss_a = loss_fn(y_pred_a, group_num_a)
                y_pred_d = discriminator_d(X_cat.detach())
                loss_d = loss_fn(y_pred_d, group_num_d)
                total_loss = loss_v + loss_a + loss_d
                total_loss.backward()
                optimizer_D_v.step()
                optimizer_D_a.step()
                optimizer_D_d.step()
                loss_mean.append(total_loss.item())

                pair_x = [] #清空，为下一个小批量训练做准备
                pair_y = []
                group_num_v = []
                group_num_a = []
                group_num_d = []


        print("Epoch %d/%d   loss:%.3f" % (epoch + 1, opt['n_epoches_2'], np.mean(loss_mean)))

    #   --------------- step 3: 鉴别器投入使用 ---------------
    print("-----step3-----")

    optimizer_g_h = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=0.001)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

    test_dataloader = dataloader.dataloader_maker(t_data)
    accuracy_all2 = []
    auc_all = []

    path = './/deap_6//' + str(t_data)
    isExists = os.path.exists(path)
    if isExists:
        pass
    else:
        os.makedirs(path)
    path2 = './/deap_6//' + str(t_data) + "//save_model"
    isExists2 = os.path.exists(path2)
    if isExists2:
        pass
    else:
        os.makedirs(path2)
    max_result = 0

    for epoch in range(opt['n_epoches_3']):

        # 冻结鉴别器，训练特征提取器和分类器
        groups, groups_label = dataloader.disorganize_groups(data_source, d_label_source, data_target, d_label_source, seed=epoch)
        G1, G2, G3, G4, G5, G6 = groups
        Y1, Y2, Y3, Y4, Y5, Y6 = groups_label
        groups_2 = [G2, G4, G5, G6]
        groups_label_2 = [Y2, Y4, Y5, Y6]  # 来自不同target 和 source标签相同和不同的
        total_sample_size = 4 * len(G2)
        index_list = torch.randperm(total_sample_size)

        total_sample_size_dcd = 6 * len(G2)
        index_list_dcd = torch.randperm(total_sample_size_dcd)

        mini_batch_size_g_h = 30  # data only contains G2 and G4 ,so decrease mini_batch
        mini_batch_size_dcd = 60  # data contains G1,G2,G3,G4 so use 40 as mini_batch
        pair_x = []
        pair_y = []
        label_x = []
        label_y = []
        dcd_labels = []
        for index in range(total_sample_size):
            group_num = index_list[index] // len(G2)  # 只有 0 和 1
            x1, x2 = groups_2[group_num][index_list[index] - len(G2) * group_num]
            y1, y2 = groups_label_2[group_num][index_list[index] - len(G2) * group_num]
            y1 = torch.LongTensor([y1.item()])
            y2 = torch.LongTensor([y2.item()])
            dcd_label = 0 if group_num == 0 or group_num == 2 else 2
            pair_x.append(x1)
            pair_y.append(x2)
            label_x.append(y1)
            label_y.append(y2)
            dcd_labels.append(dcd_label)

            if (index + 1) % mini_batch_size_g_h == 0:
                pair_x = torch.stack([tmp.float() for tmp in pair_x])
                pair_y = torch.stack([tmp.float() for tmp in pair_y])
                label_x = torch.LongTensor(label_x)
                label_y = torch.LongTensor(label_y)
                dcd_labels = torch.LongTensor(dcd_labels)
                pair_x = pair_x.to(device)
                pair_y = pair_y.to(device)
                label_x = label_x.to(device)
                label_y = label_y.to(device)
                dcd_labels = dcd_labels.to(device)
                optimizer_g_h.zero_grad()

                X_cat = torch.cat([encoder(pair_x), encoder(pair_y)], 1)
                _, y_pred_X1, _ = classifier(encoder(pair_x))
                _, y_pred_X2, _ = classifier(encoder(pair_y))
                y_pred_dcd = discriminator(X_cat)

                loss_x = loss_fn(y_pred_X1, label_x)
                loss_y = loss_fn(y_pred_X2, label_y)
                loss_dcd = loss_fn(y_pred_dcd, dcd_labels)

                loss_sum = loss_x + loss_y + 1 * loss_dcd

                loss_sum.backward()
                optimizer_g_h.step()

                pair_x = []
                pair_y = []
                label_x = []
                label_y = []
                dcd_labels = []

        # 冻结特征提取器和分类器，训练鉴别器
        pair_x = []
        pair_y = []
        group_num = []
        for index in range(total_sample_size_dcd):

            group_num = index_list_dcd[index] // len(groups[1])
            x1, x2 = groups[group_num][index_list_dcd[index] - len(groups[1]) * group_num]
            pair_x.append(x1)
            pair_y.append(x2.float())
            group_num.append(group_num)

            if (index + 1) % mini_batch_size_dcd == 0:
                pair_x = torch.stack([tmp.float() for tmp in pair_x])
                pair_y = torch.stack([tmp.float() for tmp in pair_y])
                group_num = torch.LongTensor(group_num)
                pair_x = pair_x.to(device)
                pair_y = pair_y.to(device)
                group_num = group_num.to(device)

                optimizer_d.zero_grad()
                X_cat = torch.cat([encoder(pair_x), encoder(pair_y)], 1)
                y_pred = discriminator(X_cat.detach())
                loss = loss_fn(y_pred, group_num)
                loss.backward()
                optimizer_d.step()
                pair_x = []
                pair_y = []
                group_num = []

    #   --------------- test ---------------
    print("-----test-----")
    test_dataloader = dataloader.dataloader_maker(t_data)
    acc_v = acc_d = acc_a = 0
    auc_v = auc_d = auc_a = 0

    for data, valence_labels, dominance_labels, arousal_labels in test_dataloader:  # 遍历测试数据集
        data = data.to(device)  # data:[30, 128]
        valence_labels = (valence_labels.long()).to(device)
        dominance_labels = (dominance_labels.long()).to(device)
        arousal_labels = (arousal_labels.long()).to(device)

        test_pred_v, test_pred_d, test_pred_a = classifier(encoder(data))  # y_test_pred:[30, 2],2是两个类别的预测概率

        acc_v += (torch.max(test_pred_v, 1)[1] == valence_labels).float().mean().item()
        acc_d += (torch.max(test_pred_d, 1)[1] == dominance_labels).float().mean().item()
        acc_a += (torch.max(test_pred_a, 1)[1] == arousal_labels).float().mean().item()

        auc_v += metrics.roc_auc_score(valence_labels.cpu(), torch.max(test_pred_v, 1)[1].cpu())
        auc_d += metrics.roc_auc_score(dominance_labels.cpu(), torch.max(test_pred_d, 1)[1].cpu())
        auc_a += metrics.roc_auc_score(arousal_labels.cpu(), torch.max(test_pred_a, 1)[1].cpu())

    acc_temp_v = round(acc_v / float(len(test_dataloader)), 3)
    acc_temp_d = round(acc_d / float(len(test_dataloader)), 3)
    acc_temp_a = round(acc_a / float(len(test_dataloader)), 3)

    auc_temp_v = round(auc_v / float(len(test_dataloader)), 3)
    auc_temp_d = round(auc_d / float(len(test_dataloader)), 3)
    auc_temp_a = round(auc_a / float(len(test_dataloader)), 3)

    print("final accuracy: %.3f %.3f %.3f" % (acc_temp_v, acc_temp_d, acc_temp_a))
    print("final Area Under the Curve: %.3f %.3f %.3f" % (auc_temp_v, auc_temp_d, auc_temp_a))

