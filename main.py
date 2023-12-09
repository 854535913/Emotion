import argparse
import torch
import dataloader
from Models import valance_model
from Models import muti_model
import numpy as np
import os
from sklearn import metrics

parser=argparse.ArgumentParser() #用于控制参数
parser.add_argument('--n_epoches_1',type=int,default=5)     #5 120 50
parser.add_argument('--n_epoches_2', type=int, default=1)
parser.add_argument('--n_epoches_3', type=int, default=1)
parser.add_argument('--original', type = str, default= 's01')      #源域
parser.add_argument('--target', type = str, default= 's04')        #目标域
parser.add_argument('--n_target_samples', type=int, default=120 )  #每组样本数
parser.add_argument('--batch_size', type=int, default=64)
opt=vars(parser.parse_args())

use_cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)

#设置超参数
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
discriminator = valance_model.DCD(input_features=128)

classifier.to(device) #用于GPU运算
encoder.to(device)
discriminator.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()))


# 主程序
if __name__=='__main__':

    #   --------------- step 1: 使用源域数据进行训练，得到源特征提取器和源分类器 ---------------
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

        print("step1----Epoch %d/%d  accuracy: %.3f %.3f %.3f" % (epoch + 1, opt['n_epoches_1'], acc_temp_v, acc_temp_d, acc_temp_a))

    #   --------------- step 2: 分组对比学习，冻结源分类器模型，训练鉴别器模型 ---------------

    data_source, v_label_source, d_label_source, a_label_source = dataloader.disorganize_data(o_data)  #从源域读取打乱的数据
    data_target, v_label_target, d_label_target, a_label_target = dataloader.generate_balanced_samples(opt['n_target_samples'], t_data) #从目标域读取数据，取0和1样本各120个

    #data_source:[2400, 128] v_label_source:[2400] data_target:[240, 128] v_label_target:[240]

    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    for epoch in range(opt['n_epoches_2']):
        groups, groups_pairs = dataloader.disorganize_groups(data_source, v_label_source, data_target, v_label_target, seed=epoch)  #组成样本
        # (groups, groups_pairs):  [(240, 240), (240, 240), (240, 240), (240, 240), (240, 240), (240, 240)]

        total_sample_size = 6 * len(groups[1])  #样本总数:1440
        index_list = torch.randperm(total_sample_size)   #创建随机索引:[41,50,1,...] (0-total_sample_size)
        mini_batch_size = 30  #小批量训练

        loss_mean = []
        X1 = []
        X2 = []
        group_num_matrix = []

        for index in range(total_sample_size):
            group_num = index_list[index] // len(groups[1]) #group_num记录当前在哪个组
            x1, x2 = groups[group_num][index_list[index] - len(groups[1]) * group_num] #选择一对样本
            #print('x1.shape, x2.shape', x1.shape, x2.shape)
            #x1的维度是[128]，对应1s的微分熵数据(128hz)

            X1.append(x1)
            X2.append(x2)
            group_num_matrix.append(group_num)

            if (index + 1) % mini_batch_size == 0:  #当累积的样本数量达到预定的小批量大小mini_batch_size时开始训练

                X1 = np.stack(X1)    #x1:(30, 128)
                X2 = np.stack(X2)

                group_num_matrix = torch.LongTensor(group_num_matrix)      #鉴别器靠group_num来学习
                X1 = torch.tensor(X1).to(device)
                X2 = torch.tensor(X2).to(device)
                group_num_matrix = group_num_matrix.to(device)

                # 判别器训练
                optimizer_D.zero_grad()
                X_cat = torch.cat([encoder(X1), encoder(X2)], 1)
                y_pred = discriminator(X_cat.detach())
                loss = loss_fn(y_pred, group_num_matrix)
                loss.backward()
                optimizer_D.step()
                loss_mean.append(loss.item())
                X1 = [] #清空，为下一个小批量训练做准备
                X2 = []
                group_num_matrix = []

        print("step2----Epoch %d/%d loss:%.3f" % (epoch + 1, opt['n_epoches_2'], np.mean(loss_mean)))

    # -------------------training for step 3-------------------
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
        groups, groups_y = dataloader.disorganize_groups(data_source, v_label_source, data_target, v_label_target, seed=opt['n_epoches_2'] + epoch)
        G1, G2, G3, G4, G5, G6 = groups
        Y1, Y2, Y3, Y4, Y5, Y6 = groups_y
        groups_2 = [G2, G4, G5, G6]
        groups_y_2 = [Y2, Y4, Y5, Y6]  # 来自不同target 和 source标签相同和不同的
        n_iters = 4 * len(G2)
        index_list = torch.randperm(n_iters)

        n_iters_dcd = 6* len(G2)
        index_list_dcd = torch.randperm(n_iters_dcd)

        mini_batch_size_g_h = 30  # data only contains G2 and G4 ,so decrease mini_batch
        mini_batch_size_dcd = 60  # data contains G1,G2,G3,G4 so use 40 as mini_batch
        X1 = []
        X2 = []
        ground_truths_y1 = []
        ground_truths_y2 = []
        dcd_labels = []
        for index in range(n_iters):
            ground_truth = index_list[index] // len(G2) ## 只有 0 和 1
            x1, x2 = groups_2[ground_truth][index_list[index] - len(G2) * ground_truth]
            y1, y2 = groups_y_2[ground_truth][index_list[index] - len(G2) * ground_truth]
            # print("x1.shape, x2.shape", x1.shape, x2.shape)
            y1=torch.LongTensor([y1.item()])
            y2=torch.LongTensor([y2.item()])
            dcd_label = 0 if ground_truth == 0 or ground_truth ==2 else 2
            X1.append(x1)
            X2.append(x2)
            ground_truths_y1.append(y1)
            ground_truths_y2.append(y2)
            dcd_labels.append(dcd_label)

            if (index + 1) % mini_batch_size_g_h == 0:
                X1 = torch.stack([tmp.float() for tmp in X1])
                X2 = torch.stack([tmp.float() for tmp in X2])
                ground_truths_y1 = torch.LongTensor(ground_truths_y1)
                ground_truths_y2 = torch.LongTensor(ground_truths_y2)
                dcd_labels = torch.LongTensor(dcd_labels)
                X1 = X1.to(device)
                X2 = X2.to(device)
                ground_truths_y1 = ground_truths_y1.to(device)
                ground_truths_y2 = ground_truths_y2.to(device)
                dcd_labels = dcd_labels.to(device)
                optimizer_g_h.zero_grad()
                encoder_X1 = encoder(X1)
                encoder_X2 = encoder(X2)

                X_cat = torch.cat([encoder_X1, encoder_X2], 1)
                y_pred_X1 = classifier(encoder_X1)
                y_pred_X2 = classifier(encoder_X2)
                y_pred_dcd = discriminator(X_cat)

                loss_X1 = loss_fn(y_pred_X1, ground_truths_y1)
                loss_X2 = loss_fn(y_pred_X2, ground_truths_y2)
                loss_dcd = loss_fn(y_pred_dcd, dcd_labels)

                loss_sum = loss_X1 + loss_X2 + 1 * loss_dcd

                loss_sum.backward()
                optimizer_g_h.step()

                X1 = []
                X2 = []
                ground_truths_y1 = []
                ground_truths_y2 = []
                dcd_labels = []

        #冻结特征提取器和分类器，训练鉴别器
        X1 = []
        X2 = []
        ground_truths = []
        for index in range(n_iters_dcd):

            ground_truth = index_list_dcd[index] // len(groups[1])

            x1, x2 = groups[ground_truth][index_list_dcd[index] - len(groups[1]) * ground_truth]
            X1.append(x1)
            X2.append(x2.float())  ####  w错误的地方
            ground_truths.append(ground_truth)

            if (index + 1) % mini_batch_size_dcd == 0:
                X1 = torch.stack([tmp.float() for tmp in X1])
                X2 = torch.stack([tmp.float() for tmp in X2])
                ground_truths = torch.LongTensor(ground_truths)
                X1 = X1.to(device)
                X2 = X2.to(device)
                ground_truths = ground_truths.to(device)

                optimizer_d.zero_grad()
                X_cat = torch.cat([encoder(X1), encoder(X2)], 1)
                y_pred = discriminator(X_cat.detach())
                loss = loss_fn(y_pred, ground_truths)
                loss.backward()
                optimizer_d.step()
                # loss_mean.append(loss.item())
                X1 = []
                X2 = []
                ground_truths = []

        # testing
        auc = 0
        for data, labels, _, _ in test_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            y_test_pred = classifier(encoder(data))
            acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()
            # print(labels)

            auc += metrics.roc_auc_score(labels.cpu(),torch.max(y_test_pred, 1)[1].cpu() )

        accuracy = round(acc / float(len(test_dataloader)), 3)
        auc_temp =  round(auc / float(len(test_dataloader)), 3)


        if  accuracy > max_result:
            max_result = accuracy
            torch.save(encoder, path2 + '//encode_%s.pth' % (opt['original']))
            torch.save(classifier, path2 + '//classifier_%s.pth' % (opt['original']))
        accuracy_all2.append(accuracy)
        auc_all.append(auc_temp)

        print("step3----Epoch %d/%d  accuracy: %.3f ,%.3f" % (epoch + 1, opt['n_epoches_3'], accuracy, auc_temp))

    save_dir = path + '//' + str(o_data) + '.npz'
    np.savez(save_dir, accuracy_all = accuracy_all , accuracy_all2 = accuracy_all2, auc_all= auc_all)
