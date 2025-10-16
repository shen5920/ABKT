# -*- coding: utf-8 -*-

"""

--------------------------------------

    Author: Xiaoxuan Shen
    
    Date:   2021/4/20 9:05
    
--------------------------------------

"""

import torch
from DatasetLoading import get_split_sequences
import pickle
import os
import torch.utils.data as Data
import torch.optim as opt
from sklearn.metrics import roc_auc_score,accuracy_score
from PytorchModels import K_CMF, GMF, IRT_2
import numpy as np
from scipy import sparse

class GMF_BOOSTING:
    def __init__(self,
                 embedding_k=50,
                 m_lr=0.0001,
                 dataset='ASSISTment2009',
                 type='RandomIterateSection',
                 min_length=10,
                 early_stop=50,
                 batch_size=128,
                 epoch=50000,
                 CMF_k=5,
                 CMF_guess=0.25,
                 pretrain_clip=0.4,
                 combine='add',
                 symmetric=True,
                 adj=True,
                 GMF_layer=1,
                 m_lambda=0.1,
                 device='cpu',
                 use_kfold=False,
                 kfold_data=None,
                 fold_idx=0,
                 cmf_model=None):
        
        # 原有参数初始化
        self.embedding_k = embedding_k
        self.m_lr = m_lr
        self.early_stop = early_stop
        self.dataset = dataset
        self.type = type
        self.min_length = min_length
        self.batch_size = batch_size
        self.epoch = epoch
        self.CMF_k = CMF_k
        self.CMF_guess = CMF_guess
        self.pretrain_clip = pretrain_clip
        self.combine = combine
        self.symmetric = symmetric
        self.adj = adj
        self.GMF_layer = GMF_layer
        self.m_lambda = m_lambda
        self.device = device
        self.use_kfold = use_kfold
        
        if use_kfold and kfold_data is not None:
            # 使用k折数据
            train_sequences, test_triplet, Q_matrix_s = kfold_data
            self.user_num = max(max(seq[0]) for seq in train_sequences.values()) + 1
            self.item_num = max(max(seq[0]) for seq in train_sequences.values()) + 1
            self.skill_num = max([max(i) for i in Q_matrix]) + 1
            self.record_num = sum(len(seq[0]) for seq in train_sequences.values()) + len(test_triplet)
            
            print(f"Fold {fold_idx}: {self.user_num} students, {self.item_num} questions, "
                  f"{self.skill_num} skills, {self.record_num} records")
        else:
            # 原有数据加载逻辑
            save_dataset_path = './ProcessedData/' + dataset + '-' + type + '-' + str(min_length) + '-squence'
            if os.access(save_dataset_path, os.F_OK):
                print("Processed data is existent...")
                print('Loading...')
                save_dataset_file = open(save_dataset_path, 'rb')
                [self.user_num, self.item_num, self.skill_num, self.record_num, train_sequences, test_triplet, Q_matrix_s] = \
                    pickle.load(save_dataset_file)
                save_dataset_file.close()
            else:
                print("Processed data is not existent...")
                self.user_num, self.item_num, self.skill_num, self.record_num, train_sequences, test_triplet, Q_matrix_s = \
                    get_split_sequences(dataset, type, min_length)

                train_test_sets = [self.user_num, self.item_num, self.skill_num,
                                   self.record_num, train_sequences, test_triplet, Q_matrix_s]
                save_dataset_file = open(save_dataset_path, 'wb')
                pickle.dump(train_test_sets, save_dataset_file)
                save_dataset_file.close()
                print('Training set and test set are saved in', save_dataset_path)
            print("Data is processed which has:", self.user_num, 'students,',
                  self.item_num, 'questions,',
                  self.skill_num, 'skills,', self.record_num, 'records')

        # 构建Q矩阵
        Q_matrix = torch.zeros((self.item_num, self.skill_num))
        index = 0
        for i in Q_matrix_s:
            for ii in i:
                Q_matrix[index][int(ii)] = 1
            index += 1
        self.Q_matrix = Q_matrix.to(self.device)

        # 处理训练数据
        self.train_users = []
        self.train_itemsq = []
        self.train_correctsq = []
        self.train_itemsq_length = []
        for user in train_sequences:
            if len(train_sequences[user][0]) > 0:  # 确保训练序列不为空
                self.train_users.append(user)
                self.train_itemsq.append(torch.tensor(train_sequences[user][0]).squeeze(0))
                self.train_correctsq.append(torch.tensor(train_sequences[user][1]).squeeze(0).long())
                self.train_itemsq_length.append(len(train_sequences[user][0]))

        self.test_sets = torch.tensor(test_triplet).long().to(self.device)
        self.test_users = list(set(self.test_sets[:, 0].tolist()))

        # 训练的index
        self.train_index = torch.arange(0, len(self.train_users), 1)

        # 使用提供的CMF模型或加载预训练模型
        if cmf_model is not None:
            CMF_model = cmf_model
        else:
            CMF_model = K_CMF(
                self.CMF_k,
                self.skill_num,
                self.user_num,
                self.item_num,
                self.Q_matrix,
            ).to(self.device)
            
            if not use_kfold:
                CMF_model.load_state_dict(torch.load('./Models/' + str(self.dataset) + '-' + str(self.type) + '/CMF-k-' + str(self.CMF_k) + '-' + str(self.CMF_guess) + '-earlystop'))

        print('Computing output in Pre-trained model...')
        print('Training set...')
        
        self.pre_train_output = []
        self.user_final_state = {}

        for index in self.train_index:
            user = self.train_users[index]
            itemsq = self.train_itemsq[index].to(self.device)
            user_k, _, _ = CMF_model.forward(user, itemsq)
            item_q = self.Q_matrix[itemsq, :]
            item_k = CMF_model.item_k[itemsq, :]
            pred = IRT_2(user_k[:-1, :], item_k, item_q, self.CMF_guess)
            clip_pred = pred.clamp(0 + self.pretrain_clip, 1 - self.pretrain_clip)
            self.pre_train_output.append(clip_pred.detach())
            self.user_final_state[user] = user_k[-1, :].detach().unsqueeze(0)
        
        print('Testing set...')
        test_user_state_k = []
        for test_user in self.test_users:
            if test_user in self.user_final_state:
                test_user_state_k.append(self.user_final_state[test_user])
        
        if len(test_user_state_k) > 0:
            user_states_k = torch.cat(test_user_state_k, 0)
            item_states_q = self.Q_matrix[self.test_sets[:, 1], :]
            item_state_k = CMF_model.item_k[self.test_sets[:, 1], :]
            pred_test = IRT_2(user_states_k, item_state_k, item_states_q, self.CMF_guess).detach()
            clip_pred_test = pred_test.clamp(0 + self.pretrain_clip, 1 - self.pretrain_clip)
            self.pre_test_output = clip_pred_test
            self.test_sets = torch.cat([self.test_sets, self.pre_test_output.unsqueeze(1)], 1)

        print('Convert sequence training set to triplet...')
        train_sets = []
        for index in self.train_index:
            user = torch.tensor(self.train_users[index])
            for indexi in range(len(self.train_itemsq[index])):
                item = self.train_itemsq[index][indexi]
                correct = self.train_correctsq[index][indexi]
                pre_output = self.pre_train_output[index][indexi]
                triplet = torch.tensor([user, item, correct, pre_output]).unsqueeze(0)
                train_sets.append(triplet)
        
        if len(train_sets) > 0:
            self.train_sets = torch.cat(train_sets, 0)
        else:
            print("Warning: No training data available!")
            return

        print("Building adjacent matrix...")
        aj_row = np.append(self.train_sets[:, 0].numpy(), self.train_sets[:, 1].numpy() + self.user_num)
        aj_col = np.append(self.train_sets[:, 1].numpy() + self.user_num, self.train_sets[:, 0].numpy())
        aj_data = np.ones(self.train_sets.shape[0] * 2)
        aj_matrix = sparse.coo_matrix((aj_data, (aj_row, aj_col)),
                                      shape=(self.user_num + self.item_num, self.user_num + self.item_num)).tocsr()

        print("Normalizing adjacent matrix...")
        aj_matrix = aj_matrix + sparse.eye(self.user_num + self.item_num)
        if symmetric:
            d = sparse.diags(np.power(np.array(aj_matrix.sum(1)), -0.5).flatten(), 0)
            aj_norm = aj_matrix.dot(d).transpose().dot(d).tocoo()
        else:
            d = sparse.diags(np.power(np.array(aj_matrix.sum(1)), -1).flatten(), 0)
            aj_norm = d.dot(aj_matrix).tocoo()

        del aj_matrix

        values = aj_norm.data
        indices = np.vstack((aj_norm.row, aj_norm.col))
        i = torch.tensor(indices)
        v = torch.tensor(values)
        shape = aj_norm.shape
        self.aj_norm = torch.sparse_coo_tensor(i, v, shape)

        self.train_sets = self.train_sets.to(self.device)
        self.aj_norm = self.aj_norm.to(self.device)

    def train(self):
        if not hasattr(self, 'train_sets'):
            print("No training data available!")
            return
            
        print('*' * 20, 'start training', '*' * 20)
        y = self.train_sets[:, 2]
        k = self.train_sets[:, 3]
        g = y / k - (1 - y) / (1 - k)
        w = -(y / (torch.pow(k, 2)) + (1 - y) / (torch.pow((1 - k), 2)))
        self.train_sets = torch.cat([self.train_sets, g.unsqueeze(1), w.unsqueeze(1)], 1)

        train_data_loader = Data.DataLoader(
            dataset=self.train_sets,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.bestACC = 0
        self.bestAUC = 0
        self.model = GMF(self.user_num, self.item_num, self.embedding_k, self.aj_norm,
                         self.adj, self.GMF_layer).to(self.device)
        optimizer = opt.Adam(self.model.parameters(), lr=self.m_lr)
        stop = 0
        
        for e in range(self.epoch):
            self.model.train()
            loss = 0
            l2_loss = 0
            
            for batch_idx, batch in enumerate(train_data_loader):
                optimizer.zero_grad()
                batch_num = batch.shape[0]
                u_idx = batch[:, 0].long()
                i_idx = batch[:, 1].long()
                y_batch = batch[:, 2]
                k_batch = batch[:, 3]
                g_batch = batch[:, 4]
                w_batch = batch[:, 5]

                pred, u_norm, i_norm = self.model.forward(u_idx, i_idx)
                if self.combine == 'add':
                    loss = -torch.mean(w_batch * torch.pow(pred + g_batch / w_batch, 2))
                elif self.combine == 'mul':
                    loss = -torch.mean(w_batch * k_batch * torch.pow((pred + g_batch / (w_batch * k_batch) - 1), 2))
                else:
                    print("choose right combine method from ['add','mul','none']")

                l2 = u_norm + i_norm
                totle_loss = loss + self.m_lambda * l2
                totle_loss.backward()
                optimizer.step()
                loss += loss.cpu() * batch_num
                l2_loss += l2.cpu() * batch_num

            loss /= self.train_sets.shape[0]
            l2_loss /= self.train_sets.shape[0]
            print('Epoch:', e, '| Loss:', loss.cpu().detach().numpy(), '| l2loss:', l2_loss.cpu().detach().numpy())

            # 测试阶段
            self.model.eval()
            test_pred, _, _ = self.model.forward(self.test_sets[:, 0].long(), self.test_sets[:, 1].long())
            pre_test_output = self.test_sets[:, 3]
            
            if self.combine == 'add':
                test_pred = test_pred + pre_test_output
            elif self.combine == 'mul':
                test_pred = test_pred * pre_test_output
            else:
                print("choose right combine method from ['add','mul','none']")

            test_pred_01 = test_pred.ge(0.5).float()
            test_pred = test_pred.cpu().detach().numpy()
            test_correct = self.test_sets[:, 2].cpu().numpy()
            test_pred_01 = test_pred_01.cpu().detach().numpy()
            ACC = accuracy_score(test_correct, test_pred_01)
            AUC = roc_auc_score(test_correct, test_pred)
            
            if AUC > self.bestAUC:
                self.bestAUC = AUC
                if not self.use_kfold:  # k折时不保存模型
                    save_path_k = './Models/' + str(self.dataset) + '-' + str(self.type) + '/GMF-boosting-' + str(
                        self.combine) + '-' + str(self.embedding_k) + '-earlystop'
                    torch.save(self.model.state_dict(), save_path_k)
                stop = 0
            if ACC > self.bestACC:
                self.bestACC = ACC
                stop = 0
            else:
                stop = stop + 1
            
            print('Test ACC:', ACC, '| Test AUC:', AUC)
            if stop >= self.early_stop:
                print('*' * 20, 'stop training', '*' * 20)
                print('Best ACC:', self.bestACC, '| Best AUC:', self.bestAUC)
                break

    def log_result(self):
        filename = os.path.split(__file__)[-1].split(".")[0]
        f = open("./Results/" + filename + "-" + self.dataset +'-' + self.type + ".txt", "a+")
        f.write("datasets = " + self.dataset+ "\n")
        f.write("embedding_k = " + str(self.embedding_k) + "\n")
        f.write("CMF_k = " + str(self.CMF_k) + " CMF_guess = " + str(self.CMF_guess) + "\n")
        f.write("pretrain_clip = " + str(self.pretrain_clip) + " combine method = " + str(self.combine) + "\n")
        f.write("adjustment matrix = " + str(self.adj) + " GMF layers = " + str(self.GMF_layer) + "\n")
        f.write("m_lambda = " + str(self.m_lambda) + "\n")
        f.write("Best ACC = " + str(self.bestACC) + "\n")
        f.write("Best AUC = " + str(self.bestAUC) + "\n")
        f.write("\n")
        f.write("\n")
        print("The results are logged!!!")















