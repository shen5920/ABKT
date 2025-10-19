import torch
from DatasetLoading import get_split_sequences
import pickle
import os
import numpy as np
import torch.utils.data as Data
import torch.optim as opt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,accuracy_score
from PytorchModels import K_CMF,IRT_2

class CMF:
    def __init__(self,
                 dataset='ASSISTment2009',
                 type='RandomIterateSection',
                 min_length=10,
                 early_stop=10,
                 epoch=100,
                 m_lr=0.0005,
                 k_hidden_size=5,
                 device='cpu',
                 guess=0.25,
                 use_kfold=False,
                 kfold_data=None,
                 fold_idx=0,
                 batch_size=512):
        
        # 原有参数初始化
        self.dataset = dataset
        self.type = type
        self.k_hidden_size = k_hidden_size
        self.early_stop = early_stop
        self.epoch = epoch
        self.m_lr = m_lr
        self.device = device
        self.guess = guess
        self.use_kfold = use_kfold
        self.batch_size = batch_size  # 保存batch_size
        # 在初始化中保存fold_idx
        self.fold_idx = fold_idx

        self.train_history = []  # 记录每轮训练结果
        self.test_history = []   # 记录每轮测试结果
        
        if use_kfold and kfold_data is not None:
            # 使用k折数据
            train_sequences, test_triplet, Q_matrix_s = kfold_data
            self.user_num = max(max(seq[0]) for seq in train_sequences.values()) + 1
            self.item_num = max(max(seq[0]) for seq in train_sequences.values()) + 1
            self.skill_num = max([max(i) for i in Q_matrix_s]) + 1
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

        print('*'*20, 'hyperparameters', '*'*20)
        print('k_hidden_num:', self.k_hidden_size)
        print('guess:', self.guess)

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
        
        if len(self.train_itemsq_length) > 0:
            self.train_itemsq_length = torch.tensor(self.train_itemsq_length)
        else:
            print("Warning: No training data available!")
            return

        self.test_sets = torch.tensor(test_triplet).long().to(self.device)
        self.test_users = list(set(self.test_sets[:, 0].tolist()))

        # 训练的index
        self.train_index = torch.arange(0, len(self.train_users), 1)

    def train(self):
        if len(self.train_users) == 0:
            print("No training data available!")
            return
            
        print('*' * 20, 'start training', '*' * 20)
        print(f'Using batch size: {self.batch_size}')
        
        train_data_loader = Data.DataLoader(
            dataset=self.train_index,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.K_CMF = K_CMF(
            self.k_hidden_size,
            self.skill_num,
            self.user_num,
            self.item_num,
            self.Q_matrix,
        ).to(self.device)
        BCEloss = torch.nn.BCELoss()

        train_vars = list(self.K_CMF.parameters())
        optimizer = opt.Adam(train_vars, lr=self.m_lr)

        self.bestACC = 0
        self.bestAUC = 0
        stop = 0

        for epoch in range(self.epoch):
            self.K_CMF.train()

            bce_loss = 0
            train_pred_all = []
            train_pred01_all = []
            train_correct_all = []
            
            for batch_indices in train_data_loader:
                batch_loss = 0
                
                # 对批次中的每个用户进行前向传播
                for index in batch_indices:
                    user = self.train_users[index]
                    itemsq = self.train_itemsq[index].to(self.device)
                    correctsq = self.train_correctsq[index].to(self.device)
                    
                    user_k, _, _ = self.K_CMF.forward(user, itemsq)
                    item_q = self.Q_matrix[itemsq, :]
                    item_k = self.K_CMF.item_k[itemsq, :]

                    pred = IRT_2(user_k[:-1, :], item_k, item_q, self.guess)

                    train_pred_all.append(pred.detach().cpu().numpy())
                    train_pred01_all.append(pred.ge(0.5).float().detach().cpu().numpy())
                    train_correct_all.append(correctsq.cpu().numpy())
                    
                    loss = BCEloss(pred.clamp(0, 1), correctsq.float())
                    batch_loss += loss
                
                # 计算批次平均损失并反向传播
                if len(batch_indices) > 0:
                    avg_loss = batch_loss / len(batch_indices)
                    optimizer.zero_grad()
                    avg_loss.backward()
                    optimizer.step()
                    bce_loss += avg_loss.item()

            if len(train_pred_all) > 0:
                train_y_all = np.hstack(train_correct_all)
                train_pred_all = np.hstack(train_pred_all)
                train_pred01_all = np.hstack(train_pred01_all)

                bce_loss /= len(self.train_index)
                train_ACC = accuracy_score(train_y_all, train_pred01_all)
                train_AUC = roc_auc_score(train_y_all, train_pred_all)

                print('Epoch:', epoch, '| BCEloss:', bce_loss, 
                    '| ACC:', train_ACC, '| AUC:', train_AUC)
                
                # 记录训练结果
                self.train_history.append({
                    'epoch': epoch,
                    'train_loss': bce_loss,
                    'train_ACC': train_ACC,
                    'train_AUC': train_AUC
                })                

            # 测试阶段（保持不变）
            self.K_CMF.eval()
            test_user_state_k = []
            
            for test_user in self.test_users:
                if test_user in self.train_users:
                    train_index = self.train_users.index(test_user)
                    train_index_itemsq = self.train_itemsq[train_index].to(self.device)
                    train_model_output_k, _, _ = self.K_CMF.forward(train_index, train_index_itemsq)
                    test_out_k = train_model_output_k[-1, :]
                    test_user_state_k.append(test_out_k.unsqueeze(0))

            if len(test_user_state_k) > 0:
                test_user_state_k = torch.cat(test_user_state_k, 0)
                
                # 只对有训练数据的测试用户进行预测
                test_indices = []
                for i, user in enumerate(self.test_sets[:, 0]):
                    if user in self.train_users:
                        test_indices.append(i)
                
                if len(test_indices) > 0:
                    user_states_k = test_user_state_k[[self.train_users.index(u) for u in self.test_sets[test_indices, 0]], :]
                    item_states_q = self.Q_matrix[self.test_sets[test_indices, 1], :]
                    item_state_k = self.K_CMF.item_k[self.test_sets[test_indices, 1], :]
                    pred = IRT_2(user_states_k, item_state_k, item_states_q, self.guess)

                    test_pred_collect = pred.detach().cpu().numpy()
                    test_pred01_collect = pred.ge(0.5).float().detach().cpu().numpy()
                    test_y_collect = self.test_sets[test_indices, 2].cpu().numpy()

                    ACC = accuracy_score(test_y_collect, test_pred01_collect)
                    AUC = roc_auc_score(test_y_collect, test_pred_collect)
                    self.ACC = ACC
                    self.AUC = AUC

                    # 记录测试结果
                    self.test_history.append({
                        'epoch': epoch,
                        'test_ACC': ACC,
                        'test_AUC': AUC
                    })                    

                    if AUC > self.bestAUC:
                        self.bestAUC = AUC
                        # 创建保存目录（如果不存在）
                        save_dir = './Models/' + str(self.dataset) + '-' + str(self.type)
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # 根据是否k折决定保存路径
                        if self.use_kfold:
                            save_path_k = f'{save_dir}/CMF-k-{self.k_hidden_size}-{self.guess}-fold{self.fold_idx}-earlystop'
                        else:
                            save_path_k = f'{save_dir}/CMF-k-{self.k_hidden_size}-{self.guess}-earlystop'
                        
                        torch.save(self.K_CMF.state_dict(), save_path_k)
                        stop = 0
                    if ACC > self.bestACC:
                        self.bestACC = ACC
                        stop = 0
                    else:
                        stop = stop + 1
                    
                    print('Test ACC:', ACC, '| Test AUC:', AUC)
                    print('Best ACC:', self.bestACC, '| Best AUC:', self.bestAUC)
                    
                    if stop >= self.early_stop or epoch == self.epoch - 1:
                        print('*' * 20, 'stop training', '*' * 20)
                        if not self.use_kfold:
                            save_path_k = './Models/' + str(self.dataset) + '-' + str(self.type) + '/CMF-k-' + str(self.k_hidden_size) + '-' + str(self.guess) + '-epoch' + str(epoch)
                            torch.save(self.K_CMF.state_dict(), save_path_k)
                        break



    def log_result(self):
        filename = os.path.split(__file__)[-1].split(".")[0]
        f = open("./Results/" + filename + "-" + self.dataset + ".txt", "a+")
        
        # 写入基本信息
        f.write("datasets = " + self.dataset+ "\n")
        f.write("type = " + self.type+ "\n")
        f.write("k_hidden_num = " + str(self.k_hidden_size)+ " guess = " + str(self.guess)+ "\n")
        
        # 写入每轮训练结果
        f.write("\n" + "="*50 + "\n")
        f.write("TRAINING HISTORY\n")
        f.write("="*50 + "\n")
        f.write("Epoch\tTrain_Loss\tTrain_ACC\tTrain_AUC\n")
        for record in self.train_history:
            f.write(f"{record['epoch']}\t{record['train_loss']:.6f}\t{record['train_ACC']:.6f}\t{record['train_AUC']:.6f}\n")
        
        # 写入每轮测试结果
        f.write("\n" + "="*50 + "\n")
        f.write("TESTING HISTORY\n")
        f.write("="*50 + "\n")
        f.write("Epoch\tTest_ACC\tTest_AUC\n")
        for record in self.test_history:
            f.write(f"{record['epoch']}\t{record['test_ACC']:.6f}\t{record['test_AUC']:.6f}\n")
        
        # 写入最佳结果
        f.write("\n" + "="*50 + "\n")
        f.write("BEST RESULTS\n")
        f.write("="*50 + "\n")
        f.write("Best ACC = " + str(self.bestACC) + "\n")
        f.write("Best AUC = " + str(self.bestAUC) + "\n")
        f.write("Final ACC = " + str(self.ACC) + "\n")
        f.write("Final AUC = " + str(self.AUC) + "\n")
        f.write("\n")
        f.write("\n")
        f.close()
        print("The results are logged!!!")




if __name__ == '__main__':
    hidden_num_range = [5]
    guess_range = [0.25]

    for hr in hidden_num_range:
        for g in guess_range:
            cmf = CMF(m_lr=0.001,
                      early_stop=10,
                      epoch=50,
                      device='cuda:0',
                      k_hidden_size=hr,
                      guess=g,
                      )
            cmf.train()
            cmf.log_result()





