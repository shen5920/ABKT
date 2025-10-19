import pandas as pd
import math
import numpy as np

def load_dataset(dataset,type,min_length):
    if dataset == 'ASSISTment2009':#ASSISTment2009 按type筛选
        path = "./Datasets/" + dataset + "/raw.csv"
        usecols = ['order_id', 'user_id', 'problem_id', 'correct', 'skill_id', 'type']
        print('Loading dataset from', path, ' with cols:', usecols)
        csv_data = pd.read_csv(path, usecols=usecols)
        # 按时间顺序对数据排序
        csv_data.sort_values(['order_id'], ascending=True)
        # 选择特定类型（type）的数据
        print('Choosing data where type=', type)

        '''——————————————————————————————————修改数据集 子集->全部————————————————————————————————————————'''
        #type_data = csv_data[csv_data['type'] == type]
        type_data = csv_data

        user_list = type_data['user_id'].tolist()
        problem_list = type_data['problem_id'].tolist()
        correct_list = type_data['correct'].tolist()
        skill_list = type_data['skill_id'].tolist()
        # 每一行为[user_id，problem_id，correct，skill_id] 学生 问题 答题结果 技能
        data_raw = [user_list, problem_list, correct_list, skill_list]
    elif dataset == 'AICFE':#AICFE 按学科拆分
        path = "./Datasets/" + dataset + "/"+type+"/unit-"+type+".csv"
        #type动态地构建了文件的路径
        file = open(path)
        lines = file.readlines()
        skip = 0
        user_list = []
        problem_list = []
        correct_list = []
        skill_list = []
        for line in lines:
            if skip != 0:#跳过表头
                data = line.strip('\n').split(',')
                #从 data 列表中按顺序取出我们需要的列
                user = data[0]
                problem = data[3]
                skill = data[4]
                score = data[5]
                full_score = data[6]
                if skill != 'n.a.' and full_score != 'n.a.':
                    #数据清洗：skill和full_score均有效时 才会处理此行数据
                    user_list.append(user)
                    problem_list.append(problem)
                    skill_list.append(skill)
                    if score == full_score:
                        correct_list.append(1)
                    #满分记为答对 否则记为答错
                    else:
                        correct_list.append(0)
            else:
                skip += 1
        data_raw = [user_list, problem_list, correct_list, skill_list]

    #统计用户序列长度
    user_count = {}
    for i in range(data_raw[0].__len__()):#在学生个数上循环
        userid = data_raw[0][i]
        #如果学生出现过，则此学生的序列长度+1（比如第五次见到此学生 意思为这是此学生做的第五题）
        if user_count.__contains__(userid):
            user_count[userid] += 1
        else:
            user_count[userid] = 1
    #得到每个学生的序列长度（做题个数）


    #过滤与重新编号
    '''过滤：剔除掉那些答题数量太少（序列长度小于 min_length）的学生。
    重新编号：将原始的、可能不连续的ID（如 'user_123', 'problem_abc'）
                 转换成从0开始的、连续的整数索引（如 0, 1, 2…）。'''
    user_id = {}
    item_id = {}
    skill_id = {}
    user_list_filtered = []
    item_list_filtered = []
    correct_list_filtered = []
    filtered_Q_matrix = []
    print('Filting data where sequence length>=',min_length)
    for i in range(data_raw[0].__len__()):
        user = data_raw[0][i]
        item = data_raw[1][i]
        correct = data_raw[2][i]
        if dataset == 'ASSISTment2009':
            skillids = data_raw[3][i].split(',')
        else:
            skillids = data_raw[3][i].split('~~')

        if user_count[user] >= min_length:
            if not user_id.__contains__(user):
                user_id[user] = user_id.__len__()
            if not item_id.__contains__(item):
                item_id[item] = item_id.__len__()
                skills = []
                for skill in skillids:
                    if not skill_id.__contains__(skill):
                        skill_id[skill] = skill_id.__len__()
                    skills.append(skill_id[skill])
                filtered_Q_matrix.append(skills)
            user_list_filtered.append(user_id[user])
            item_list_filtered.append(item_id[item])
            correct_list_filtered.append(correct)
    print(user_id)
    print(skill_id)
    return [user_list_filtered,item_list_filtered,correct_list_filtered,filtered_Q_matrix]


def get_split_triplet(dataset, type, min_length):
    [user_list, item_list, correct_list,Q_matrix] = load_dataset(dataset, type, min_length)
    #计算总数
    user_num = max(user_list) + 1
    item_num = max(item_list) + 1
    skill_num = max([max(i) for i in Q_matrix]) + 1
    record_num = user_list.__len__()

    #all_sequences = {userid:[[itemids,...],[correct,...]]}
    all_sequences = {}
    for i in range(user_list.__len__()):
        if all_sequences.__contains__(user_list[i]):
            all_sequences[user_list[i]][0].append(item_list[i])
            all_sequences[user_list[i]][1].append(correct_list[i])
        else:
            all_sequences[user_list[i]] = [[item_list[i]],[correct_list[i]]]
    #all_sequences处理后得到字典：{学生ID:[[题目ID1,题目ID2，...]，[答题结果1,答题结果2，...]]} 题目和结果对应（数量相等）

    #切分训练集和测试集
    '''与下一个函数的核心区别（train_triplet） 适用于不关心序列内部顺序的模型'''
    # train_triplet [[userid,itemid,corect],...] 
    # eg：[[0, 10, 1], [0, 12, 1], [1, 11, 0]]罗列每条数据 学生ID可能重复
    # test_triplet [[userid,itemid,corect],...]
    #对于每个学生，把他序列的倒数第二个位置作为训练集的终点。
    train_triplet = []
    test_triplet = []
    for user in all_sequences:
        sequence_length = all_sequences[user][0].__len__()
        train_length = sequence_length-1
        for index in range(sequence_length):
            if index < train_length:
                train_triplet.append([user,
                                      all_sequences[user][0][index],
                                      all_sequences[user][1][index]])
            else:
                test_triplet.append([user,
                                      all_sequences[user][0][index],
                                      all_sequences[user][1][index]])

    return user_num,item_num,skill_num,record_num,train_triplet,test_triplet,Q_matrix


def get_split_sequences(dataset, type, min_length):
    [user_list, item_list, correct_list, Q_matrix] = load_dataset(dataset, type, min_length)
    user_num = max(user_list) + 1
    item_num = max(item_list) + 1
    skill_num = max([max(i) for i in Q_matrix]) + 1
    record_num = user_list.__len__()
    # all_sequences = {userid:[[itemids,...],[correct,...]]}
    all_sequences = {}
    for i in range(user_list.__len__()):
        if all_sequences.__contains__(user_list[i]):
            all_sequences[user_list[i]][0].append(item_list[i])
            all_sequences[user_list[i]][1].append(correct_list[i])
        else:
            all_sequences[user_list[i]] = [[item_list[i]], [correct_list[i]]]

    train_sequences = {}
    test_triplet = []
    '''与上一个函数的核心区别（train_sequences） 适用于必须处理完整序列的模型'''
    # train_sequences = {userid:[[itemids,...],[correct,...]]}
    # test_triplet [[userid,itemid,corect],...]
    for user in all_sequences:
        sequence_length = all_sequences[user][0].__len__()
        train_length = sequence_length - 1
        train_sequences[user] = [[all_sequences[user][0][0:train_length]],
                                 [all_sequences[user][1][0:train_length]]]
        test_item_sequence = all_sequences[user][0][train_length:]
        test_correct_sequence = all_sequences[user][1][train_length:]
        for i in range(test_item_sequence.__len__()):
            test_triplet.append([user,test_item_sequence[i],test_correct_sequence[i]])
    return user_num,item_num,skill_num,record_num,train_sequences,test_triplet,Q_matrix


'''以 AICFE 数据集的 math 学科为例，只保留答题数不少于 10 的学生，并执行一遍完整的
数据加载、过滤和重新编号流程。这通常用于调试和验证'''
if __name__ == '__main__':
    dataset = 'ASSISTment2009'
    type = 'RandomIterateSection'
    #dataset = 'AICFE'
    #type = 'math'
    min_length = 10
    load_dataset(dataset, type, min_length)

'''——————————————————添加五折交叉验证——————————————————————————'''
def get_k_fold_split_sequences(dataset, type, min_length, k=5, random_seed=42):
    """
    实现五折交叉验证的数据划分
    将用户数据集随机打乱并均分为5个互斥子集
    """
    [user_list, item_list, correct_list, Q_matrix] = load_dataset(dataset, type, min_length)
    
    # 按用户分组序列
    all_sequences = {}
    for i in range(len(user_list)):
        user = user_list[i]
        if user not in all_sequences:
            all_sequences[user] = [[], []]
        all_sequences[user][0].append(item_list[i])
        all_sequences[user][1].append(correct_list[i])
    
    # 过滤短序列
    filtered_sequences = {u: seq for u, seq in all_sequences.items() 
                         if len(seq[0]) >= min_length}
    
    # 获取用户列表并随机打乱
    user_ids = list(filtered_sequences.keys())
    np.random.seed(random_seed)
    np.random.shuffle(user_ids)
    
    # 创建k折
    folds = []
    fold_size = len(user_ids) // k
    
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k-1 else len(user_ids)
        test_users = set(user_ids[start:end])
        
        train_sequences = {}
        test_triplet = []
        
        for user, seq in filtered_sequences.items():
            if user in test_users:
                # 测试集：使用序列的最后20%作为测试（确保有足够测试数据）
                seq_len = len(seq[0])
                test_size = max(1, int(seq_len * 0.2))  # 至少保留1个测试样本
                train_size = seq_len - test_size
                
                # 添加训练序列（如果有的话）
                if train_size > 0:
                    train_sequences[user] = [seq[0][:train_size], seq[1][:train_size]]
                
                # 添加测试样本
                for j in range(test_size):
                    test_triplet.append([user, seq[0][train_size + j], seq[1][train_size + j]])
            else:
                # 训练集：使用完整序列
                train_sequences[user] = seq
        
        folds.append((train_sequences, test_triplet, Q_matrix))
    
    return folds
