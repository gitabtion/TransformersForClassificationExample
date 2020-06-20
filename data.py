import torch

from torch.utils.data import Dataset, DataLoader
import json

class BertClfDataset(Dataset):
    """
    用于BERT文本分类的数据集，该处定义其数据结构
    """

    def __init__(self, data):
        """
        bert的输入需要以下三种数据，labels为文本类别用于训练及评测。
        :param data:
        """
        self.ids = data['ids']
        self.attention_masks = data['attention_masks']
        self.token_type_ids = data['token_type_ids']
        self.labels = data['labels']

    def __len__(self):
        """
        数据集总大小
        :return:
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        从数据集中获取单条数据
        :param index: 索引
        :return:
        """
        return self.ids[index], self.attention_masks[index], self.token_type_ids[index], self.labels[index]


def get_loader(data, batch_size=4, shuffle=False):
    """
    定义数据加载器，可批量加载并简单处理数据
    :param data:
    :param batch_size:
    :param shuffle: 是否打乱顺序
    :return:
    """
    dataset = BertClfDataset(data)

    loader = DataLoader(dataset, batch_size, shuffle, collate_fn=_collate_fn)
    return loader


def _collate_fn(data):
    """
    在输入模型前的最后一道数据处理
    :param data:
    :return:
    """
    ids, attention_masks, token_type_ids, labels = zip(*data)
    ids = torch.tensor(ids).long()
    attention_masks = torch.tensor(attention_masks).long()
    token_type_ids = torch.tensor(token_type_ids).long()
    labels = torch.tensor(labels).long()
    return ids, attention_masks, token_type_ids, labels


def preprocess(raw_data_fn):
    """
    处理数据
    :param raw_data_fn: 原始数据文件名
    :return:
    """
    # 读取文件，可按需修改
    # with open(raw_data_fn, 'r', encoding='utf8') as f:
    #     raw_data = json.loads(f.read())

    data = dict()
    """
    请进行数据处理，raw_data转化为BertClfData所需的“data”
    我们约定：
        - data为一个字典，并且包含以下字段
            - ids： 所有句子的one-hot
            - token_type_ids： 用于区分句子，详情请查看文档
            - attention_masks： 标示出有用信息，非padding部分为1，padding部分为0
            - labels: 句子所属类别
            
        如：["语言与智能系统实验室", "我们是一家人"],且其分类分别为1，2
        则data为：
        {
            "ids": [
                [101, 6427, 6241, 680, 3255, 5543, 5143, 5320, 2141, 7741, 2147, 102]+[0]*(512-12),
                [101, 2769, 812, 3221, 671, 2157, 782, 102] + [0]*(512-8)],
            "attention_masks": [
                [1] * 12 + [0] * (512-12),
                [1] * 8 + [0] * (512-8)],
            "token_type_ids": [[0]*512, [0]*512],
            "labels": [1, 2]
        }
    此外，须在此处拆分出训练集与测试集
    
    """

    ### 以下需自行进行数据处理
    data['ids'] = [[101, 6427, 6241, 680, 3255, 5543, 5143, 5320, 2141, 7741, 2147, 102] + [0] * (512 - 12),
                   [101, 2769, 812, 3221, 671, 2157, 782, 102] + [0] * (512 - 8)]
    data['attention_masks'] = [[1] * 12 + [0] * (512 - 12), [1] * 8 + [0] * (512 - 8)]
    data['token_type_ids'] = [[0] * 512, [0] * 512]
    data['labels'] = [1, 2]

    train_data, test_data = data, data
    ### 以上需自行进行数据处理

    return train_data, test_data
