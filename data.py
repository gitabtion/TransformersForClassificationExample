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
        self.texts = [line['text'] for line in data]
        self.labels = [line['label'] for line in data]

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
        return self.texts[index], self.labels[index]


def get_loader(data_obj, tokenizer, batch_size=4, shuffle=False):
    """
    定义数据加载器，可批量加载并简单处理数据
    :param data_obj:
    :param tokenizer: BERT Tokenizer
    :param batch_size:
    :param shuffle: 是否打乱顺序
    :return:
    """
    dataset = BertClfDataset(data_obj)

    def _collate_fn(data):
        """
        在输入模型前的最后一道数据处理
        :param data:
        :return:
        """
        texts, labels = zip(*data)
        labels = torch.tensor(labels, dtype=torch.long)
        inputs = tokenizer(list(texts), padding=True, return_tensors='pt')
        return inputs.data, labels

    loader = DataLoader(dataset, batch_size, shuffle, collate_fn=_collate_fn)
    return loader


def preprocess(raw_data_fn):
    """
    处理数据
    :param raw_data_fn: 原始数据文件名
    :return:
    """
    # 读取文件，可按需修改
    # with open(raw_data_fn, 'r', encoding='utf8') as f:
    #     raw_data = json.loads(f.read())

    data = list()
    """
    请进行数据处理，raw_data转化为BertClfData所需的“data”
    我们约定：
        - data为一个字典列表，每个字典包含以下字段
            - text(str)： 文本，
            - label(int): 该文本所属的类别
            
        如：["语言与智能系统实验室", "我们是一家人"],且其分类分别为1，2
        则data为：
        [{'text':"语言与智能系统实验室", 'label':1},
         {'text':"我们是一家人", 'label':2}]
    此外，须在此处拆分出训练集与测试集
    
    """

    ### 以下需自行进行数据处理
    data = [{'text': "语言与智能系统实验室", 'label': 1},
            {'text': "我们是一家人", 'label': 2}]

    train_data, test_data = data, data
    ### 以上需自行进行数据处理

    return train_data, test_data
