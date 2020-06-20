import torch
import os
from data import get_loader, preprocess
from transformers import BertForSequenceClassification, BertConfig

# 定义配置项
model_name = 'prev_trained_model/bert-base-chinese'
raw_data = 'data.json'
batch_size = 4
lr = 1e-3
epochs = 10


def train_entry():
    # 加载数据
    train_data, test_data = preprocess(raw_data)
    train_loader = get_loader(train_data, batch_size=batch_size, shuffle=True)
    # 测试集无需打乱顺序
    test_loader = get_loader(train_data, batch_size=8, shuffle=False)

    # 加载模型及配置方法
    bert_config = BertConfig.from_pretrained(model_name, num_labels=15)  # 头条文本分类数据集为15类
    model = BertForSequenceClassification.from_pretrained(model_name, config=bert_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        train(model, optimizer, train_loader)
        # 评测时停止计算梯度
        with torch.no_grad():
            test(model, train_loader)


def train(model, optimizer, data_loader):
    """
    训练，请自行实现
    :param model:
    :param optimizer:
    :param data_loader:
    :return:
    """
    pass


def test(model, test_loader):
    """
    测试，请自行实现
    :param model:
    :param test_loader:
    :return:
    """
    for *x, y in test_loader:
        outputs = model(*x, labels=y)
        print(outputs[0])


if __name__ == '__main__':
    train_entry()
