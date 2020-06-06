from model.element.Bert import Bert
from model.element.CNN import CNN
from model.element.LSTM import LSTM
from model.element.DPCNN import DPCNN
from model.element.BiDAF import BiDAF
from model.element.BertAtt import BertAtt

model_list = {
    "Bert": Bert,
    "CNN": CNN,
    "DPCNN": DPCNN,
    "LSTM": LSTM,
    "BiDAF": BiDAF,
    "BertAtt": BertAtt
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
