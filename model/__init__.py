from model.scm.Bert import SCMBert
from model.scm.CNN import SCMCNN
from model.scm.bidaf import BiDAF
from model.scm.ABCNN import SCMABCNN
from model.scm.SMASH import SCMSMASH

model_list = {
    "SCMBert": SCMBert,
    "SCMCNN": SCMCNN,
    "SCMBiDAF": BiDAF,
    "SCMABCNN": SCMABCNN,
    "SCMSMASH": SCMSMASH
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
