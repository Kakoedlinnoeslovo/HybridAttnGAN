import torch
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable

from cfg import configs as configs
from dataset.datasetZSL import FeatDataLayer, LoadDataset


class TextDataset(data.Dataset):
    def __init__(self, type="train"):
        self.dataset = LoadDataset(configs)
        if type == "train":
            self.data_layer = FeatDataLayer(self.dataset.labels_train, self.dataset.pfc_feat_data_train, configs)
        else:
            self.data_layer = FeatDataLayer(self.dataset.labels_test, self.dataset.pfc_feat_data_test, configs)
        self.type = type


    def __getitem__(self, index):
        blobs = self.data_layer.forward()
        feat_data = blobs['data']  # image data
        labels = blobs['labels'].astype(int)  # class labels
        if self.type == "train":
            text_feat = np.array([self.dataset.train_text_feature[i, :] for i in labels])
        else:
            text_feat = np.array([self.dataset.test_text_feature[i, :] for i in labels])

        text_feat = Variable(torch.from_numpy(text_feat.astype('float32'))).to(configs.device)
        img_feat = Variable(torch.from_numpy(feat_data)).to(configs.device)
        y_true = Variable(torch.from_numpy(labels.astype('int'))).to(configs.device)

        return img_feat, text_feat, y_true

    def __len__(self):
        return len(self.dataset.train_text_feature)

