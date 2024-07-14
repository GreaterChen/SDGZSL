import json
import os
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import torch

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label


class DATA_LOADER(object):
    def __init__(self, opt):
        self.finetune = opt.finetune
        if opt.dataset in ['FLO_EPGN','CUB_STC']:
            if self.finetune:
                self.read_fine_tune(opt)
            else:
                self.read(opt)
        elif opt.dataset in ['CUB', 'AWA2', 'APY', 'FLO', 'SUN']:
            self.read_matdataset(opt)
        elif opt.dataset == "ZDFY":
            self.read_turmor(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.feature_dim = self.train_feature.shape[1]
        self.att_dim = self.attribute.shape[1]
        self.text_dim = self.att_dim
        self.tr_cls_centroid = np.zeros([self.seenclasses.shape[0], self.feature_dim], np.float32)
        for i in range(self.seenclasses.shape[0]):
            self.tr_cls_centroid[i] = np.mean(self.train_feature[self.train_label == i].numpy(), axis=0)

    def read_fine_tune(self,opt):
        if opt.dataset == "CUB_STC":
            opt.dataset = "CUB"
        if opt.dataset == "FLO_EPGN":
            opt.dataset = "FLO"

        # matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        # feature = matcontent['features'].T
        # label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        # matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/cub_feat.mat")
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + "_finetuned.mat")

        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        # feature = matcontent['features'].T
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/data.mat")
        # feature = matcontent['features'].T
        # label = matcontent['labels'].astype(int).squeeze() - 1
        train_att = matcontent['att_train']
        seen_pro = matcontent['seen_pro']
        attribute = matcontent['attribute']
        unseen_pro = matcontent['unseen_pro']
        self.attribute = torch.from_numpy(attribute).float()
        self.train_att = seen_pro.astype(np.float32)
        self.test_att = unseen_pro.astype(np.float32)

        scaler = preprocessing.MinMaxScaler()
        _train_feature = scaler.fit_transform(feature[trainval_loc])
        _test_seen_feature = scaler.transform(feature[test_seen_loc])
        _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
        self.train_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_feature.max()
        self.train_feature.mul_(1 / mx)
        self.train_label = torch.from_numpy(label[trainval_loc]).long()
        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
        self.test_unseen_feature.mul_(1 / mx)
        self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1 / mx)
        self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.train_label = map_label(self.train_label, self.seenclasses)
        self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
        self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)

        self.train_att = self.attribute[self.seenclasses].numpy()
        self.test_att = self.attribute[self.unseenclasses].numpy()

    def read(self, opt):
        if opt.dataset == "CUB_STC":
            opt.dataset = "CUB"
        if opt.dataset == "FLO_EPGN":
            opt.dataset = "FLO"

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/data.mat")
        train_att = matcontent['att_train']
        seen_pro = matcontent['seen_pro']
        attribute = matcontent['attribute']
        unseen_pro = matcontent['unseen_pro']
        self.attribute = torch.from_numpy(attribute).float()
        self.train_att = seen_pro.astype(np.float32)
        self.test_att = unseen_pro.astype(np.float32)

        train_fea = matcontent['train_fea']
        test_seen_fea = matcontent['test_seen_fea']
        test_unseen_fea = matcontent['test_unseen_fea']

        scaler = preprocessing.MinMaxScaler()
        _train_feature = scaler.fit_transform(train_fea)
        _test_seen_feature = scaler.transform(test_seen_fea)
        _test_unseen_feature = scaler.transform(test_unseen_fea)
        mx = _train_feature.max()
        train_fea = train_fea * (1 / mx)
        test_seen_fea = test_seen_fea * (1 / mx)
        test_unseen_fea = test_unseen_fea * (1 / mx)

        self.train_feature = torch.from_numpy(train_fea).float()
        self.test_seen_feature = torch.from_numpy(test_seen_fea).float()
        self.test_unseen_feature = torch.from_numpy(test_unseen_fea).float()

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/label.mat")

        train_idx = matcontent['train_idx'] - 1
        train_label = matcontent['train_label_new']
        test_unseen_idex = matcontent['test_unseen_idex'] - 1
        test_seen_idex = matcontent['test_seen_idex'] - 1
        self.train_label = torch.from_numpy(train_idx.squeeze()).long()
        self.test_seen_label = torch.from_numpy(test_seen_idex.squeeze()).long()
        self.test_unseen_label = torch.from_numpy(test_unseen_idex.squeeze()).long()

        self.seenclasses = torch.from_numpy(np.unique(self.test_seen_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
        self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)


    def read_turmor(self, opt):
        self.train_class = [1, 2]

        # 加载训练特征和标签并转换为 Tensor
        self.train_feature = np.load(os.path.join(opt.dataroot, 'resnet101', 'train_features.npy'))  # (606, 2048)
        self.train_label = np.load(os.path.join(opt.dataroot, 'resnet101', 'train_targets.npy'))
        self.train_feature = torch.tensor(self.train_feature, dtype=torch.float32)
        self.train_label = torch.tensor(self.train_label, dtype=torch.long)

        self.ntrain = self.train_feature.shape[0]
        self.ntrain_class = 2
        self.ntest_class = 1

        # 加载测试特征和标签并转换为 Tensor
        self.test_feature = np.load(os.path.join(opt.dataroot, 'resnet101', 'valid_features.npy'))  # (171, 2048)
        self.test_label = np.load(os.path.join(opt.dataroot, 'resnet101', 'valid_targets.npy'))
        self.test_feature = torch.tensor(self.test_feature, dtype=torch.float32)
        self.test_label = torch.tensor(self.test_label, dtype=torch.long)

        # 加载属性嵌入并转换为 Tensor
        file_path = os.path.join(opt.dataroot, 'att', 'embeddings.json')
        with open(file_path, 'r') as f:
            data = json.load(f)

        attribute = {}
        for key, value in data.items():
            attribute[key] = np.array(value)

        categories = list(attribute.keys())
        embedding_list = [attribute[category] for category in categories]
        self.attribute = torch.tensor(np.array(embedding_list), dtype=torch.float32)

        self.allclasses = torch.from_numpy(np.array([0, 1, 2]))
        self.seenclasses = torch.from_numpy(np.array([1, 2]))
        self.unseenclasses = torch.from_numpy(np.array([0]))
        self.attribute_seen = self.attribute[self.seenclasses]
        
        self.train_att = self.attribute_seen.numpy()
        self.test_att = self.attribute[self.unseenclasses].numpy()

        # 提取标签为 1 和 2 的测试数据
        indices_seen = (self.test_label == 1) | (self.test_label == 2)
        self.test_seen_feature = self.test_feature[indices_seen]
        self.test_seen_label = self.test_label[indices_seen]

        # 提取标签为 0 的测试数据
        indices_unseen = self.test_label == 0
        self.test_unseen_feature = self.test_feature[indices_unseen]
        self.test_unseen_label = self.test_label[indices_unseen]

        # 计算每个类别的样本数量
        self.train_samples_class_index = torch.tensor([self.train_label.eq(i_class).sum().float() for i_class in self.train_class])
        
        
        self.train_label = map_label(self.train_label, self.seenclasses)
        self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
        self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)

        # 打印结果以确认
        print("训练集特征形状:", self.train_feature.shape)
        print("训练集标签形状:", self.train_label.shape)
        print("测试集特征形状:", self.test_feature.shape)
        print("测试集标签形状:", self.test_label.shape)
        print("提取的测试集特征形状 (seen):", self.test_seen_feature.shape)
        print("提取的测试集标签形状 (seen):", self.test_seen_label.shape)
        print("提取的测试集特征形状 (unseen):", self.test_unseen_feature.shape)
        print("提取的测试集标签形状 (unseen):", self.test_unseen_label.shape)
        print("每个类别的样本数量:", self.train_samples_class_index)
    
    
    
    def read_matdataset(self, opt):
        


        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        label = matcontent['labels'].astype(int).squeeze() - 1
        if self.finetune:
            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + "_finetuned.mat")
            # label = matcontent['labels'].astype(int).squeeze() - 1

        feature = matcontent['features'].T
        if opt.dataset == "APY" and self.finetune:
            feature = feature.T
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        # if opt.dataset == "FLO":
        #     temp_norm = torch.norm(self.attribute, p=2, dim=1).unsqueeze(1).expand_as(self.attribute)
        #     self.attribute = self.attribute.div(temp_norm + 1e-5)

        #
        #
        # matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/data.mat")
        #
        # train_att = matcontent['att_train']
        # seen_pro = matcontent['seen_pro']
        # attribute = matcontent['attribute']
        # unseen_pro = matcontent['unseen_pro']
        # self.attribute = torch.from_numpy(attribute).float()
        # self.train_att = seen_pro.astype(np.float32)
        # self.test_att = unseen_pro.astype(np.float32)

        scaler = preprocessing.MinMaxScaler()
        _train_feature = scaler.fit_transform(feature[trainval_loc])
        _test_seen_feature = scaler.transform(feature[test_seen_loc])
        _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
        self.train_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_feature.max()
        self.train_feature.mul_(1 / mx)
        self.train_label = torch.from_numpy(label[trainval_loc]).long()
        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
        self.test_unseen_feature.mul_(1 / mx)
        self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1 / mx)
        self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.train_label = map_label(self.train_label, self.seenclasses)
        self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
        self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)

        self.train_att = self.attribute[self.seenclasses].numpy()
        self.test_att = self.attribute[self.unseenclasses].numpy()


class FeatDataLayer(object):
    def __init__(self, label, feat_data,  opt):
        """Set the roidb to be used by this layer during training."""
        assert len(label) == feat_data.shape[0]
        self._opt = opt
        self._feat_data = feat_data
        self._label = label
        self._shuffle_roidb_inds()
        self._epoch = 0

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._label)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + self._opt.batchsize >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1

        db_inds = self._perm[self._cur:self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize

        return db_inds

    def forward(self):
        new_epoch = False
        if self._cur + self._opt.batchsize >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1
            new_epoch = True

        db_inds = self._perm[self._cur:self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize

        minibatch_feat = np.array([self._feat_data[i] for i in db_inds])
        minibatch_label = np.array([self._label[i] for i in db_inds])
        blobs = {'data': minibatch_feat, 'labels': minibatch_label, 'newEpoch': new_epoch, 'idx': db_inds}
        return blobs

    def get_whole_data(self):
        blobs = {'data': self._feat_data, 'labels': self._label}
        return blobs
