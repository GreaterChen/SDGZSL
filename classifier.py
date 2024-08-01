import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.nn.init as init


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias, 0.0)

class CLASSIFIER:
    # train_Y is integer
    def __init__(self, opt, _train_X, _train_Y, data_loader, test_seen_feature, test_unseen_feature, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20,
                 _batch_size=100,  generalized=True, MCA=True):
        self.train_X =  _train_X
        self.train_Y = _train_Y
        self.test_seen_feature = test_seen_feature
        self.test_seen_label = data_loader.test_seen_label

        self.test_unseen_feature = test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label

        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.ntrain_class = data_loader.ntrain_class
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.shape[1]
        self.cuda = _cuda
        self.MCA = MCA
        self.model =  LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        self.model.apply(weights_init)
        self.criterion = nn.NLLLoss()
        self.opt = opt
        self.input = torch.FloatTensor(_batch_size, self.input_dim)
        self.label = torch.LongTensor(_batch_size)

        self.lr = _lr
        self.beta1 = _beta1
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))

        if self.cuda:
            self.model.to(opt.gpu)
            self.criterion.to(opt.gpu)
            self.input = self.input.to(opt.gpu)
            self.label = self.label.to(opt.gpu)

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.shape[0]

        if generalized:
            self.acc_seen, self.acc_unseen, self.H = self.fit()
        else:
            self.acc = self.fit_zsl()

    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        last_loss_epoch = 1e8
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                mean_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            if acc > best_acc:
                best_acc = acc
        return best_acc * 100

    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()

            acc_seen, probs_seen, preds_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label)
            acc_unseen, probs_unseen, preds_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label+self.ntrain_class)
            H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
            print('acc_seen=%.4f, acc_unseen=%.4f, h=%.4f' % (acc_seen, acc_unseen, H))
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
                self.save_results(probs_seen, preds_seen, self.test_seen_label, probs_unseen, preds_unseen, self.test_unseen_label, best_seen, best_unseen, best_H, epoch)
        return best_seen * 100, best_unseen * 100, best_H * 100

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]

    def val_gzsl(self, test_X, test_label):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        all_probs = []
        with torch.no_grad():
            for i in range(0, ntest, self.batch_size):
                end = min(ntest, start+self.batch_size)
                if self.cuda:
                    output = self.model(test_X[start:end].to(self.opt.gpu))
                else:
                    output = self.model(test_X[start:end])
                probs = torch.softmax(output, dim=1).cpu().numpy()
                _, predicted_label[start:end] = torch.max(output.data, 1)
                all_probs.append(probs)
                start = end

        all_probs = np.vstack(all_probs)
        if self.MCA:
            acc = self.eval_MCA(predicted_label.numpy(), test_label.numpy())
        else:
            acc = (predicted_label.numpy() == test_label.numpy()).mean()
        return acc, all_probs, predicted_label.numpy()

    def eval_MCA(self, preds, y):
        cls_label = np.unique(y)
        acc = list()
        for i in cls_label:
            acc.append((preds[y == i] == i).mean())
        return np.asarray(acc).mean()

    def save_results(self, probs_seen, preds_seen, true_labels_seen, probs_unseen, preds_unseen, true_labels_unseen, best_seen, best_unseen, best_H, epoch):
        # 将 CUDA Tensor 移动到 CPU 并转换为 NumPy 数组
        true_labels_seen = true_labels_seen.cpu().numpy()
        
        if probs_unseen is not None:
            seen_df = pd.DataFrame(probs_seen, columns=[f'class_{i}' for i in range(probs_seen.shape[1])])
            seen_df['true_label'] = true_labels_seen
            seen_df['predicted_label'] = preds_seen

            true_labels_unseen = true_labels_unseen.cpu().numpy()
            
            unseen_df = pd.DataFrame(probs_unseen, columns=[f'class_{i}' for i in range(probs_unseen.shape[1])])
            unseen_df['true_label'] = true_labels_unseen
            unseen_df['predicted_label'] = preds_unseen

            all_df = pd.concat([seen_df, unseen_df])
            filename = f'/home/LAB/chenlb24/compare_model/SDGZSL/out/ZDFY/results_h_{best_H:.4f}_acc_{best_seen:.4f}_unseen_acc_{best_unseen:.4f}_epoch_{epoch}.csv'
        else:
            all_df = pd.DataFrame(probs_seen, columns=[f'class_{i}' for i in range(probs_seen.shape[1])])
            all_df['true_label'] = true_labels_seen
            all_df['predicted_label'] = preds_seen
            filename = f'results_acc_{best_seen:.4f}_epoch_{epoch}.csv'

        all_df.to_csv(filename, index=False)


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x):
        o = self.logic(self.fc(x))
        return o