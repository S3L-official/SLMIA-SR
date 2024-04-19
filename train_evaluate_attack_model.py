
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn as nn
import torch
import time
import copy

def parse_args(input_args=None):
    
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-system_type', type=str, default='LSTM_GE2E')
    parser.add_argument('-dataset', type=str, default='vox2')

    parser.add_argument('-num', type=int, default=None)
    parser.add_argument('-imposter_num', type=int, default=None)
    parser.add_argument('-imposter_voice', type=int, default=None)
    # parser.add_argument('-imposter_voice_max', type=int, default=None)

    parser.add_argument('-voice_chunk_splitting', action='store_true', default=False)
    parser.add_argument('-imposter_VCS', action='store_true', default=False)
    parser.add_argument('-partial_utterance_n_frames', type=int, default=160) # voice_chunk_splitting factor, recommended: 320

    parser.add_argument('-voice_label', type=str, default='train', choices=['train']) # will automatic using mixing ratio training
    parser.add_argument('-voice_label_2', type=str, default='nontrain', choices=['nontrain'])

    parser.add_argument('-sim', type=str, default='cosine', choices=['cosine', 'L2'])
    parser.add_argument('-seed', type=int, nargs='+', default=[0, 111, 222, 333, 444, 555, 666, 777, 888, 999])

    parser.add_argument('-fpr', type=float, default=0.1, choices=[0.1, 0.2])

    args = parser.parse_args() if input_args is None else parser.parse_args(input_args)
    return args


def main(args):

    train_X = None
    train_Y = None
    test_X = None
    test_Y = None
    valid_spks = {}
    for flag, using_metrics in zip(['compact', 'far'], [list(range(24)), list(range(96))]):

        train_X_flag = None
        train_Y_flag = None

        args.model = 'shadow'
        model_flag = ''
        member_train_file = 'FE_outputs/MI_score_{}-{}{}-{}-member-{}-{}.txt'.format(flag, args.dataset, model_flag, args.model, args.voice_label, args.sim)
        non_member_file = 'FE_outputs/MI_score_{}-{}{}-{}-nonmember-{}-{}.txt'.format(flag, args.dataset, model_flag, args.model, args.voice_label_2, args.sim)   

        if args.num is not None:
            member_train_file = member_train_file[:-4] + '-num={}'.format(args.num) + '.txt'
            non_member_file = non_member_file[:-4] + '-num={}'.format(args.num) + '.txt'

        if args.voice_chunk_splitting:
            member_train_file = member_train_file[:-4] + '-aug-part' + '.txt'
            non_member_file = non_member_file[:-4] + '-aug-part' + '.txt'
        
        if args.partial_utterance_n_frames != 160:
            member_train_file = member_train_file[:-4] + '-p={}.txt'.format(args.partial_utterance_n_frames)
            non_member_file = non_member_file[:-4] + '-p={}.txt'.format(args.partial_utterance_n_frames)
            if args.imposter_VCS and flag == 'far':
                member_train_file = member_train_file[:-4] + '-imposter-part.txt'
                non_member_file = non_member_file[:-4] + '-imposter-part.txt'
        
        if flag == 'far':

            if args.imposter_num is not None:
                member_train_file = member_train_file[:-4] + 'im-num={}.txt'.format(args.imposter_num)
                non_member_file = non_member_file[:-4] + 'im-num={}.txt'.format(args.imposter_num)

            if args.imposter_voice is not None:
                member_train_file = member_train_file[:-4] + '-im-vnum={}.txt'.format(args.imposter_voice)
                non_member_file = non_member_file[:-4] + '-im-vnum={}.txt'.format(args.imposter_voice)
                    
        print(member_train_file, non_member_file)
        member = np.loadtxt(member_train_file, usecols=tuple([x+3 for x in using_metrics]), dtype=float)
        non_member = np.loadtxt(non_member_file, usecols=tuple([x+3 for x in using_metrics]), dtype=float)

        member_spks = np.loadtxt(member_train_file, usecols=(0,), dtype=str)
        non_member_spks = np.loadtxt(non_member_file, usecols=(0,), dtype=str)
        if flag == 'compact':
            valid_spks['{}-{}-{}'.format(args.model, args.num, 'm1')] = member_spks.tolist()
            valid_spks['{}-{}-{}'.format(args.model, args.num, 'nm')] = non_member_spks.tolist()
        else:
            member_2 = []
            non_member_2 = []
            for a, spk in zip(member, member_spks):
                if spk in valid_spks['{}-{}-{}'.format(args.model, args.num, 'm1')]:
                    member_2.append(a)
            for a, spk in zip(non_member, non_member_spks):
                if spk in valid_spks['{}-{}-{}'.format(args.model, args.num, 'nm')]:
                    non_member_2.append(a)
            member = np.stack(member_2)
            non_member = np.stack(non_member_2)
        
        target_member_ratio_1_len = len(member)
        target_nonmember_len = len(non_member)
        train_X_flag = np.concatenate((train_X_flag, member, non_member)) if train_X_flag is not None else np.concatenate((member, non_member))
        train_Y_flag = np.concatenate((train_Y_flag, np.ones(member.shape[0]), np.zeros(non_member.shape[0]))) if train_Y_flag is not None else np.concatenate((np.ones(member.shape[0]), np.zeros(non_member.shape[0])))

        part_ori = args.voice_label
        args.voice_label = 'nontrain' if part_ori == 'train' else 'train' # using mixing ratio training
        args.model = 'shadow'
        model_flag = ''
        member_train_file = 'FE_outputs/MI_score_{}-{}{}-{}-member-{}-{}.txt'.format(flag, args.dataset, model_flag, args.model, args.voice_label, args.sim)
        non_member_file = 'FE_outputs/MI_score_{}-{}{}-{}-nonmember-{}-{}.txt'.format(flag, args.dataset, model_flag, args.model, args.voice_label_2, args.sim)

        if args.num is not None:
            member_train_file = member_train_file[:-4] + '-num={}'.format(args.num) + '.txt'
            non_member_file = non_member_file[:-4] + '-num={}'.format(args.num) + '.txt'

        if args.voice_chunk_splitting:
            member_train_file = member_train_file[:-4] + '-aug-part' + '.txt'
            non_member_file = non_member_file[:-4] + '-aug-part' + '.txt'
        
        if args.partial_utterance_n_frames != 160:
            member_train_file = member_train_file[:-4] + '-p={}.txt'.format(args.partial_utterance_n_frames)
            non_member_file = non_member_file[:-4] + '-p={}.txt'.format(args.partial_utterance_n_frames)
            if args.imposter_VCS and flag == 'far':
                member_train_file = member_train_file[:-4] + '-imposter-part.txt'
                non_member_file = non_member_file[:-4] + '-imposter-part.txt'
        
        if flag == 'far':

            if args.imposter_num is not None:
                member_train_file = member_train_file[:-4] + 'im-num={}.txt'.format(args.imposter_num)
                non_member_file = non_member_file[:-4] + 'im-num={}.txt'.format(args.imposter_num)

            if args.imposter_voice is not None:
                member_train_file = member_train_file[:-4] + '-im-vnum={}.txt'.format(args.imposter_voice)
                non_member_file = non_member_file[:-4] + '-im-vnum={}.txt'.format(args.imposter_voice)

        print(member_train_file, non_member_file)
        member = np.loadtxt(member_train_file, usecols=tuple([x+3 for x in using_metrics]), dtype=float)
        non_member = np.loadtxt(non_member_file, usecols=tuple([x+3 for x in using_metrics]), dtype=float)

        member_spks = np.loadtxt(member_train_file, usecols=(0,), dtype=str)
        non_member_spks = np.loadtxt(non_member_file, usecols=(0,), dtype=str)
        if flag == 'compact':
            valid_spks['{}-{}-{}'.format(args.model, args.num, 'm2')] = member_spks.tolist()
            valid_spks['{}-{}-{}'.format(args.model, args.num, 'nm')] = non_member_spks.tolist()
        else:
            member_2 = []
            non_member_2 = []
            for a, spk in zip(member, member_spks):
                if spk in valid_spks['{}-{}-{}'.format(args.model, args.num, 'm2')]:
                    member_2.append(a)
            for a, spk in zip(non_member, non_member_spks):
                if spk in valid_spks['{}-{}-{}'.format(args.model, args.num, 'nm')]:
                    non_member_2.append(a)
            member = np.stack(member_2)
            non_member = np.stack(non_member_2)
        
        target_member_ratio_2_len = len(member)
        assert target_nonmember_len == len(non_member)
        train_X_flag = np.concatenate((train_X_flag, member, non_member)) if train_X_flag is not None else np.concatenate((member, non_member))
        train_Y_flag = np.concatenate((train_Y_flag, np.ones(member.shape[0]), np.zeros(non_member.shape[0]))) if train_Y_flag is not None else np.concatenate((np.ones(member.shape[0]), np.zeros(non_member.shape[0])))
        args.voice_label = part_ori

        if flag == 'compact':
            train_X = train_X_flag
            train_Y = train_Y_flag
        else:
            train_X = np.concatenate((train_X, train_X_flag), axis=1)
        
        test_X_flag = None
        test_Y_flag = None
        test_X_split_idx = []
        test_X_split_idx_2 = []
        test_SPKS = []

        args.model = 'target'
        model_flag = ''
        member_train_file = 'FE_outputs/MI_score_{}-{}{}-{}-member-{}-{}.txt'.format(flag, args.dataset, model_flag, args.model, args.voice_label, args.sim)
        non_member_file = 'FE_outputs/MI_score_{}-{}{}-{}-nonmember-{}-{}.txt'.format(flag, args.dataset, model_flag, args.model, args.voice_label_2, args.sim)

        if args.num is not None:
            member_train_file = member_train_file[:-4] + '-num={}'.format(args.num) + '.txt'
            non_member_file = non_member_file[:-4] + '-num={}'.format(args.num) + '.txt'

        if args.voice_chunk_splitting:
            member_train_file = member_train_file[:-4] + '-aug-part' + '.txt'
            non_member_file = non_member_file[:-4] + '-aug-part' + '.txt'
        
        if args.partial_utterance_n_frames != 160:
            member_train_file = member_train_file[:-4] + '-p={}.txt'.format(args.partial_utterance_n_frames)
            non_member_file = non_member_file[:-4] + '-p={}.txt'.format(args.partial_utterance_n_frames)
            if args.imposter_VCS and flag == 'far':
                member_train_file = member_train_file[:-4] + '-imposter-part.txt'
                non_member_file = non_member_file[:-4] + '-imposter-part.txt'
        
        if flag == 'far':

            if args.imposter_num is not None:
                member_train_file = member_train_file[:-4] + 'im-num={}.txt'.format(args.imposter_num)
                non_member_file = non_member_file[:-4] + 'im-num={}.txt'.format(args.imposter_num)

            if args.imposter_voice is not None:
                member_train_file = member_train_file[:-4] + '-im-vnum={}.txt'.format(args.imposter_voice)
                non_member_file = non_member_file[:-4] + '-im-vnum={}.txt'.format(args.imposter_voice)
        
        print(member_train_file, non_member_file)
        member = np.loadtxt(member_train_file, usecols=tuple([x+3 for x in using_metrics]), dtype=float)
        non_member = np.loadtxt(non_member_file, usecols=tuple([x+3 for x in using_metrics]), dtype=float)
        
        member_spks = np.loadtxt(member_train_file, usecols=(0,), dtype=str)
        non_member_spks = np.loadtxt(non_member_file, usecols=(0,), dtype=str)
        if flag == 'compact':
            valid_spks['{}-{}-{}'.format(args.model, args.num, 'm1')] = member_spks.tolist()
            valid_spks['{}-{}-{}'.format(args.model, args.num, 'nm')] = non_member_spks.tolist()
        else:
            member_2 = []
            non_member_2 = []
            for a, spk in zip(member, member_spks):
                if spk in valid_spks['{}-{}-{}'.format(args.model, args.num, 'm1')]:
                    member_2.append(a)
                    test_SPKS.append(spk)
            for a, spk in zip(non_member, non_member_spks):
                if spk in valid_spks['{}-{}-{}'.format(args.model, args.num, 'nm')]:
                    non_member_2.append(a)
                    test_SPKS.append(spk)
            member = np.stack(member_2)
            non_member = np.stack(non_member_2)

        member_non_member = np.concatenate((member, non_member))
        start_idx = test_X_flag.shape[0] if test_X_flag is not None else 0
        test_X_flag = np.concatenate((test_X_flag, member, non_member)) if test_X_flag is not None else np.concatenate((member, non_member))
        test_Y_flag = np.concatenate((test_Y_flag, np.ones(member.shape[0]), np.zeros(non_member.shape[0]))) if test_Y_flag is not None else np.concatenate((np.ones(member.shape[0]), np.zeros(non_member.shape[0]))) 
        end_idx = test_X_flag.shape[0] if test_X_flag is not None else 0
        end_idx_2 = start_idx + member.shape[0]
        test_X_split_idx.append((start_idx, end_idx))
        test_X_split_idx_2 += [(start_idx, end_idx_2), (end_idx_2, end_idx)]


        part_ori = args.voice_label
        args.voice_label = 'nontrain' if part_ori == 'train' else 'train'
        model_flag = ''
        member_train_file = 'FE_outputs/MI_score_{}-{}{}-{}-member-{}-{}.txt'.format(flag, args.dataset, model_flag, args.model, args.voice_label, args.sim)
        non_member_file = 'FE_outputs/MI_score_{}-{}{}-{}-nonmember-{}-{}.txt'.format(flag, args.dataset, model_flag, args.model, args.voice_label_2, args.sim)

        if args.num is not None:
            member_train_file = member_train_file[:-4] + '-num={}'.format(args.num) + '.txt'
            non_member_file = non_member_file[:-4] + '-num={}'.format(args.num) + '.txt'

        if args.voice_chunk_splitting:
            member_train_file = member_train_file[:-4] + '-aug-part' + '.txt'
            non_member_file = non_member_file[:-4] + '-aug-part' + '.txt'
        
        if args.partial_utterance_n_frames != 160:
            member_train_file = member_train_file[:-4] + '-p={}.txt'.format(args.partial_utterance_n_frames)
            non_member_file = non_member_file[:-4] + '-p={}.txt'.format(args.partial_utterance_n_frames)
            if args.imposter_VCS and flag == 'far':
                member_train_file = member_train_file[:-4] + '-imposter-part.txt'
                non_member_file = non_member_file[:-4] + '-imposter-part.txt'
        
        if flag == 'far':

            if args.imposter_num is not None:
                member_train_file = member_train_file[:-4] + 'im-num={}.txt'.format(args.imposter_num)
                non_member_file = non_member_file[:-4] + 'im-num={}.txt'.format(args.imposter_num)

            if args.imposter_voice is not None:
                member_train_file = member_train_file[:-4] + '-im-vnum={}.txt'.format(args.imposter_voice)
                non_member_file = non_member_file[:-4] + '-im-vnum={}.txt'.format(args.imposter_voice)
        
        print(member_train_file, non_member_file)
        member = np.loadtxt(member_train_file, usecols=tuple([x+3 for x in using_metrics]), dtype=float)
        non_member = np.loadtxt(non_member_file, usecols=tuple([x+3 for x in using_metrics]), dtype=float)

        member_spks = np.loadtxt(member_train_file, usecols=(0,), dtype=str)
        non_member_spks = np.loadtxt(non_member_file, usecols=(0,), dtype=str)
        if flag == 'compact':
            valid_spks['{}-{}-{}'.format(args.model, args.num, 'm2')] = member_spks.tolist()
            valid_spks['{}-{}-{}'.format(args.model, args.num, 'nm')] = non_member_spks.tolist()
        else:
            member_2 = []
            non_member_2 = []
            for a, spk in zip(member, member_spks):
                if spk in valid_spks['{}-{}-{}'.format(args.model, args.num, 'm2')]:
                    member_2.append(a)
                    test_SPKS.append(spk)
            for a, spk in zip(non_member, non_member_spks):
                if spk in valid_spks['{}-{}-{}'.format(args.model, args.num, 'nm')]:
                    non_member_2.append(a)
                    test_SPKS.append(spk)
            member = np.stack(member_2)
            non_member = np.stack(non_member_2)

        start_idx = test_X_flag.shape[0] if test_X_flag is not None else 0
        test_X_flag = np.concatenate((test_X_flag, member, non_member)) if test_X_flag is not None else np.concatenate((member, non_member))
        test_Y_flag = np.concatenate((test_Y_flag, np.ones(member.shape[0]), np.zeros(non_member.shape[0]))) if test_Y_flag is not None else np.concatenate((np.ones(member.shape[0]), np.zeros(non_member.shape[0]))) 
        end_idx = test_X_flag.shape[0] if test_X_flag is not None else 0
        end_idx_2 = start_idx + member.shape[0]
        test_X_split_idx.append((start_idx, end_idx))
        test_X_split_idx_2 += [(start_idx, end_idx_2), (end_idx_2, end_idx)]
        args.voice_label = part_ori

        if flag == 'compact':
            test_X = test_X_flag
            test_Y = test_Y_flag
        else:
            test_X = np.concatenate((test_X, test_X_flag), axis=1)
    

    class MLP(nn.Module):
        def __init__(self, in_dim=11, hidden_dim=64):
            super(MLP, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            )
            self.in_dim = in_dim
            
        def forward(self, x):
            assert len(x.shape) == 2
            assert x.shape[1] == self.in_dim
            x = self.layers(x)
            return x
        
        def prob(self, x):
            x = self.forward(x)
            x = torch.nn.functional.softmax(x, dim=1)
            return x

        def make_decision(self, x):
            x = self.prob(x)
            y = torch.max(x, dim=1)[1]
            return y, x
    
    train_X = torch.from_numpy(train_X).float().to('cuda')
    train_Y = torch.from_numpy(train_Y).long().to('cuda')
    test_X_ori = torch.from_numpy(test_X).float().to('cuda')
    test_Y_ori = torch.from_numpy(test_Y).long().to('cuda')

    # split train and validate
    all_idx = list(range(len(train_X)-target_nonmember_len))
    train_idx = np.random.choice(all_idx, size=int(len(all_idx) * 0.9), replace=False).tolist()
    val_idx = [idx for idx in all_idx if idx not in train_idx]

    train_idx += [idx-target_member_ratio_1_len+len(all_idx) for idx in train_idx if idx in range(target_member_ratio_1_len, target_member_ratio_1_len + target_nonmember_len)]
    val_idx += [idx-target_member_ratio_1_len+len(all_idx) for idx in val_idx if idx in range(target_member_ratio_1_len, target_member_ratio_1_len + target_nonmember_len)]

    val_X = train_X[val_idx, :]
    val_Y = train_Y[val_idx]
    train_X = train_X[train_idx, :]
    train_Y = train_Y[train_idx]

    RES = np.zeros((1, 8))
    for random_state in args.seed:
        np.random.seed(random_state)
        torch.cuda.manual_seed(random_state)
        torch.manual_seed(random_state)
        MLP_MODEL = MLP(in_dim=train_X.shape[1]).to('cuda')

        # training
        num_epoches = 10000
        optimizer = torch.optim.Adam(MLP_MODEL.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        all_accuracies = []
        f_model = None
        best_acc = -1
        for i_epoch in range(num_epoches):
            start_t = time.time()
            MLP_MODEL.train()
            outputs = MLP_MODEL(train_X)
            loss = criterion(outputs, train_Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            MLP_MODEL.eval()
            predictions, _ = MLP_MODEL.make_decision(val_X)
            acc = torch.where(predictions == val_Y)[0].size()[0] / predictions.size()[0]

            predictions, _ = MLP_MODEL.make_decision(train_X)
            acc_train = torch.where(predictions == train_Y)[0].size()[0] / predictions.size()[0]

            end_t = time.time() 
            all_accuracies.append(acc)

            if acc > best_acc:
                f_model = copy.deepcopy(MLP_MODEL)
                best_acc = acc

        with torch.no_grad():
            f_model.eval()
            return_ = []
            for start_idx, end_idx in test_X_split_idx:
                my_idx = list(range(start_idx, end_idx))
                test_X = test_X_ori[my_idx, :]
                test_Y = test_Y_ori[my_idx]

                pred, pred_p = f_model.make_decision(test_X)
                pred = pred.detach().cpu().numpy()
                pred_p = pred_p.detach().cpu().numpy()[:, 1]
                test_Y = test_Y.detach().cpu().numpy()
                acc = np.argwhere(pred == test_Y).flatten().size / len(my_idx)
                return_.append(acc)
                auroc = roc_auc_score(test_Y, pred_p)
                return_.append(auroc)

                print(acc, auroc)
                fpr, tpr, threshold = roc_curve(test_Y, pred_p, drop_intermediate=False)
                for target_fpr in [args.fpr, 1]:
                    start = np.max(np.argwhere(fpr <= target_fpr / 100).flatten())
                    print(target_fpr, fpr[start] * 100, tpr[start] * 100)
                    return_ += [tpr[start] * 100]
                
            return_ = np.array(return_).reshape(-1, 8)
            RES += return_

    RES /= len(args.seed)
    m_idx = 0
    results = {}
    for b in ['r=1', 'r=0']:
        for a in ['Accuracy', 'AUROC', 'TPR-0.1', 'TPR-1']:
            results['{}#{}'.format(a, b)] = RES[:, m_idx].tolist()
            m_idx += 1
    
    df = pd.DataFrame(results, index=['avg'])
    print(df)


if __name__ == '__main__':

    args = parse_args()
    main(args)