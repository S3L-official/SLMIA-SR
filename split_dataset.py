
import argparse
import os
import warnings
import numpy as np
import math
from tqdm import tqdm

def parse_args(input_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vox2', choices=['vox2', 'ls'], help='dataset name vox2: VoxCeleb2, ls: LibriSpeech')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory')
    parser.add_argument('--num_spks_split', type=int, nargs='+', default=None, 
                        help='num_spks_split format: #imposters #train-speakers-of-target #non-train-speakers-of-target #train-speakers-of-shadow #non-train-speakers-of-shadow')
    parser.add_argument('--seed', '-seed', type=int)
    args = parser.parse_args() if input_args is None else parser.parse_args(input_args)

    return args

def main(args):

    np.random.seed(args.seed)

    dataset_root = os.path.abspath(os.path.join(args.data_dir, f'{args.dataset}-wav'))
    assert os.path.exists(dataset_root), f'{dataset_root} does not exist! please run prepare_dataset.py.'

    des_root = f'{args.data_dir}/{args.dataset}-wav-split'

    print('gather all infos...')
    infos = []
    for sub in sorted(os.listdir(dataset_root)):
        sub_dir = os.path.join(dataset_root, sub)
        if not os.path.isdir(sub_dir):
            continue
        for spk_id in sorted(os.listdir(sub_dir)):
            spk_dir = os.path.join(sub_dir, spk_id)
            if not os.path.isdir(spk_dir):
                continue
            chaps = []
            names = []
            for chap in sorted(os.listdir(spk_dir)):
                chap_dir = os.path.join(spk_dir, chap)
                if not os.path.isdir(chap_dir):
                    continue
                for name in sorted(os.listdir(chap_dir)):
                    if name[-4:] != '.wav':
                        continue
                    chaps.append(chap)
                    names.append(name)
            infos.append((spk_id, sub, chaps, names))
    np.random.shuffle(infos)
    print('gather all infos done')

    if args.num_spks_split is None:
        args.num_spks_split = [1222, 1222, 1222, 1222, 1222] if args.dataset == 'vox2' else [400, 521, 521, 521, 521]
    assert len(args.num_spks_split) == 5, \
        'num_spks_split format: #imposters #train-speakers-of-target #non-train-speakers-of-target #train-speakers-of-shadow #non-train-speakers-of-shadow'
    assert args.num_spks_split[0] + args.num_spks_split[1] + args.num_spks_split[2] + args.num_spks_split[3] + args.num_spks_split[4] <= len(infos), \
        'num_spks_split should be less than the number of speakers in the dataset'
    if args.num_spks_split[2] < 1000:
        warnings.warn('the number of non-training speakers of the target SRS is less than 1000, so the metric TPR @ 0.1\% FPR cannot be computed.')
    if args.num_spks_split[2] < 500:
        warnings.warn('the number of non-training speakers of the target SRS is less than 500, so the metric TPR @ 0.2\% FPR cannot be computed.')
    if args.num_spks_split[2] == 1000:
        warnings.warn('the number of non-training speakers of the target SRS is exactly 1000, \
                      consider using slightly more speakers for more precisely computing the metric TPR @ 0.1\% FPR')
    if args.num_spks_split[2] == 500:
        warnings.warn('the number of non-training speakers of the target SRS is exactly 500, \
                      consider using slightly more speakers for more precisely computing the metric TPR @ 0.2\% FPR')

    idxs = [[0, args.num_spks_split[0]]]
    for num_spks in args.num_spks_split[1:]:
        idxs.append([idxs[-1][-1], idxs[-1][-1]+num_spks])
    flags = ['imposter', 'target-train_speaker', 'target-non_train_speaker', 'shadow-train_speaker', 'shadow-non_train_speaker']
    for flag, idx in zip(flags, idxs):
        
        my_infos = infos[idx[0]:idx[1]]
        des_dir = os.path.join(des_root, flag)
        if not os.path.exists(des_dir):
            os.makedirs(des_dir, exist_ok=True)
        if flag == 'imposter':
            for spk_id, sub, _, _ in tqdm(my_infos, desc=f'creating {flag} dataset'):
                spk_dir = os.path.join(dataset_root, sub, spk_id)
                des_spk_dir = os.path.join(des_dir, spk_id)
                if not os.path.exists(des_spk_dir):
                    os.symlink(spk_dir, des_spk_dir)
        else:
            for spk_id, sub, chaps, names in my_infos:
                all_index = list(range(len(chaps)))
                np.random.shuffle(all_index)
                for flag_2, idx_2 in zip([f'{flag}-train_voice', f'{flag}-non_train_voice'], 
                                        [[0, math.ceil(len(all_index) / 2)], [math.ceil(len(all_index) / 2), len(all_index)]]):
                    if 'non_train_speaker-train_voice' in flag_2:
                        continue
                    des_dir_2 = os.path.join(des_dir, flag_2)
                    for my_index in tqdm(all_index[idx_2[0]:idx_2[1]], desc=f'creating {flag_2} dataset for spk {spk_id}'):
                        chap = chaps[my_index]
                        name = names[my_index]
                        src_path = os.path.join(dataset_root, sub, spk_id, chap, name)
                        des_dir_3 = os.path.join(des_dir_2, spk_id, chap)
                        if not os.path.exists(des_dir_3):
                            os.makedirs(des_dir_3, exist_ok=True)
                        des_path = os.path.join(des_dir_3, name)
                        if not os.path.exists(des_path):
                            os.symlink(src_path, des_path)


if __name__ == '__main__':
    
    args = parse_args()
    main(args)