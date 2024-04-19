
import math
from pathlib import Path
import numpy as np
import os
import torch
import math
import torchaudio
import librosa
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

from model.SV2TTS import SV2TTS

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-data_dir', type=str, default='data', help='data directory')
parser.add_argument('-dataset', type=str, default='vox', choices=['vox2', 'ls'])

parser.add_argument('-model', type=str, default='target', choices=['target', 'shadow'])
parser.add_argument('-model_step', type=int, default=None)

parser.add_argument('-spk_label', type=str, default='member', choices=['member', 'nonmember'])
parser.add_argument('-voice_label', type=str, default='train', choices=['train', 'nontrain'])

parser.add_argument('-num', type=int, default=None)

parser.add_argument('-voice_chunk_splitting', action='store_true', default=False)
parser.add_argument('-partial_utterance_n_frames', type=int, default=160) # voice_chunk_splitting factor, recommended: 320

parser.add_argument('-sim', type=str, default='cosine', choices=['cosine', 'L2'])
parser.add_argument('-seed', type=int, default=555)

subparser = parser.add_subparsers(dest='system_type')

ge2e_parser = subparser.add_parser("LSTM_GE2E")

args = parser.parse_args()

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = _device

model_dir = 'saved_models'
if args.model_step is None:
    args.model_step = 315000 if args.dataset == 'vox2' else 255000
if args.system_type == 'LSTM_GE2E':
    enc_model_fpath = Path("{}/{}-{}/encoder_{}.bak".format(model_dir, args.dataset, args.model, args.model_step))
    print(enc_model_fpath)
    encoder = SV2TTS(enc_model_fpath, device=_device, partial_utterance_n_frames=args.partial_utterance_n_frames)

np.random.seed(args.seed)

spk_root = '{}/{}-wav-split/{}-{}/{}-{}-{}'.format(args.data_dir, args.dataset, args.model, args.spk_label, args.model, args.spk_label, args.voice_label)
print(spk_root)

def get_emb(in_fpaths, return_all=False, batch_size=None):

    with torch.no_grad():

        in_fpaths = in_fpaths
        load_wavs = [0] * len(in_fpaths)
        
        def _worker(start, end):
            for idx in range(start, end):
                in_fpath_wok = in_fpaths[idx]
                wav_wok, _ = torchaudio.load(in_fpath_wok)
                load_wavs[idx] = wav_wok
        
        # multiple thread load wav
        n_audios = len(in_fpaths)
        n_jobs = min(5, n_audios)
        n_jobs = n_jobs if n_jobs <= n_audios else n_audios
        n_audios_per_job = n_audios // n_jobs
        process_index = []
        for ii in range(n_jobs):
            process_index.append([ii*n_audios_per_job, (ii+1)*n_audios_per_job])
        if n_jobs * n_audios_per_job != n_audios:
            process_index[-1][-1] = n_audios
        futures = set()
        with ThreadPoolExecutor() as executor:
            for job_id in range(n_jobs):
                future = executor.submit(_worker, process_index[job_id][0], process_index[job_id][1])
                futures.add(future)
            for future in as_completed(futures):
                pass

        n_samples = len(in_fpaths)
        if batch_size is None:
            batch_size = n_samples
        batch_size = n_samples if n_samples < batch_size else batch_size
        n_run = math.ceil(n_samples / batch_size)
        all_embs = None
        for kk in range(n_run):
            wavs = []
            durations = []
            for wav in load_wavs[kk*batch_size:(kk+1)*batch_size]:
                wavs.append(wav)
                durations.append(wav.shape[-1])
            target_len = int(np.median(durations))
            for idx, wav in enumerate(wavs):
                if wav.shape[-1] > target_len:
                    start_pos = np.random.choice(wav.shape[-1] - target_len+1)
                    wavs[idx] = wav[..., start_pos:start_pos + target_len]
                elif wav.shape[-1] < target_len:
                    pad_zero = torch.zeros((1, target_len - wav.shape[-1]))
                    wavs[idx] = torch.cat((wav, pad_zero), -1)
            wavs = torch.stack(wavs).to(_device)
            return_partial = args.voice_chunk_splitting

            emb = encoder.embedding(wavs, return_partial=return_partial)

            if return_partial:
                emb = emb.view(-1, emb.shape[-1])
            if all_embs is None:
                all_embs = emb
            else:
                all_embs = torch.cat((all_embs, emb), dim=0)

        embed = torch.mean(all_embs, dim=0, keepdim=True) # (1, D)
        if 'LSTM_GE2E' == args.system_type:
            embed /= torch.linalg.norm(embed, ord=2) # (1, D)

        if return_all:
            return embed, all_embs
        else:
            return embed

batch_size = 1

flag = '{}-{}'.format(args.spk_label, args.voice_label)
file = 'MI_score_compact-{}-{}-{}-{}.txt'.format(args.dataset, args.model, flag, args.sim)

if args.num is not None:
    file = file[:-4] + '-num={}'.format(args.num) + '.txt'

if args.voice_chunk_splitting:
    file = file[:-4] + '-aug-part' + '.txt'
    assert batch_size == 1

if args.partial_utterance_n_frames != 160:
    file = file[:-4] + '-p={}.txt'.format(args.partial_utterance_n_frames)

os.makedirs('FE_outputs', exist_ok=True)
file = os.path.join('FE_outputs', file)
print(file)

if os.path.exists(file):
    r = open(file, 'r')
    len_existing_lines = len(r.readlines())
    r.close()
    print('len_existing_lines:', len_existing_lines)
else:
    len_existing_lines = -np.infty
wr = open(file, 'a')

root = spk_root
for spk_idx, spk_id in enumerate(sorted(os.listdir(root))):
    
    in_fpaths = []
    spk_dir = os.path.join(root, spk_id)
    for chap in sorted(os.listdir(spk_dir)):
        chap_dir = os.path.join(spk_dir, chap)
        for name in sorted(os.listdir(chap_dir)):
            if name[-4:] != '.wav' and name[-5:] != '.flac' and name[-4:] != '.m4a':
                continue
            path = os.path.join(chap_dir, name)
            in_fpaths.append(path)
    
    real_num = args.num
    if real_num is not None and len(in_fpaths) > real_num:
        in_fpaths = np.random.choice(in_fpaths, real_num, replace=False).tolist()

    mean_emb, all_embs = get_emb(in_fpaths, return_all=True, batch_size=batch_size) # (1, emb), (n_utt, emb)
    if all_embs.shape[0] <= 1:
        continue

    if args.sim == 'cosine':
        sim = torch.nn.functional.cosine_similarity(all_embs.unsqueeze(-1), mean_emb.unsqueeze(0).transpose(1, 2), dim=1).squeeze(-1).detach().cpu().numpy() # (n_utt, )
    else:
        sim = -1. * (((all_embs.unsqueeze(-1) - mean_emb.unsqueeze(0).transpose(1, 2)) ** 2).sum(1)).sqrt().squeeze(-1).detach().cpu().numpy()
    
    sim_pair_all = None
    bs = all_embs.shape[0] // 2
    n_run = math.ceil(all_embs.shape[0] / bs)
    for run_id in range(n_run):
        if args.sim == 'cosine':
            sim_pair_all_ = torch.nn.functional.cosine_similarity(all_embs[run_id*bs:(run_id+1)*bs, :].unsqueeze(-1), all_embs.unsqueeze(0).transpose(1, 2), dim=1).detach().cpu().numpy() # (n_utt, n_utt)
        else:
            sim_pair_all_ = -1. * (((all_embs[run_id*bs:(run_id+1)*bs, :].unsqueeze(-1) - all_embs.unsqueeze(0).transpose(1, 2)) ** 2).sum(1)).sqrt().detach().cpu().numpy()

        if sim_pair_all is None:
            sim_pair_all = sim_pair_all_
        else:
            sim_pair_all = np.concatenate((sim_pair_all, sim_pair_all_), axis=0)
    
    sim_pair = []
    for i in range(all_embs.shape[0]):
        for j in range(i+1, all_embs.shape[0]):
            sim_pair.append(sim_pair_all[i][j])
    
    sim_pair_avg = []
    sim_pair_std = []
    sim_pair_max = []
    sim_pair_min = []
    for i in range(all_embs.shape[0]):
        other_results = np.delete(sim_pair_all[i], i)
        sim_pair_avg_ = np.mean(other_results)
        sim_pair_std_ = np.std(other_results)
        sim_pair_max_ = np.max(other_results)
        sim_pair_min_ = np.min(other_results)
        sim_pair_avg.append(sim_pair_avg_)
        sim_pair_std.append(sim_pair_std_)
        sim_pair_max.append(sim_pair_max_)
        sim_pair_min.append(sim_pair_min_)
    
    line = '{} {} {}'.format(spk_id, flag, all_embs.shape[0])
    for idx, stat in enumerate([sim, sim_pair, sim_pair_avg, sim_pair_std, sim_pair_max, sim_pair_min]):
        x = np.mean(stat)
        y = np.std(stat)
        z = np.max(stat)
        h = np.min(stat)
        line = '{} {} {} {} {}'.format(line, x, y, z, h)
    print(spk_idx, line)
    line = line + '\n'
    wr.write(line)

wr.close()