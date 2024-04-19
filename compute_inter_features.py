
import math
from pathlib import Path
import numpy as np
import os
import torch
import torchaudio
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
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
parser.add_argument('-imposter_num', type=int, default=None)
parser.add_argument('-imposter_voice', type=int, default=None)
parser.add_argument('-imposter_voice_max', type=int, default=None)

parser.add_argument('-voice_chunk_splitting', action='store_true', default=False)
parser.add_argument('-partial_utterance_n_frames', type=int, default=160) # voice_chunk_splitting factor, recommended: 320
parser.add_argument('-imposter_VCS', action='store_true', default=False)

parser.add_argument('-sim', type=str, default='cosine', choices=['cosine', 'L2'])
parser.add_argument('-seed', type=int, default=555)

subparser = parser.add_subparsers(dest='system_type')

ge2e_parser = subparser.add_parser("LSTM_GE2E")

args = parser.parse_args()

np.random.seed(args.seed)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = _device

batch_size = 1

model_dir = 'saved_models'
if args.model_step is None:
    args.model_step = 315000 if args.dataset == 'vox2' else 255000
if args.system_type == 'LSTM_GE2E':
    enc_model_fpath = Path("{}/{}-{}/encoder_{}.bak".format(model_dir, args.dataset, args.model, args.model_step))
    print(enc_model_fpath)
    encoder = SV2TTS(enc_model_fpath, device=_device, partial_utterance_n_frames=args.partial_utterance_n_frames)

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


spk_root = '{}/{}-wav-split/{}-{}/{}-{}-{}'.format(args.data_dir, args.dataset, args.model, args.spk_label, args.model, args.spk_label, args.voice_label)
print(spk_root)

imposter_dir_root = f'{args.data_dir}/{args.dataset}-wav-split/imposter'
dump_path_im_emb_list = '{}/{}-wav-split/im_embs-{}-{}-{}{}.list'.format(args.data_dir, args.dataset, args.system_type, args.model_step, args.model, '' if not args.imposter_VCS else f'-VCS={args.partial_utterance_n_frames}')
dump_path_im_emb_dict = '{}/{}-wav-split/im_embs-{}-{}-{}{}.dict'.format(args.data_dir, args.dataset, args.system_type, args.model_step, args.model, '' if not args.imposter_VCS else f'-VCS={args.partial_utterance_n_frames}')
dump_path_im_emb_mean_list = '{}/{}-wav-split/im_embs_mean-{}-{}-{}{}.list'.format(args.data_dir, args.dataset, args.system_type, args.model_step, args.model, '' if not args.imposter_VCS else f'-VCS={args.partial_utterance_n_frames}')
dump_path_im_emb_mean_dict = '{}/{}-wav-split/im_embs_mean-{}-{}-{}{}.dict'.format(args.data_dir, args.dataset, args.system_type, args.model_step, args.model, '' if not args.imposter_VCS else f'-VCS={args.partial_utterance_n_frames}')
print(dump_path_im_emb_list, dump_path_im_emb_dict, dump_path_im_emb_mean_list, dump_path_im_emb_mean_dict)

def get_imposters():

    imposter_spk_root = f'{args.data_dir}/{args.dataset}-wav-split/imposter'
    print('compute imposter embeddings start', len(os.listdir(imposter_spk_root)))
    imposter_embs = None
    imposter_embs_mean = None
    im_spk_2_emb = dict()
    im_spk_2_emb_mean = dict()
    utt_index = 0
    for idx, spk_id in enumerate(sorted(os.listdir(imposter_spk_root))):
        spk_dir = os.path.join(imposter_spk_root, spk_id)
        in_fpaths = []
        for chap in sorted(os.listdir(spk_dir)):
            chap_dir = os.path.join(spk_dir, chap)
            for name in sorted(os.listdir(chap_dir)):
                if name[-4:] != '.wav' and name[-5:] != '.flac' and name[-4:] != '.m4a':
                    continue
                path = os.path.join(chap_dir, name)
                in_fpaths.append(path)

        mean_emb, all_embs = get_emb(in_fpaths, return_all=True, batch_size=batch_size) # (1, emb), (n_utt, emb)
        if imposter_embs_mean is None:
            imposter_embs_mean = mean_emb
            imposter_embs = all_embs
        else:
            imposter_embs_mean = torch.cat((imposter_embs_mean, mean_emb), dim=0)
            imposter_embs = torch.cat((imposter_embs, all_embs), dim=0)
        
        im_spk_2_emb[spk_id] = list(range(utt_index, utt_index+all_embs.shape[0]))
        im_spk_2_emb_mean[spk_id] = idx
        utt_index = utt_index + all_embs.shape[0]
    
    torch.save(imposter_embs_mean, dump_path_im_emb_mean_list)
    torch.save(imposter_embs, dump_path_im_emb_list)
    torch.save(im_spk_2_emb_mean, dump_path_im_emb_mean_dict)
    torch.save(im_spk_2_emb, dump_path_im_emb_dict)

    print('compute imposter embeddings end', len(os.listdir(imposter_spk_root)))
    return imposter_embs, im_spk_2_emb, imposter_embs_mean, im_spk_2_emb_mean

if os.path.exists(dump_path_im_emb_list) and os.path.exists(dump_path_im_emb_dict) and os.path.exists(dump_path_im_emb_mean_list) and os.path.exists(dump_path_im_emb_mean_dict)
    imposter_embs_ori = torch.load(dump_path_im_emb_list)
    im_spk_2_emb_ori = torch.load(dump_path_im_emb_dict)
    imposter_embs_mean_ori = torch.load(dump_path_im_emb_mean_list)
    im_spk_2_emb_mean_ori = torch.load(dump_path_im_emb_mean_dict)
else:
    imposter_embs_ori, im_spk_2_emb_ori, imposter_embs_mean_ori, im_spk_2_emb_mean_ori = get_imposters()

my_imposter_voice = args.imposter_voice_max if args.imposter_voice_max is not None else args.imposter_voice
print(my_imposter_voice)
if my_imposter_voice is None:
    src_spk_ids = list(im_spk_2_emb_mean_ori.keys())
else:
    src_spk_ids = []
    for spk_id, utt_idx in im_spk_2_emb_ori.items():
        if len(utt_idx) >= my_imposter_voice:
            src_spk_ids.append(spk_id)
select_spk_num = args.imposter_num if args.imposter_num is not None else len(src_spk_ids)
if select_spk_num > len(src_spk_ids):
    assert "{} spks have {} voices, but you require {} spks".format(len(src_spk_ids), args.imposter_voice_max if args.imposter_voice_max is not None else args.imposter_voice, select_spk_num)
print('Select imposters:', len(list(im_spk_2_emb_mean_ori.keys())), args.imposter_voice, len(src_spk_ids), select_spk_num)
imposter_ids = np.random.choice(src_spk_ids, select_spk_num, replace=False)
print(imposter_ids)
imposter_embs_mean = []
imposter_embs = []
im_spk_2_emb_mean = {}
im_spk_2_emb = {}
for ppp, i_id in enumerate(imposter_ids):
    spk_idx = im_spk_2_emb_mean_ori[i_id]
    imposter_embs_mean.append(imposter_embs_mean_ori[spk_idx])
    utt_idx = im_spk_2_emb_ori[i_id]
    start_cnt = len(imposter_embs)
    for u_i in utt_idx:
        imposter_embs.append(imposter_embs_ori[u_i])
    im_spk_2_emb_mean[i_id] = ppp
    end_cnt = len(imposter_embs)
    im_spk_2_emb[i_id] = list(range(start_cnt, end_cnt))
imposter_embs_mean = torch.stack(imposter_embs_mean)
imposter_embs = torch.stack(imposter_embs)

print('Select imposter', imposter_embs_mean_ori.shape, imposter_embs_mean.shape)
imposter_embs_mean_ori = imposter_embs_mean
imposter_embs_ori = imposter_embs
im_spk_2_emb_mean_ori = im_spk_2_emb_mean
im_spk_2_emb_ori = im_spk_2_emb

if args.imposter_voice is not None:
    imposter_embs_ori_2 = []
    im_spk_2_emb_ori_2 = {}
    imposter_embs_mean_ori_2 = []
    for spk_id_hhh in im_spk_2_emb_mean_ori.keys():
        utt_idx = im_spk_2_emb_ori[spk_id_hhh]
        select_utt_idx_all = np.random.choice(utt_idx, args.imposter_voice) if len(utt_idx) > args.imposter_voice else utt_idx
        select_utt_idx = []
        for i_j_k in range(args.imposter_voice):
            select_utt_idx.append(select_utt_idx_all[i_j_k])
        start_cnt = len(imposter_embs_ori_2)
        x = None
        for u_i in select_utt_idx:
            imposter_embs_ori_2.append(imposter_embs_ori[u_i])
            if x is None:
                x = imposter_embs_ori[u_i]
            else:
                x += imposter_embs_ori[u_i]
        x /= len(select_utt_idx)
        if 'LSTM_GE2E' in args.system_type or args.norm:
            x /= torch.linalg.norm(x, ord=2) # (1, D)
        imposter_embs_mean_ori_2.append(x)
        end_cnt = len(imposter_embs_ori_2)
        im_spk_2_emb_ori_2[spk_id_hhh] = list(range(start_cnt, end_cnt))
    imposter_embs_ori_2 = torch.stack(imposter_embs_ori_2)
    imposter_embs_mean_ori_2 = torch.stack(imposter_embs_mean_ori_2)
    print('Select imposter voice', args.imposter_voice, imposter_embs_ori.shape, imposter_embs_ori_2.shape)
    imposter_embs_ori = imposter_embs_ori_2
    im_spk_2_emb_ori = im_spk_2_emb_ori_2
    imposter_embs_mean_ori = imposter_embs_mean_ori_2
else:
    pass

imposter_embs_mean = imposter_embs_mean_ori
imposter_embs = imposter_embs_ori
im_spk_2_emb_mean = im_spk_2_emb_mean_ori
im_spk_2_emb = im_spk_2_emb_ori

flag = '{}-{}'.format(args.spk_label, args.voice_label)
file = 'MI_score_far-{}-{}-{}-{}.txt'.format(args.dataset, args.model, flag, args.sim)

if args.num is not None:
    file = file[:-4] + '-num={}'.format(args.num) + '.txt'

if args.voice_chunk_splitting:
    file = file[:-4] + '-aug-part' + '.txt'
    assert batch_size == 1

if args.partial_utterance_n_frames != 160:
    file = file[:-4] + '-p={}.txt'.format(args.partial_utterance_n_frames)
    if args.imposter_VCS:
        file = file[:-4] + '-imposter-part.txt'
    
if args.imposter_num is not None:
    file = file[:-4] + 'im-num={}.txt'.format(args.imposter_num)

if args.imposter_voice is not None:
    file = file[:-4] + '-im-vnum={}.txt'.format(args.imposter_voice)

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
for spk_idx_2, spk_id in enumerate(sorted(os.listdir(root))):

    if spk_idx_2 < len_existing_lines:
        continue

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

    if args.sim == 'cosine':
        m_m = 1. - torch.nn.functional.cosine_similarity(imposter_embs_mean.unsqueeze(-1), mean_emb.unsqueeze(0).transpose(1, 2), dim=1).squeeze(-1) # (n_imposter, )
    else:
        m_m = ((imposter_embs_mean.unsqueeze(-1) - mean_emb.unsqueeze(0).transpose(1, 2)) ** 2).sum(1).sqrt().squeeze(-1) # (n_imposter, )
    
    if args.sim == 'cosine':
        m_i = 1. - torch.nn.functional.cosine_similarity(imposter_embs.unsqueeze(-1), mean_emb.unsqueeze(0).transpose(1, 2), dim=1).squeeze(-1) # (n_imposter_utt, )
    else:
        m_i = ((imposter_embs.unsqueeze(-1) - mean_emb.unsqueeze(0).transpose(1, 2)) **2).sum(1).sqrt().squeeze(-1) # (n_imposter_utt, )
    
    n_spk = len(im_spk_2_emb.keys())
    m_i_avg = torch.zeros(n_spk, device=_device)
    m_i_std = torch.zeros(n_spk, device=_device)
    m_i_max = torch.zeros(n_spk, device=_device)
    m_i_min = torch.zeros(n_spk, device=_device)
    for spk_idx, spk in enumerate(im_spk_2_emb.keys()):
        utt_index = im_spk_2_emb[spk]
        m_i_avg[spk_idx] = torch.mean(m_i[utt_index]).item()
        m_i_std[spk_idx] = torch.std(m_i[utt_index]).item()
        m_i_max[spk_idx] = torch.max(m_i[utt_index]).item()
        m_i_min[spk_idx] = torch.min(m_i[utt_index]).item()

    if args.sim == 'cosine':
        i_m = 1. - torch.nn.functional.cosine_similarity(imposter_embs_mean.unsqueeze(-1), all_embs.unsqueeze(0).transpose(1, 2), dim=1) # (n_imposter, n_utt)
    else:
        i_m = ((imposter_embs_mean.unsqueeze(-1) - all_embs.unsqueeze(0).transpose(1, 2)) ** 2).sum(1).sqrt() # (n_imposter, n_utt)
    i_m_1_avg = torch.mean(i_m, dim=0)
    i_m_1_std = torch.std(i_m, dim=0) if i_m.shape[0] > 1 else torch.std(i_m, dim=0, unbiased=False)
    i_m_1_max = torch.max(i_m, dim=0)[0]
    i_m_1_min = torch.min(i_m, dim=0)[0]
    i_m_2_avg = torch.mean(i_m, dim=1)
    i_m_2_std = torch.std(i_m, dim=1) if i_m.shape[1] > 1 else torch.std(i_m, dim=1, unbiased=False)
    i_m_2_max = torch.max(i_m, dim=1)[0]
    i_m_2_min = torch.min(i_m, dim=1)[0]

    i_i = None
    batch_size_2 = 1000 if args.voice_chunk_splitting else 5000
    batch_size_2 = batch_size_2 if imposter_embs.shape[0] >= batch_size_2 else imposter_embs.shape[0]
    n_batch = math.ceil(imposter_embs.shape[0] / batch_size_2)
    for b_i in range(n_batch):
        if args.sim == 'cosine':
            i_i_batch = 1. - torch.nn.functional.cosine_similarity(imposter_embs[b_i*batch_size_2:(b_i+1)*batch_size_2].unsqueeze(-1), all_embs.unsqueeze(0).transpose(1, 2), dim=1) # (n_imposter_utt, n_utt)
        else:
            i_i_batch = ((imposter_embs[b_i*batch_size_2:(b_i+1)*batch_size_2].unsqueeze(-1) - all_embs.unsqueeze(0).transpose(1, 2)) ** 2).sum(1).sqrt() # (n_imposter_utt, n_utt)
        if i_i is None:
            i_i = i_i_batch
        else:
            i_i = torch.cat((i_i, i_i_batch), dim=0)
    i_i_1_avg = torch.mean(i_i, dim=0)
    i_i_1_std = torch.std(i_i, dim=0) if i_i.shape[0] > 1 else torch.std(i_i, dim=0, unbiased=False)
    i_i_1_max = torch.max(i_i, dim=0)[0]
    i_i_1_min = torch.min(i_i, dim=0)[0]

    i_i_2_avg = torch.mean(i_i, dim=1)
    i_i_2_std = torch.std(i_i, dim=1) if i_i.shape[1] > 1 else torch.std(i_i, dim=1, unbiased=False)
    i_i_2_max = torch.max(i_i, dim=1)[0]
    i_i_2_min = torch.min(i_i, dim=1)[0]

    line = '{} {} {}'.format(spk_id, flag, len(in_fpaths) if not args.voice_chunk_splitting else all_embs.shape[0])
    for m_idx, metric in enumerate([m_m, m_i, m_i_avg, m_i_std, m_i_max, m_i_min, i_m, i_m_1_avg, i_m_1_std, i_m_1_max, i_m_1_min, i_m_2_avg, i_m_2_std, i_m_2_max, i_m_2_min, 
    i_i, i_i_1_avg, i_i_1_std, i_i_1_max, i_i_1_min, i_i_2_avg, i_i_2_std, i_i_2_max, i_i_2_min]):
        line = '{} {} {} {} {}'.format(line, 
        torch.mean(metric).item(), torch.std(metric).item() if metric.shape[0] > 1 else torch.std(metric, unbiased=False).item(), torch.max(metric).item(), torch.min(metric).item()) 
    print(spk_idx_2, line)
    line = line + '\n'
    wr.write(line)

wr.close()