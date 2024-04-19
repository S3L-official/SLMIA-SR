
import argparse
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default='vox', choices=['vox2', 'ls'])
parser.add_argument('-data_dir', type=str, default='data', help='data directory')

parser.add_argument('-num', type=int, default=None)
parser.add_argument('-imposter_num', type=int, default=None)
parser.add_argument('-imposter_voice_num', type=int, default=None)
parser.add_argument('-imposter_voice_num_max', type=int, default=None)

parser.add_argument('-voice_chunk_splitting', action='store_true', default=False)
parser.add_argument('-partial_utterance_n_frames', type=int, default=320) # voice_chunk_splitting factor, recommended: 320
parser.add_argument('-imposter_VCS', action='store_true', default=False)

parser.add_argument('-sim', type=str, default='cosine', choices=['cosine', 'L2'])

parser.add_argument('-seed', type=int, default=555)

parser.add_argument('-model_step', type=int, default=None)

subparser = parser.add_subparsers(dest='system_type')
ge2e_parser = subparser.add_parser("LSTM_GE2E")

args = parser.parse_args()

print('print:', args.system_type)

if args.model_step is None:
    args.model_step = 315000 if args.dataset == 'vox2' else 255000


for model in ['target', 'shadow']:
    for spk_label in ['member', 'nonmember']:
        for voice_label in ['train', 'nontrain']:
            if spk_label == 'nonmember' and voice_label == 'train':
                continue

            ## intra features
            command = f'python compute_intra_features.py -data_dir {args.data_dir} -dataset {args.dataset} -model {model} -spk_label {spk_label} -voice_label {voice_label}'
            if args.num is not None:
                command = f'{command} -num {args.num}'
            if args.voice_chunk_splitting:
                command = f'{command} -voice_chunk_splitting'
            command = f'{command} -partial_utterance_n_frames {args.partial_utterance_n_frames} -sim {args.sim} -seed {args.seed} -model_step {args.model_step} {args.system_type}'
            print(command)
            os.system(command)


            ## inter-features
            command = f'python compute_inter_features.py -data_dir {args.data_dir} -dataset {args.dataset} -model {model} -spk_label {spk_label} -voice_label {voice_label}'
            if args.imposter_num is not None:
                command = f'{command} -imposter_num {args.imposter_num}'
            if args.imposter_voice_num is not None:
                command = f'{command} -imposter_voice_num {args.imposter_voice_num}'
            if args.imposter_voice_num_max is not None:
                command = f'{command} -imposter_voice_num_max {args.imposter_voice_num_max}'
            if args.voice_chunk_splitting:
                command = f'{command} -voice_chunk_splitting'
            if args.imposter_VCS:
                command = f'{command} -imposter_VCS'
            command = f'{command} -partial_utterance_n_frames {args.partial_utterance_n_frames} -sim {args.sim} -seed {args.seed} -model_step {args.model_step} {args.system_type}'
            print(command)
            os.system(command)