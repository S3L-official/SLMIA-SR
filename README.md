# SLMIA-SR
This repository contains the code for the paper ***"SLMIA-SR: Speaker-Level Membership Inference Attacks against Speaker Recognition Systems"***, 
which has been published at the 31st Network and Distributed System Security Symposium (NDSS 2024).

[[paper]](https://www.ndss-symposium.org/ndss-paper/slmia-sr-speaker-level-membership-inference-attacks-against-speaker-recognition-systems/)

## Environment setup
```
conda create -n SLMIA-SR python=3.7.11
conda activate SLMIA-SR
sh install_requirements.sh
```

Install `ffmpeg` according to the instructions in [ffmpeg install instructions](instructions_ffmpeg.md) 

## Dataset preparation
Download VoxCeleb-2 and LibriSpeech from [VoxCeleb-2 download link](https://mm.kaist.ac.kr/datasets/voxceleb/) and [LibriSpeech download link](https://www.openslr.org/12). 
You should have the following dataset structure:

- $data_dir/vox2/dev/aac/spk_dir/chap_dir/*.m4a
- $data_dir/vox2/test/aac/spk_dir/chap_dir/*.m4a
- $data_dir/ls/train-clean-100/spk_dir/chap_dir/*.flac
- $data_dir/ls/train-clean-360/spk_dir/chap_dir/*.flac
- $data_dir/ls/train-other-500/spk_dir/chap_dir/*.flac
- $data_dir/ls/test-clean/spk_dir/chap_dir/*.flac
- $data_dir/ls/test-other/spk_dir/chap_dir/*.flac
- $data_dir/ls/dev-clean/spk_dir/chap_dir/*.flac
- $data_dir/ls/dev-other/spk_dir/chap_dir/*.flac

Run `python -W ignore prepare_dataset.py --data_dir $data_dir --dataset vox2 --n_jobs 20` to convert the audio file into the pytorch tensor.

Run `python split_dataset.py --data_dir $data_dir --dataset vox2` to split the dataset as specified by the Table-III in the paper. 

## SRS training
Run `sh train_SRS.sh $data_dir vox2` to train both the target and shadow SRSs on the dataset VoxCeleb-2.

## Extract membership inference features
#### Setting-1 (TABLE V in the paper)
Run `python extract_feature.py -dataset vox2 -data_dir $data_dir LSTM_GE2E`
#### Setting-2 (TABLE VI in the paper)
Run `python extract_feature.py -dataset vox2 -data_dir $data_dir -num 10 -imposter_num 20 -imposter_voice_num 10 -voice_chunk_splitting -partial_utterance_n_frames 320 -imposter_VCS LSTM_GE2E`

## Train and evaluate the attack model
#### Setting-1 (TABLE V in the paper)
Run `python train_evaluate_attack_model.py -system_type LSTM_GE2E -dataset vox2 -fpr 0.1`
#### Setting-2 (TABLE VI in the paper)
Run `python train_evaluate_attack_model.py -system_type LSTM_GE2E -dataset vox2 -num 10 -imposter_num 20 -imposter_voice 10 -voice_chunk_splitting -imposter_VCS -partial_utterance_n_frames 320 -fpr 0.1`

Feel free to change the dataset to `ls` and fpr to 0.2 since this dataset has less than 1,000 speakers in each splitted set.


If our work or code is useful for you, consider citing our paper as follows:
```
@inproceedings{SLMIA-SR,
  author       = {Guangke Chen and
                  Yedi Zhang and
                  Fu Song},
  title        = {SLMIA-SR: Speaker-Level Membership Inference Attacks against Speaker Recognition Systems},
  booktitle    = {Proceedings of the 31st Annual Network and Distributed System Security (NDSS) Symposium},
  year         = {2024},
}
```