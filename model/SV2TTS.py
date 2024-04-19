
import numpy as np
import torch.nn as nn
import torch

# from model._SV2TTS.encoder import inference as encoder
from model._SV2TTS.encoder.inference_class import inference_class
from model._SV2TTS.encoder import audio, inference
from model._SV2TTS.encoder.audio import normalize_volume, trim_long_silences
from model._SV2TTS.encoder.params_data import *
from model.utils import check_input_range, parse_enroll_model_file, parse_mean_file_2
from warnings import warn

try:
    import webrtcvad
except:
    # warn("Unable to import 'webrtcvad'. This package enables noise removal and is recommended.") 
    webrtcvad=None

class SV2TTS(nn.Module):
# class SV2TTS(audionet_csine):

    def __init__(self, extractor_file, model_file=None, threshold=None, device="cpu", mean_file=None, backward=False, partial_utterance_n_frames=160, encoder_type='A1') -> None:
        # super().__init__()
        nn.Module.__init__(self)

        self.device = device

        self.encoder = inference_class()
        # print(device)
        self.encoder.load_model(extractor_file, device=device, backward=backward, encoder_type=encoder_type)


        if model_file is not None:
            self.num_spks, self.spk_ids, self.z_norm_means, self.z_norm_stds, self.enroll_embs = \
                parse_enroll_model_file(model_file, self.device)
        
        if mean_file is not None:
            self.emb_mean = parse_mean_file_2(mean_file, device)
        else:
            self.emb_mean = 0.

        # If you need SV or OSI, must input threshold
        self.threshold = threshold if threshold else -np.infty # Valid for SV and OSI tasks; CSI: -infty

        self.allowed_flags = sorted([
            0, 1
        ]) # 0: wav; 1: raw feat; 2: delta feat; 3: cmvn feat
        self.range_type = 'scale'

        self.partial_utterance_n_frames = partial_utterance_n_frames

    
    def compute_feat(self, x, flag=1, using_partial=False, return_full=False):
        """
        x: wav with shape [B, 1, T]
        flag: the flag indicating to compute what type of features (1: raw feat)
        return: feats with shape [B, T, F] (T: #Frames, F: #feature_dim)
        """
        assert flag in [f for f in self.allowed_flags if f != 0]
        x = check_input_range(x, range_type=self.range_type)

        feats = self.raw(x, using_partial=using_partial, return_full=return_full) # (B, T, F)
        if flag == 1: # calulate ori feat
            return feats
        else: # will not go to this branch
            pass

    
    def embedding(self, x, flag=0, return_partial=False, not_return_partial_using_partial=True):
        """
        x: wav or acoustic features (raw/delta/cmvn)
        flag: indicating the type of x (0: wav; 1: raw feat)
        """
        assert flag in self.allowed_flags
        if flag == 0:
            # x = check_input_range(x, range_type=self.range_type) # no need, since compute_feat will check
            # feats = self.compute_feat(x, flag=self.allowed_flags[-1])
            if return_partial:
                using_partial = True
            else:
                using_partial = not_return_partial_using_partial
            # feats = self.compute_feat(x, flag=self.allowed_flags[-1], using_partial=True)
            feats = self.compute_feat(x, flag=self.allowed_flags[-1], using_partial=using_partial)
        elif flag == 1:
            feats = x
        else: # will not go to this branch
            pass
        if len(feats.shape) == 3:
            feats = feats.unsqueeze(1)
        # print('############## Nan Check Feat:', np.any(np.isnan(feats.detach().cpu().numpy())))
        emb = self.extract_emb(feats, return_partial=return_partial)
        if np.any(np.isnan(emb.detach().cpu().numpy())):
            print('############## Nan Check emb:', np.any(np.isnan(emb.detach().cpu().numpy())), x.shape)
        # else:
        #      print('############## Nan Check emb:', np.any(np.isnan(emb.detach().cpu().numpy())), x.shape)
        # return emb - self.emb_mean # [B, D]
        return emb # already subtract emb mean in self.extract_emb(feats)
    

    def raw(self, x, using_partial=False, return_full=False):
        """
        x: (B, 1, T)
        """
        x = x.squeeze(1)
        x = normalize_volume(x, audio_norm_target_dBFS, increase_only=True)
        # if x.shape[0] == 0 and webrtcvad:
        #     x_vad = None
        #     for x_ in x:
        #         x_vad_ = trim_long_silences(x_)
        #         if x_vad is None:
        #             x_vad = x_vad_
        #         else:
        #             x_vad = torch.cat((x_vad, x_vad_), dim=0)
        if x.shape[0] == 0 and webrtcvad:
            x_vad = trim_long_silences(x.squeeze(0)).unsqueeze(0)
        else:
            x_vad = x
        if not using_partial:
            feat = audio.wav_to_mel_spectrogram_torch(x_vad) # (B, T, F)
            # print('Not Using Partial')
            return feat
        else:
            # wave_slices, mel_slices = self.encoder.compute_partial_slices(x_vad.shape[1])
            wave_slices, mel_slices = self.encoder.compute_partial_slices(x_vad.shape[1], partial_utterance_n_frames=self.partial_utterance_n_frames)
            max_wave_length = wave_slices[-1].stop
            if max_wave_length >= x_vad.shape[1]:
                # wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
                x_vad = torch.nn.functional.pad(x_vad, (0, max_wave_length - x_vad.shape[1]), mode="constant")
            frames = audio.wav_to_mel_spectrogram_torch(x_vad) # (B, T, F)
            frames_batch = None
            for s in mel_slices:
                f = frames[:, s, :].unsqueeze(1)
                if frames_batch is None:
                    frames_batch = f
                else:
                    frames_batch = torch.cat((frames_batch, f), dim=1)
            if not return_full:
                return frames_batch # (B, n, T1, F)
            else:
                return frames, mel_slices
    

    def extract_emb(self, x, return_partial=False):
        '''
        x: (B, n, T1, F)
        '''
        # print('EMB Shape:', x.shape)
        B, n, T1, F = x.shape
        frames_batch = x.view(-1, T1, F) # (B * n , T1, F)
        partial_embeds = self.encoder.embed_frames_batch(frames_batch) # (B * n, D)
        partial_embeds = partial_embeds.view(B, n, -1) # (B, n, D)
        if return_partial:
            return partial_embeds
        # Compute the utterance embedding from the partial embeddings
        # raw_embed = np.mean(partial_embeds, axis=0)
        raw_embed = torch.mean(partial_embeds, axis=1) # (B, D)
        # embed = raw_embed / np.linalg.norm(raw_embed, 2)
        embed = raw_embed / torch.norm(raw_embed, p=2, dim=1, keepdim=True)
        # return embed # (B, D)
        return embed - self.emb_mean # (B, D)
        # embed_sub = embed - self.emb_mean
        # return embed_sub / torch.norm(embed_sub, p=2, dim=1, keepdim=True) # (B, D)


    def forward(self, x, flag=0, return_emb=False, enroll_embs=None):
        """
        x: wav or acoustic features (raw/delta/cmvn)
        flag: indicating the type of x (0: wav; 1: raw feat; 2: delta feat; 3: cmvn feat)
        """
        embedding = self.embedding(x, flag=flag)
        # print('############## Nan Check:', np.any(np.isnan(embedding.detach().cpu().numpy())))
        
        if not hasattr(self, 'enroll_embs'):
            assert enroll_embs is not None
        enroll_embs = enroll_embs if enroll_embs is not None else self.enroll_embs
        # scores = self.scoring_trials(enroll_embs=enroll_embs, embs=embedding)
        scores = torch.nn.functional.cosine_similarity(embedding.unsqueeze(2), enroll_embs.transpose(0, 1).unsqueeze(0), dim=1)
        if not return_emb:
            return scores
        else:
            return scores, embedding

    
    def score(self, x, flag=0, enroll_embs=None):
        """
        x: wav or acoustic features (raw/delta/cmvn)
        flag: indicating the type of x (0: wav; 1: raw feat; 2: delta feat; 3: cmvn feat)
        """
        logits = self.forward(x, flag=flag, enroll_embs=enroll_embs)
        scores = logits
        return scores
    

    def make_decision(self, x, flag=0, enroll_embs=None):
        """
        x: wav or acoustic features (raw/delta/cmvn)
        flag: indicating the type of x (0: wav; 1: raw feat; 2: delta feat; 3: cmvn feat)
        """
        scores = self.score(x, flag=flag, enroll_embs=enroll_embs)

        decisions = torch.argmax(scores, dim=1)
        max_scores = torch.max(scores, dim=1)[0]
        decisions = torch.where(max_scores > self.threshold, decisions,
                        torch.tensor([-1] * decisions.shape[0], dtype=torch.int64, device=decisions.device)) # -1 means reject

        return decisions, scores
        



