from copy import deepcopy

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from basics.base_augmentation import BaseAugmentation, require_same_keys
from basics.base_pe import BasePE
from modules.fastspeech.param_adaptor import VARIANCE_CHECKLIST
from modules.fastspeech.tts_modules import LengthRegulator
from utils.binarizer_utils import SinusoidalSmoothingConv1d, get_mel2ph_torch
from utils.hparams import hparams
from utils.infer_utils import resample_align_curve


class VarianceStretchAugmentation(BaseAugmentation):
    def __init__(self, data_dirs: list, augmentation_args: dict, pe: BasePE = None):
        super().__init__(data_dirs, augmentation_args)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr = LengthRegulator().to(self.device)
        self.pe = pe
        self.midi_smooth = (
            SinusoidalSmoothingConv1d(
                round(hparams["midi_smooth_width"] / self.timestep)
            )
            .eval()
            .to(self.device)
        )

    @require_same_keys
    def process_item(
        self, item: dict, key_shift=0.0, speed=1.0, replace_spk_id=None
    ) -> dict:
        aug_item = deepcopy(item)

        # Pitch shifting (no waveform required)
        if key_shift != 0.0:
            aug_item["pitch"] += key_shift

            if replace_spk_id is not None:
                aug_item["spk_id"] = replace_spk_id
            else:
                aug_item["key_shift"] = key_shift

        # Time stretching (requires waveform)
        has_wav = (
            aug_item.get("wav_fn") is not None and aug_item.get("wav_fn") != "None"
        )

        if has_wav and speed != 1.0:
            timestep = self.timestep

            # new_length from ph_dur sum
            ph_dur_sum_frames = int(aug_item["ph_dur"].sum())
            new_length = max(1, round(ph_dur_sum_frames / speed))

            # effective speed quantization
            aug_item["speed"] = round(hparams["hop_size"] * speed) / hparams["hop_size"]
            aug_item["seconds"] /= aug_item["speed"]
            aug_item["length"] = new_length

            # waveform
            waveform, _ = librosa.load(
                aug_item["wav_fn"], sr=hparams["audio_sample_rate"], mono=True
            )

            # ph_dur: frames -> seconds -> scale -> back to frames
            ph_dur_sec_np = (
                aug_item["ph_dur"].astype(np.float32) * timestep / aug_item["speed"]
            )
            ph_dur_sec = torch.from_numpy(ph_dur_sec_np)
            ph_acc = torch.round(
                torch.cumsum(ph_dur_sec, dim=0) / timestep + 0.5
            ).long()
            new_ph_dur = torch.diff(ph_acc, dim=0, prepend=ph_acc.new_zeros(1))
            aug_item["ph_dur"] = new_ph_dur.cpu().numpy()

            # mel2ph (conditional)
            if "mel2ph" in aug_item:
                mel2ph = get_mel2ph_torch(
                    self.lr, ph_dur_sec, new_length, timestep, self.device
                )
                aug_item["mel2ph"] = mel2ph.cpu().numpy()

            # note_dur and mel2note (conditional on predict_pitch)
            if "note_dur" in aug_item:
                nd_sec_np = (
                    aug_item["note_dur"].astype(np.float32)
                    * timestep
                    / aug_item["speed"]
                )
                nd_sec = torch.from_numpy(nd_sec_np)

                if "mel2note" in aug_item:
                    mel2note = get_mel2ph_torch(
                        self.lr, nd_sec, new_length, timestep, self.device
                    )
                    aug_item["mel2note"] = mel2note.cpu().numpy()

                nd_acc = torch.round(
                    torch.cumsum(nd_sec, dim=0) / timestep + 0.5
                ).long()
                new_nd = torch.diff(nd_acc, dim=0, prepend=nd_acc.new_zeros(1))
                aug_item["note_dur"] = new_nd.cpu().numpy()

            # pitch and uv re-extraction (conditional)
            if "pitch" in aug_item or "uv" in aug_item:
                f0_hz, uv_mask = self.pe.get_pitch(
                    waveform,
                    samplerate=hparams["audio_sample_rate"],
                    length=new_length,
                    hop_size=hparams["hop_size"],
                    speed=speed,
                    f0_min=hparams["f0_min"],
                    f0_max=hparams["f0_max"],
                    interp_uv=True,
                )
                if "pitch" in aug_item:
                    aug_item["pitch"] = librosa.hz_to_midi(f0_hz.astype(np.float32))
                if "uv" in aug_item:
                    aug_item["uv"] = uv_mask

            # variance curve resampling
            for v_name in VARIANCE_CHECKLIST:
                if v_name in aug_item:
                    orig = aug_item[v_name]
                    aug_item[v_name] = resample_align_curve(
                        orig,
                        original_timestep=timestep,
                        target_timestep=timestep * speed,
                        align_length=new_length,
                    )

            # base_pitch recomputation (conditional on predict_pitch)
            if "note_midi" in aug_item and "mel2note" in aug_item:
                nm_t = torch.from_numpy(aug_item["note_midi"]).to(self.device)
                mn_t = torch.from_numpy(aug_item["mel2note"]).to(self.device)
                fmp = torch.gather(F.pad(nm_t, [1, 0], value=0.0), 0, mn_t)
                aug_item["base_pitch"] = (
                    self.midi_smooth(fmp[None])[0].detach().cpu().numpy()
                )

            # midi recomputation (conditional on predict_dur)
            if "midi" in aug_item:
                if "mel2ph" in aug_item:
                    m2ph = torch.from_numpy(aug_item["mel2ph"])
                else:
                    pd_local = aug_item["ph_dur"].astype(np.float32) * timestep
                    m2ph = get_mel2ph_torch(
                        self.lr, pd_local, new_length, timestep, self.device
                    )

                if "pitch" in aug_item:
                    p_t = torch.from_numpy(aug_item["pitch"])
                else:
                    f0_m, _ = self.pe.get_pitch(
                        waveform,
                        samplerate=hparams["audio_sample_rate"],
                        length=new_length,
                        hop_size=hparams["hop_size"],
                        speed=speed,
                        f0_min=hparams["f0_min"],
                        f0_max=hparams["f0_max"],
                        interp_uv=True,
                    )
                    p_t = torch.from_numpy(librosa.hz_to_midi(f0_m))

                pd_t = torch.from_numpy(aug_item["ph_dur"])
                T_ph = len(aug_item["tokens"])
                mel2dur = torch.gather(F.pad(pd_t, [1, 0], value=1), 0, m2ph)
                pm = p_t.new_zeros(T_ph + 1).scatter_add(0, m2ph, p_t / mel2dur)[1:]
                aug_item["midi"] = pm.round().long().clamp(0, 127).cpu().numpy()

        return aug_item
