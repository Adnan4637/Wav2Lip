import librosa
import librosa.filters
import numpy as np
from scipy.io import wavfile
from scipy import signal
from hparams import hparams as hp

# ----------------- Audio Utilities -----------------

def load_wav(path, sr):
    return librosa.load(path, sr=sr)[0]

def save_wav(wav, path, sr):
    wav = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wavfile.write(path, sr, wav.astype(np.int16))

def preemphasis(wav, k, preemphasize=True):
    return signal.lfilter([1, -k], [1], wav) if preemphasize else wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    return signal.lfilter([1], [1, -k], wav) if inv_preemphasize else wav

def get_hop_size():
    hop_size = hp.hop_size
    if hop_size is None:
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    return hop_size

def _stft(y):
    return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=get_hop_size(), win_length=hp.win_size)

def linearspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db
    return _normalize(S) if hp.signal_normalization else S

def melspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db
    return _normalize(S) if hp.signal_normalization else S

# ----------------- Mel Conversion -----------------

def _linear_to_mel(spectrogram):
    mel_basis = _build_mel_basis()
    return np.dot(mel_basis, spectrogram)

def _build_mel_basis():
    return librosa.filters.mel(
        sr=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax
    )

# ----------------- DB/Normalization -----------------

def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _normalize(S):
    if hp.symmetric_mels:
        return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                       -hp.max_abs_value, hp.max_abs_value)
    else:
        return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)
