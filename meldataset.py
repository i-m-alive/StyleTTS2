import os
import time
import random
import numpy as np
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import logging
from soundfile import SoundFile

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Constants and symbol sets
_PAD = "$"
_PUNCTUATION = ';:,.!?¡¿—…"«»“” '
_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_LETTERS_IPA = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
SYMBOLS = [_PAD] + list(_PUNCTUATION) + list(_LETTERS) + list(_LETTERS_IPA)

# Create a dictionary of symbols
SYMBOLS_DICT = {symbol: idx for idx, symbol in enumerate(SYMBOLS)}

# Text cleaning class
class TextCleaner:
    def __init__(self):
        self.word_index_dictionary = SYMBOLS_DICT

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                logger.warning(f"Unrecognized character: {char}")
        return indexes

# Preprocessing parameters
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
}

# Mel spectrogram transformation
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=MEL_PARAMS["n_mels"], n_fft=SPECT_PARAMS["n_fft"], win_length=SPECT_PARAMS["win_length"], hop_length=SPECT_PARAMS["hop_length"]
)
mean, std = -4, 4  # Normalization parameters

# Function to preprocess audio waveform into mel spectrogram
def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

# Function to validate audio files
def is_valid_audio(file_path):
    try:
        with SoundFile(file_path) as sf:
            return True
    except Exception as e:
        logger.error(f"Invalid audio file {file_path}: {e}")
        return False

# Dataset class for loading audio and text data
class FilePathDataset(Dataset):
    def __init__(self, data_list, root_path, sr=24000, data_augmentation=False, validation=False, OOD_data="/content/dataset.txt", min_length=50):
        # Initialize the dataset
        self.data_list = [line.strip().split('|') for line in data_list]  # Split by '|'
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in self.data_list]
        self.text_cleaner = TextCleaner()
        self.sr = sr
        self.df = pd.DataFrame(self.data_list)
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)
        self.data_augmentation = data_augmentation and not validation
        self.max_mel_length = 192
        self.min_length = min_length

        # Read OOD data (Out-of-Distribution data)
        with open(OOD_data, 'r', encoding='utf-8') as f:
            self.ptexts = [line.split('|')[1] for line in f.readlines() if '.wav' in line.split('|')[0]]

        self.root_path = root_path

        # Validate all audio files
        self.data_list = [data for data in self.data_list if is_valid_audio(os.path.join(self.root_path, data[0]))]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        path = data[0]

        # Load audio and text data
        wave, text_tensor, speaker_id = self._load_tensor(data)

        # Convert wave to mel spectrogram
        mel_tensor = preprocess(wave).squeeze()

        # Generate reference sample (random speaker data)
        ref_data = self.df[self.df[2] == str(speaker_id)].sample(n=1).iloc[0].tolist()
        ref_mel_tensor, ref_label = self._load_data(ref_data[:3])

        # Generate out-of-distribution text
        ps = ""
        while len(ps) < self.min_length:
            rand_idx = np.random.randint(0, len(self.ptexts) - 1)
            ps = self.ptexts[rand_idx]

        text = self.text_cleaner(ps)
        text.insert(0, 0)
        text.append(0)
        ref_text = torch.LongTensor(text)

        return speaker_id, mel_tensor, text_tensor, ref_text, ref_mel_tensor, ref_label, path, wave

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)

        # Load and resample audio if needed
        wave, sr = sf.read(os.path.join(self.root_path, wave_path))
        if wave.shape[-1] == 2:  # Stereo to mono conversion
            wave = wave[:, 0].squeeze()
        if sr != self.sr:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=self.sr)
            logger.info(f"Resampling {wave_path} from {sr}Hz to {self.sr}Hz")

        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)

        # Clean text
        text = self.text_cleaner(text)
        text.insert(0, 0)  # Start token
        text.append(0)  # End token
        text = torch.LongTensor(text)

        return wave, text, speaker_id

    def _load_data(self, data):
        wave, text_tensor, speaker_id = self._load_tensor(data)
        mel_tensor = preprocess(wave).squeeze()

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, speaker_id

# Collate function to batch data
class Collater:
    def __init__(self, return_wave=False):
        self.return_wave = return_wave
        self.min_mel_length = 192
        self.max_mel_length = 192

    def __call__(self, batch):
        batch_size = len(batch)

        # Sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])
        max_rtext_length = max([b[3].shape[0] for b in batch])

        # Prepare tensors
        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        ref_texts = torch.zeros((batch_size, max_rtext_length)).long()
        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        paths = ['' for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]

        # Batch data
        for bid, (label, mel, text, ref_text, ref_mel, ref_label, path, wave) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            rtext_size = ref_text.size(0)
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            ref_labels[bid] = ref_label
            waves[bid] = wave

        return waves, texts, input_lengths, ref_texts, ref_lengths, mels, output_lengths, ref_mels

# DataLoader Builder function
def build_dataloader(path_list, root_path, validation=False, OOD_data="/content/dataset.txt", min_length=50, batch_size=4, num_workers=1, device='cpu', collate_config={}, dataset_config={}):
    dataset = FilePathDataset(path_list, root_path, OOD_data=OOD_data, min_length=min_length, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=not validation, num_workers=num_workers, drop_last=not validation, collate_fn=collate_fn, pin_memory=(device != 'cpu'))
    return data_loader
