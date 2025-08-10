# -*- coding: utf-8 -*-
import torch
import torchaudio
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model


class EmbeddingGenerator:
    def __init__(self, model_name: str, ornekleme_orani: int = 16000, device: str = None):
        """
        Ses verilerinden embedding üreten sınıf.

        Args:
            model_name (str): HuggingFace model adı.
            ornekleme_orani (int): Hedef sampling rate.
            device (str): 'cuda', 'cpu' veya None (otomatik seçer).
        """
        self.model_name = model_name
        self.target_sampling_rate = ornekleme_orani
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"'{self.model_name}' modeli ve işlemcisi yükleniyor...")
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2Model.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        print(f"Model ve işlemci yüklendi. Kullanılan cihaz: {self.device}")

    @torch.no_grad()
    def get_embedding_from_file(self, audio_path: str) -> torch.Tensor:
        """Ses dosyasından embedding üretir."""
        speech_array, sampling_rate = sf.read(audio_path)
        return self._process_audio(speech_array, sampling_rate)

    @torch.no_grad()
    def get_embedding_from_data(self, audio_data: np.ndarray, sampling_rate: int) -> torch.Tensor:
        """Bellekteki ses verisinden embedding üretir."""
        return self._process_audio(audio_data, sampling_rate)

    def _process_audio(self, audio_array: np.ndarray, sampling_rate: int) -> torch.Tensor:
        """Ses verisini işleyerek embedding döndürür."""
        # NumPy array → Torch tensor
        speech_tensor = torch.from_numpy(audio_array).float()

        if speech_tensor.ndim == 0 or speech_tensor.numel() == 0:
            print("Uyarı: Boş veya geçersiz ses verisi.")
            return torch.empty(0)

        # Mono değilse tek kanala indir
        if speech_tensor.ndim > 1:
            speech_tensor = torch.mean(speech_tensor, dim=1)

        # Resample
        if sampling_rate != self.target_sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sampling_rate,
                new_freq=self.target_sampling_rate
            )
            speech_tensor = resampler(speech_tensor)

        # Model girişine uygun format
        input_values = self.processor(
            speech_tensor,
            sampling_rate=self.target_sampling_rate,
            return_tensors="pt",
            padding=True
        ).input_values.to(self.device)

        # Model çıkışı → ortalama ile embedding
        hidden_states = self.model(input_values).last_hidden_state
        embedding = torch.mean(hidden_states, dim=1).squeeze()

        return embedding.cpu()  # CPU'ya al ki RAM'de kullanılabilsin
