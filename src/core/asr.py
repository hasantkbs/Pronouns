# -*- coding: utf-8 -*-
import os
import torch
import librosa
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from peft import PeftModel
import config

try:
    from pyctcdecode import build_ctcdecoder
    PYCTCDECODE_AVAILABLE = True
except ImportError:
    PYCTCDECODE_AVAILABLE = False


from Levenshtein import distance as lev_dist

class ASRSystem:
    """
    Otomatik Konuşma Tanıma (ASR) sistemi - Wav2Vec2 tabanlı.
    Fuzzy matching desteği ile konuşma bozukluğu olan kullanıcının
    hatalı telaffuzlarını kayıtlı kelimelerle düzeltir.
    """

    def __init__(self, model_name=None, user_id=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lm_decoder = None
        self.user_id = user_id
        self.user_vocabulary = self._load_user_vocabulary(user_id) if user_id else []

        if model_name is None:
            model_name = config.MODEL_NAME

        if os.path.exists(model_name) and os.path.isdir(model_name):
            base_model_name = config.MODEL_NAME
            try:
                print(f"Temel model yukleniyor: {base_model_name}")
                self.processor = Wav2Vec2Processor.from_pretrained(base_model_name)
                base_model = Wav2Vec2ForCTC.from_pretrained(
                    base_model_name,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    vocab_size=len(self.processor.tokenizer),
                )
                print(f"Kisisellestirilmis adapter yukleniyor: {model_name}")
                self.model = PeftModel.from_pretrained(base_model, model_name)
                self.model.to(self.device)
                self.model.eval()
                print(f"ASR Sistemi hazir. Kisisellestirilmis model: {model_name} | Cihaz: {self.device}")
            except Exception as e:
                print(f"Kisisellestirilmis model yuklenemedi: {e}")
                print(f"Varsayilan model kullanilacak: {base_model_name}")
                self._load_base_model(base_model_name)
        else:
            self._load_base_model(model_name)

        self._build_lm_decoder()

    def _load_base_model(self, model_name):
        try:
            print(f"Model yukleniyor: {model_name}")
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"ASR Sistemi hazir. Model: {model_name} | Cihaz: {self.device}")
        except Exception as e:
            print(f"Model yuklenirken kritik hata: {e}")
            raise

    def _build_lm_decoder(self):
        """
        KenLM dil modeli mevcutsa pyctcdecode ile CTC beam search decoder kurar.
        Dil modeli yoksa greedy decoding kullanilir.
        """
        if not PYCTCDECODE_AVAILABLE:
            return

        lm_path = config.KENLM_MODEL_PATH
        if not Path(lm_path).exists():
            return

        try:
            vocab_dict = self.processor.tokenizer.get_vocab()
            sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
            labels = [token for token, _ in sorted_vocab]

            self._lm_decoder = build_ctcdecoder(
                labels=labels,
                kenlm_model=lm_path,
                alpha=config.LM_ALPHA,
                beta=config.LM_BETA,
            )
            print(f"Dil modeli yuklendi: {lm_path} (alpha={config.LM_ALPHA}, beta={config.LM_BETA})")
        except Exception as e:
            print(f"Dil modeli yuklenemedi, greedy decoding kullanilacak: {e}")
            self._lm_decoder = None

    def _greedy_decode(self, logits_np):
        """Standart argmax CTC decoding."""
        predicted_ids = torch.argmax(torch.from_numpy(logits_np), dim=-1)
        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return text.strip()

    def _beam_decode(self, logits_np):
        """KenLM destekli CTC beam search decoding."""
        import numpy as np
        log_probs = np.log(np.exp(logits_np[0]) / np.exp(logits_np[0]).sum(axis=-1, keepdims=True) + 1e-10)
        text = self._lm_decoder.decode(log_probs, beam_width=config.LM_BEAM_WIDTH)
        return text.strip()

    @staticmethod
    def _confidence_from_logits(logits_tensor):
        """
        Logit'lerden ortalama token güven skoru hesaplar (0-1 arası).
        Greedy seçilen token'in softmax olasılıklarının geometrik ortalamasıdır.
        Düşük skor -> model kararsız; yüksek skor -> güvenli tanıma.
        """
        probs = torch.softmax(logits_tensor, dim=-1)
        max_probs, _ = probs.max(dim=-1)
        non_blank = max_probs[max_probs < 0.99]
        if len(non_blank) == 0:
            return 0.0
        log_mean = torch.log(non_blank + 1e-10).mean()
        confidence = float(torch.exp(log_mean).clamp(0.0, 1.0))
        return confidence

    def _load_user_vocabulary(self, user_id):
        """Kullanıcının kaydettiği kelimeleri bir sözlük olarak yükler."""
        vocab = set()
        metadata_path = Path(config.BASE_PATH) / user_id / "metadata_words.csv"
        if metadata_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(metadata_path)
                vocab.update(df["transcription"].str.lower().unique())
                print(f"Kullanici sozrugu yuklendi: {len(vocab)} kelime.")
            except Exception as e:
                print(f"Kullanici sozrugu yuklenirken hata: {e}")
        return list(vocab)

    def _fuzzy_correct(self, raw_text):
        """
        Tanınan metindeki kelimeleri, kullanıcının bildiği kelimelerle karşılaştırır
        ve en yakın olanla değiştirerek anlamlı cümle kurar.
        """
        if not self.user_vocabulary or not raw_text:
            return raw_text

        words = raw_text.split()
        corrected_words = []
        threshold = getattr(config, "FUZZY_MATCH_THRESHOLD", 0.6)

        for word in words:
            best_match = word
            min_dist = float('inf')
            
            for vocab_word in self.user_vocabulary:
                dist = lev_dist(word, vocab_word)
                if dist < min_dist:
                    min_dist = dist
                    best_match = vocab_word
            
            # Mesafe kelime uzunluğunun belli bir oranından azsa düzeltilir
            if min_dist / max(len(word), 1) < (1 - threshold):
                corrected_words.append(best_match)
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)

    def transcribe(self, audio_path):
        """
        Ses dosyasini metne donusturur ve fuzzy correction ile anlamli hale getirir.
        """
        try:
            speech, sr = librosa.load(audio_path, sr=config.ORNEKLEME_ORANI)

            if len(speech) == 0 or np.max(np.abs(speech)) < 0.001:
                return None, 0.0

            input_values = self.processor(
                speech,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            ).input_values.to(self.device)

            with torch.no_grad():
                logits = self.model(input_values).logits

            confidence = self._confidence_from_logits(logits[0])

            if self._lm_decoder is not None:
                logits_np = logits.cpu().numpy()
                text = self._beam_decode(logits_np)
            else:
                text = self._greedy_decode(logits.cpu().numpy())

            if not text:
                return None, 0.0

            # --- Kelime duzeltme ve anlamli cumle kurma ---
            text = self._fuzzy_correct(text)

            return text, confidence

        except FileNotFoundError:
            print(f"Ses dosyasi bulunamadi: {audio_path}")
            return None, 0.0
        except Exception as e:
            print(f"ASR transkripsiyon hatasi: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0


if __name__ == "__main__":
    import sys

    user_id = "Furkan"
    personalized_model_dir = f"data/models/personalized_models/{user_id}"

    print("=" * 50)
    print("ASR Sistemi Test Modu")
    print("=" * 50)

    if os.path.exists(personalized_model_dir):
        asr_system = ASRSystem(model_name=personalized_model_dir)
    else:
        asr_system = ASRSystem()

    test_file = "data/users/Furkan/words/Furkan_kelime_1_rep1.wav"
    if len(sys.argv) > 1:
        test_file = sys.argv[1]

    if os.path.exists(test_file):
        print(f"\nTest dosyasi: {test_file}")
        text, confidence = asr_system.transcribe(test_file)
        if text:
            print(f"Tanınan metin : '{text}'")
            print(f"Guven skoru   : {confidence:.2%}")
        else:
            print("Tanima basarisiz veya sessizlik algilandi.")
    else:
        print(f"Test dosyasi bulunamadi: {test_file}")
