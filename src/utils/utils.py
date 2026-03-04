# -*- coding: utf-8 -*-
import pyaudio
import wave
import numpy as np
import threading
import time
import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config


# --- Ses Kayıt Fonksiyonu ---

# -*- coding: utf-8 -*-
import pyaudio
import wave
import numpy as np
import threading
import time
import sys
import os
import math # Added for RMS calculation
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

# --- Ses Kayıt Fonksiyonu ---

def record_audio(file_path: str = config.GECICI_DOSYA_YOLU, record_seconds: int = config.KAYIT_SURESI_SN, prompt: str = None) -> str:
    """
    Mikrofon kullanarak ses kaydı yapar ve dosyaya kaydeder.
    Konuşma bozukluğu olan bireyler için optimize edilmiştir.
    Ses aktivitesi algılama (VAD) içerir.

    Args:
        file_path (str): Kaydedilecek ses dosyasının yolu
        record_seconds (int): Kayıt süresi (saniye). Bu süre, ses algılandıktan sonraki aktif konuşma süresidir.
        prompt (str): Kullanıcıya gösterilecek mesaj

    Returns:
        str: Kaydedilen ses dosyasının yolu
    """
    # Kayıt ayarları config dosyasından alınır
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  # Mono
    RATE = config.ORNEKLEME_ORANI
    CHUNK = 1024
    
    # VAD ayarları (konuşma bozukluğu için optimize)
    SOUND_THRESHOLD = config.SES_ESIK_DEGERI
    SILENCE_LIMIT_SECONDS = config.VAD_SILENCE_LIMIT_SEC  # Konuşma bittikten sonra beklenen sessizlik
    SPEECH_WAIT_SECONDS = config.VAD_SPEECH_WAIT_SEC  # Konuşma beklenen maksimum süre
    
    audio = pyaudio.PyAudio()
    
    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        if prompt:
            print(prompt)
        else:
            print(f"🎤 Ses kaydı başlıyor. Konuşma algılandığında kayıt başlayacak...")
        
        print("🔴 Dinleniyor... Konuşmaya başlayın.")
        
        frames = []
        speaking = False
        silence_start_time = None
        
        start_time = time.time()
        
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                # int16 üzerinde karesel işlem taşma yapabileceği için önce float'a çevir
                audio_float = audio_data.astype(np.float32)
                rms = np.sqrt(np.mean(audio_float ** 2))
                
                if rms > SOUND_THRESHOLD * 32767: # Normalize threshold to 16-bit audio range
                    if not speaking:
                        print("🗣️ Konuşma algılandı, kayıt başladı!")
                        speaking = True
                        silence_start_time = None
                    frames.append(data)
                elif speaking:
                    frames.append(data) # Keep recording for a short silence period
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    elif (time.time() - silence_start_time) > SILENCE_LIMIT_SECONDS:
                        print(f"Silence detected for {SILENCE_LIMIT_SECONDS} seconds. Stopping recording.")
                        break # Stop if silence limit reached after speaking
                
                # Stop recording after max duration if no speech detected or if speech has been ongoing
                if speaking and (time.time() - start_time) > record_seconds + SILENCE_LIMIT_SECONDS:
                    print(f"Max recording duration ({record_seconds}s active speech + {SILENCE_LIMIT_SECONDS}s silence) reached. Stopping.")
                    break
                elif not speaking and (time.time() - start_time) > SPEECH_WAIT_SECONDS:  # Konuşma bozukluğu için daha uzun bekleme
                    print(f"No speech detected for {SPEECH_WAIT_SECONDS} seconds. Stopping recording.")
                    break

            except IOError as e:
                if e.errno == pyaudio.paInputOverflowed:
                    print("⚠️  Uyarı: Ses girişi taştı, bazı veriler kaybolmuş olabilir.")
                else:
                    raise
        
        print("🟢 Kayıt tamamlandı!")
        
        stream.stop_stream()
        stream.close()
        
        if not frames:
            print("❌ Ses algılanmadı, dosya kaydedilmedi.")
            return None

        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        
        print(f"💾 Ses dosyası kaydedildi: {file_path}")
        return file_path
        
    except Exception as e:
        print(f"❌ Ses kaydı sırasında hata oluştu: {e}")
        return None
        
    finally:
        audio.terminate()

def get_audio_info(file_path: str) -> dict:
    """
    Ses dosyası hakkında bilgi döndürür.
    
    Args:
        file_path (str): Ses dosyasının yolu
        
    Returns:
        dict: Ses dosyası bilgileri
    """
    try:
        with wave.open(file_path, 'rb') as wf:
            return {
                "channels": wf.getnchannels(),
                "sample_width": wf.getsampwidth(),
                "frame_rate": wf.getframerate(),
                "frames": wf.getnframes(),
                "duration": wf.getnframes() / wf.getframerate()
            }
    except Exception as e:
        print(f"❌ Ses dosyası bilgisi alınamadı: {e}")
        return {}

def calculate_audio_quality(file_path: str) -> dict:
    """
    Ses dosyasının kalitesini analiz eder ve skorlar.
    
    Args:
        file_path (str): Ses dosyasının yolu
        
    Returns:
        dict: Kalite metrikleri ve skor
    """
    try:
        with wave.open(file_path, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
            
            # Ses bilgileri
            duration = wf.getnframes() / wf.getframerate()
            sample_rate = wf.getframerate()
            
            # RMS (Root Mean Square) hesapla
            rms = np.sqrt(np.mean(audio_float ** 2)) * 32768.0  # Denormalize for display
            
            # SNR tahmini (basit yöntem: sesli bölümler vs sessiz bölümler)
            # Sesli bölümler için eşik
            threshold = config.SES_ESIK_DEGERI * 32767
            speech_frames = audio_data[np.abs(audio_data) > threshold]
            silence_frames = audio_data[np.abs(audio_data) <= threshold]
            
            if len(speech_frames) > 0 and len(silence_frames) > 0:
                speech_power = np.mean(speech_frames.astype(np.float32) ** 2)
                noise_power = np.mean(silence_frames.astype(np.float32) ** 2)
                if noise_power > 0:
                    snr_db = 10 * np.log10(speech_power / noise_power)
                else:
                    snr_db = 50  # Çok temiz ses
            else:
                snr_db = 30  # Varsayılan değer
            
            # Kalite skorunu hesapla (0-100) - Konuşma bozukluğu için daha toleranslı
            quality_score = 100
            
            # RMS kontrolü (daha toleranslı)
            if rms < config.MIN_RMS_LEVEL:
                quality_score -= 20  # Çok düşük ses (daha az ceza)
            elif rms > config.MAX_RMS_LEVEL:
                quality_score -= 15  # Distortion riski (daha az ceza)
            
            # Süre kontrolü (konuşma bozukluğu için daha toleranslı)
            if duration < config.MIN_DURATION_SEC:
                quality_score -= 20  # Çok kısa (daha az ceza)
            elif duration > config.MAX_DURATION_SEC:
                quality_score -= 10  # Çok uzun (daha az ceza, yavaş konuşma normal)
            
            # SNR kontrolü (daha toleranslı)
            if snr_db < 8:  # Daha düşük eşik
                quality_score -= 15  # Düşük SNR (daha az ceza)
            elif snr_db < 12:  # Daha düşük eşik
                quality_score -= 5  # Orta SNR (daha az ceza)
            
            # Skoru 0-100 aralığında tut
            quality_score = max(0, min(100, quality_score))
            
            return {
                "rms": float(rms),
                "snr_db": float(snr_db),
                "duration": float(duration),
                "quality_score": float(quality_score),
                "is_valid": quality_score >= config.QUALITY_THRESHOLD
            }
    except Exception as e:
        print(f"❌ Ses kalitesi analizi başarısız: {e}")
        return {
            "rms": 0,
            "snr_db": 0,
            "duration": 0,
            "quality_score": 0,
            "is_valid": False
        }

def check_consistency(durations: list, tolerance: float = 0.5) -> dict:
    """
    Aynı kelime için farklı tekrarların süre tutarlılığını kontrol eder.
    
    Args:
        durations (list): Kayıt süreleri listesi (saniye)
        tolerance (float): Tolerans değeri (saniye)
        
    Returns:
        dict: Tutarlılık bilgileri
    """
    if len(durations) < 2:
        return {
            "is_consistent": True,
            "avg_duration": durations[0] if durations else 0,
            "std_deviation": 0,
            "max_diff": 0
        }
    
    avg_duration = sum(durations) / len(durations)
    std_deviation = np.std(durations)
    max_diff = max(durations) - min(durations)
    
    # Tüm süreler ortalamaya yakınsa tutarlı kabul et
    is_consistent = max_diff <= tolerance
    
    return {
        "is_consistent": is_consistent,
        "avg_duration": float(avg_duration),
        "std_deviation": float(std_deviation),
        "max_diff": float(max_diff),
        "tolerance": tolerance
    }

def play_audio(file_path: str) -> bool:
    """
    Ses dosyasını oynatır (basit terminal tabanlı).
    
    Args:
        file_path (str): Oynatılacak ses dosyasının yolu
        
    Returns:
        bool: Başarılı ise True
    """
    try:
        import subprocess
        import platform
        
        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", file_path], check=True)
        elif system == "Linux":
            subprocess.run(["aplay", file_path], check=True)
        elif system == "Windows":
            subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{file_path}').PlaySync()"], check=True)
        else:
            print(f"⚠️  Bu işletim sistemi için ses oynatma desteği yok: {system}")
            return False
        return True
    except Exception as e:
        print(f"⚠️  Ses oynatılamadı: {e}")
        return False

def record_audio_auto(
    file_path: str,
    sound_threshold: float = None,
    silence_limit_sec: float = None,
    speech_wait_sec: float = None,
) -> str | None:
    """
    ENTER gerektirmeden otomatik VAD ile ses kaydı yapar.

    Mikrofonu sürekli dinler; ses eşiği aşılınca kaydı başlatır,
    sessizlik limiti dolunca otomatik olarak durdurur ve dosyaya yazar.

    Farklı noktaları:
    - Herhangi bir kullanıcı girişi (ENTER vb.) beklemez.
    - Küçük CHUNK (512) kullanarak VAD tepki süresini yarıya indirir.
    - Kısa sessizlik limiti (varsayılan 1 sn) ile kelime kayıtları için optimize.
    - speech_wait_sec içinde hiç ses gelmezse None döndürür.

    Args:
        file_path: Kaydedilecek WAV dosyasının yolu.
        sound_threshold: Normalize ses eşiği (0-1). None -> config'den alınır.
        silence_limit_sec: Sessizlik limiti (sn). None -> config'den alınır.
        speech_wait_sec: İlk ses bekleme süresi (sn). None -> config'den alınır.

    Returns:
        Kaydedilen dosya yolu, ya da hiç ses algılanmazsa None.
    """
    import pyaudio
    import wave as _wave

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = config.ORNEKLEME_ORANI
    CHUNK = 512   # 32 ms — mevcut 1024'ten daha hızlı VAD yanıtı

    threshold_norm = sound_threshold if sound_threshold is not None else config.AUTO_SOUND_THRESHOLD
    silence_limit  = silence_limit_sec if silence_limit_sec is not None else config.AUTO_SILENCE_LIMIT_SEC
    speech_wait    = speech_wait_sec if speech_wait_sec is not None else config.AUTO_SPEECH_WAIT_SEC

    # int16 alanında eşik değerini hesapla
    THRESHOLD_INT16 = threshold_norm * 32767

    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        frames = []
        speaking = False
        silence_start = None
        wait_start = time.time()

        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except IOError:
                continue

            audio_int16 = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            rms = np.sqrt(np.mean(audio_int16 ** 2))

            if rms > THRESHOLD_INT16:
                if not speaking:
                    speaking = True
                    silence_start = None
                frames.append(data)
            elif speaking:
                frames.append(data)
                if silence_start is None:
                    silence_start = time.time()
                elif (time.time() - silence_start) >= silence_limit:
                    break
            else:
                # Henüz konuşma başlamadı; bekleme zaman aşımı kontrolü
                if (time.time() - wait_start) >= speech_wait:
                    break

        stream.stop_stream()
        stream.close()

        if not frames or not speaking:
            return None

        with _wave.open(file_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pa.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))

        return file_path

    except Exception as e:
        print(f"record_audio_auto hatasi: {e}")
        return None
    finally:
        pa.terminate()


def validate_recording(file_path: str, expected_word: str = None, asr_system=None) -> dict:
    """
    Kaydedilmiş bir ses dosyasını üç kademeli filtreden geçirir:

    1. Kalite filtresi  : RMS, SNR, süre (calculate_audio_quality kullanılır).
    2. Sessizlik filtresi: Konuşma çerçevesi oranı minimum eşiğin üstünde mi?
    3. ASR doğrulaması  : (opsiyonel) Konuşulan kelime beklenen kelimeyle eşleşiyor mu?
       Konuşma bozukluğuna tolerans için karakter hata oranı (CER) bazlı.

    Args:
        file_path: Doğrulanacak WAV dosyası.
        expected_word: Beklenen hedef kelime (ASR doğrulaması için).
        asr_system: ASRSystem örneği. None ise ASR doğrulaması atlanır.

    Returns:
        {
            "valid": bool,
            "reason": str,        # Reddetme nedeni (valid=False ise)
            "quality": dict,      # calculate_audio_quality() çıktısı
            "speech_ratio": float,
            "asr_text": str | None,
            "cer": float | None,
        }
    """
    import wave as _wave

    result = {
        "valid": False,
        "reason": "",
        "quality": {},
        "speech_ratio": 0.0,
        "asr_text": None,
        "cer": None,
    }

    # --- 1. Kalite filtresi ---
    quality = calculate_audio_quality(file_path)
    result["quality"] = quality

    if quality["rms"] < config.MIN_RMS_LEVEL:
        result["reason"] = f"Ses cok sessiz (RMS {quality['rms']:.0f} < {config.MIN_RMS_LEVEL})"
        return result

    if quality["duration"] < config.MIN_DURATION_SEC:
        result["reason"] = f"Kayit cok kisa ({quality['duration']:.2f}s < {config.MIN_DURATION_SEC}s)"
        return result

    if quality["snr_db"] < 6:
        result["reason"] = f"Gurultu fazla (SNR {quality['snr_db']:.1f} dB < 6 dB)"
        return result

    # --- 2. Sessizlik / boş ses filtresi ---
    try:
        with _wave.open(file_path, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        audio_int16 = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        threshold_int16 = config.AUTO_SOUND_THRESHOLD * 32767
        speech_frames = np.sum(np.abs(audio_int16) > threshold_int16)
        speech_ratio = speech_frames / max(len(audio_int16), 1)
        result["speech_ratio"] = float(speech_ratio)

        min_ratio = getattr(config, "AUTO_MIN_SPEECH_RATIO", 0.10)
        if speech_ratio < min_ratio:
            result["reason"] = (
                f"Konusma orani cok dusuk ({speech_ratio:.1%} < {min_ratio:.0%}); "
                "muhtemelen bos veya gurultulu kayit"
            )
            return result
    except Exception as e:
        result["reason"] = f"Ses analizi hatasi: {e}"
        return result

    # --- 3. ASR dogrulamasi (opsiyonel) ---
    if asr_system is not None and expected_word and getattr(config, "AUTO_ASR_VERIFY", False):
        try:
            text, confidence = asr_system.transcribe(file_path)
            result["asr_text"] = text

            if text is None:
                result["reason"] = "ASR bos sonuc donurdu"
                return result

            # Karakter hata orani hesapla (basit Levenshtein)
            cer = _char_error_rate(expected_word.lower(), text.lower())
            result["cer"] = cer
            max_cer = getattr(config, "AUTO_ASR_MAX_CER", 0.6)

            if cer > max_cer:
                result["reason"] = (
                    f"ASR eslesmedi: beklenen='{expected_word}' "
                    f"tanınan='{text}' CER={cer:.2f} > {max_cer}"
                )
                return result
        except Exception as e:
            # ASR hatası kayıt reddetmesin; sadece logla
            result["asr_text"] = None

    result["valid"] = True
    return result


def _char_error_rate(ref: str, hyp: str) -> float:
    """Basit karakter hata oranı (Levenshtein mesafesi / referans uzunluğu)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    ref_len = len(ref)
    hyp_len = len(hyp)
    # DP matrisi
    dp = list(range(hyp_len + 1))
    for i, rc in enumerate(ref):
        new_dp = [i + 1]
        for j, hc in enumerate(hyp):
            new_dp.append(min(
                dp[j + 1] + 1,
                new_dp[j] + 1,
                dp[j] + (0 if rc == hc else 1),
            ))
        dp = new_dp
    return dp[hyp_len] / ref_len


def save_model_and_processor(model, processor, path: str):
    """
    Saves the model and processor to the given path.
    """
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    print(f"✅ Model saved to {path}")

def normalize_path_for_cross_platform(file_path: str, base_path: Path) -> str:
    """
    Dosya yolunu CSV'ye kaydederken her zaman POSIX formatına (ileri slash)
    dönüştürür. Bu sayede Windows'ta kaydedilen yollar Linux/macOS'ta da
    doğru okunur.

    Kural:
      - Mutlak yol ise: base_path'e göre göreli yola çevir, POSIX olarak döndür.
      - Göreli yol ise: sadece separator'ları normalize et, POSIX olarak döndür.
    """
    # Önce gelen yoldaki ters slash'ları düzelt (Windows'tan gelmiş olabilir)
    normalized_str = str(file_path).replace("\\", "/")
    file_path_obj = Path(normalized_str)

    if file_path_obj.is_absolute():
        try:
            relative_path = file_path_obj.relative_to(base_path)
            return relative_path.as_posix()  # Her zaman forward slash
        except ValueError:
            return file_path_obj.as_posix()

    return file_path_obj.as_posix()  # Her zaman forward slash


def resolve_audio_path(raw_path: str, user_base: Path) -> str | None:
    """
    CSV'de saklanan göreli ya da mutlak bir ses dosyası yolunu, çalışan
    platformda geçerli mutlak yola dönüştürür.

    Desteklenen giriş biçimleri (hepsi aynı sonucu verir):
      - POSIX göreli : ``words/merhaba/rep1.wav``
      - Windows göreli: ``words\\merhaba\\rep1.wav``
      - Mutlak (herhangi platform): ``/home/.../data/users/Furkan/words/merhaba/rep1.wav``
        ya da ``C:\\...\\words\\merhaba\\rep1.wav``

    Args:
        raw_path  : CSV'den okunan ham yol dizesi.
        user_base : Kullanıcı veri klasörü (``data/users/<user_id>``).

    Returns:
        Dosyanın geçerli mutlak yolu (str), yoksa None.
    """
    if not raw_path or (isinstance(raw_path, float)):
        return None

    # 1. Separator normalizasyonu — her iki yön de desteklenir
    normalized = str(raw_path).replace("\\", "/")

    # 2. Göreli yol dene: user_base / normalized
    candidate = user_base / Path(normalized)
    if candidate.exists():
        return str(candidate)

    # 3. Mutlak yol olarak dene
    abs_candidate = Path(normalized)
    if abs_candidate.is_absolute() and abs_candidate.exists():
        return str(abs_candidate)

    # 4. Yalnızca "words/" sonrasını al ve user_base altında ara
    #    Örnek: "C:/old/path/data/users/X/words/merhaba/rep1.wav"
    #           → "words/merhaba/rep1.wav" → user_base / words/merhaba/rep1.wav
    lower = normalized.lower()
    marker = "/words/"
    idx = lower.rfind(marker)
    if idx != -1:
        rel_tail = normalized[idx + 1:]  # "words/merhaba/rep1.wav"
        candidate2 = user_base / Path(rel_tail)
        if candidate2.exists():
            return str(candidate2)

    return None

if __name__ == '__main__':
    print("🎧 Ses Kayıt Testi")
    print("=" * 30)
    
    # Test kaydı
    test_file = record_audio("test_recording.wav", 3, "Test kaydı başlıyor...")
    
    if test_file:
        info = get_audio_info(test_file)
        print(f"\n📊 Ses Dosyası Bilgileri:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Test dosyasını sil
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\n🗑️  Test dosyası silindi: {test_file}")
    else:
        print("❌ Test kaydı başarısız!")
