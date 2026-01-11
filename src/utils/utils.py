# -*- coding: utf-8 -*-
import pyaudio
import wave
import numpy as np
import threading
import time
import sys
import os
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

def save_model(model, processor, path: str):
    """
    Saves the model and processor to the given path.
    """
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    print(f"✅ Model saved to {path}")

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
