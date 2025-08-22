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
    
    # VAD ayarları
    SOUND_THRESHOLD = config.SES_ESIK_DEGERI
    SILENCE_LIMIT_SECONDS = 1.5 # Konuşma bittikten sonra ne kadar sessizlik beklenmeli
    
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
                rms = math.sqrt(np.mean(audio_data**2))
                
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
                elif not speaking and (time.time() - start_time) > record_seconds * 2: # Max wait for speech
                    print(f"No speech detected for {record_seconds * 2} seconds. Stopping recording.")
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
