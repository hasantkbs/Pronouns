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

def record_audio(file_path: str = "temp_recording.wav", record_seconds: int = 5, prompt: str = None) -> str:
    """
    Mikrofon kullanarak ses kaydı yapar ve dosyaya kaydeder.
    Konuşma bozukluğu olan bireyler için optimize edilmiştir.

    Args:
        file_path (str): Kaydedilecek ses dosyasının yolu
        record_seconds (int): Kayıt süresi (saniye)
        prompt (str): Kullanıcıya gösterilecek mesaj

    Returns:
        str: Kaydedilen ses dosyasının yolu
    """
    # Kayıt ayarları
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  # Mono
    RATE = 16000  # 16kHz
    CHUNK = 1024
    
    # PyAudio nesnesini oluştur
    audio = pyaudio.PyAudio()
    
    try:
        # Stream'i aç
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        # Kullanıcıya bilgi ver
        if prompt:
            print(prompt)
        else:
            print(f"🎤 {record_seconds} saniye ses kaydı başlıyor...")
        
        print("🔴 Kayıt başladı - Konuşabilirsiniz...")
        
        frames = []
        
        # Ses kaydı
        for i in range(0, int(RATE / CHUNK * record_seconds)):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except IOError as e:
                if e.errno == pyaudio.paInputOverflowed:
                    print("⚠️  Uyarı: Ses girişi taştı, bazı veriler kaybolmuş olabilir.")
                else:
                    raise
        
        print("🟢 Kayıt tamamlandı!")
        
        # Stream'i kapat
        stream.stop_stream()
        stream.close()
        
        # Ses dosyasını kaydet
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
        # PyAudio'yu temizle
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
