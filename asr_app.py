import os
import csv
import pyaudio
import wave
import uuid
import threading

# --- Ayarlar ---
DATA_DIR = "asr_data"
TRANSCRIPTS_FILE = os.path.join(DATA_DIR, "transcripts.csv")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")

# Kayıt Ayarları
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

def ensure_dirs_and_files():
    """Gerekli klasörlerin ve dosyaların var olduğundan emin olur."""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    if not os.path.exists(TRANSCRIPTS_FILE):
        with open(TRANSCRIPTS_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "transcript"])

def record_sentence():
    """
    Kullanıcıdan bir cümle alır, ses kaydı yapar ve transkript dosyasına ekler.
    """
    ensure_dirs_and_files()
    
    transcript = input("Lütfen kaydedilecek cümleyi yazın (veya çıkmak için boş bırakıp Enter'a basın): ")
    if not transcript:
        return

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    
    print("\n--------------------------------------------------")
    print(">>> Kayıt başladı... Konuşabilirsiniz.")
    print(">>> Bitirmek için Enter'a basın.")
    
    frames = []
    stop_recording = threading.Event()

    def read_audio():
        while not stop_recording.is_set():
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except IOError as e:
                # Bu durum genellikle stream kapatıldığında olur
                if e.errno == pyaudio.paInputOverflowed:
                    print("Uyarı: Giriş taşması. Bazı ses verileri kaybolmuş olabilir.")
                else:
                    break

    audio_thread = threading.Thread(target=read_audio)
    audio_thread.start()

    input()  # Kullanıcının Enter'a basmasını bekle
    stop_recording.set()
    audio_thread.join()

    print(">>> Kayıt tamamlandı.")
    
    # Stream'i ve PyAudio'yu güvenli bir şekilde kapat
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Dosyayı kaydet
    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(AUDIO_DIR, filename)
    
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        
    print(f"Ses dosyası kaydedildi: {filepath}")

    # Transkripti kaydet
    with open(TRANSCRIPTS_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([filename, transcript])
        
    print("Transkript kaydedildi.")
    print("--------------------------------------------------")


def main():
    """Ana uygulama menüsü."""
    print("\n--- Cümle Tanıma ASR Projesi ---")
    while True:
        print("\n1. Yeni Cümle Kaydet")
        print("2. Modeli Eğit (Henüz aktif değil)")
        print("3. Canlı Tanıma Yap (Henüz aktif değil)")
        print("4. Çıkış")
        choice = input("Seçiminiz: ")

        if choice == '1':
            record_sentence()
        elif choice == '4':
            break
        else:
            print("Geçersiz seçim. Lütfen 1-4 arası bir seçim yapın.")

if __name__ == "__main__":
    main()

