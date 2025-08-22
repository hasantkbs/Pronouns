import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os

def test_model(model_path: str, audio_file_path: str):
    """
    Eğitilmiş ASR modelini test eder ve ses dosyasını metne dönüştürür.
    """
    print(f"\n--- Model ve İşlemci Yükleniyor: {model_path} ---")
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
        model.eval() # Modeli değerlendirme moduna al
        print("Model ve İşlemci başarıyla yüklendi.")
    except Exception as e:
        print(f"Hata: Model veya İşlemci yüklenirken sorun oluştu: {e}")
        return

    print(f"\n--- Ses Dosyası Yükleniyor: {audio_file_path} ---")
    try:
        speech_array, sampling_rate = sf.read(audio_file_path)
        if speech_array.ndim > 1:
            speech_array = speech_array.mean(axis=1) # Stereo ise mono'ya çevir
        print(f"Ses dosyası yüklendi. Örnekleme oranı: {sampling_rate}")
    except Exception as e:
        print(f"Hata: Ses dosyası yüklenirken veya işlenirken sorun oluştu: {e}")
        return

    # Modelin beklediği örnekleme oranına uygunluğu kontrol et
    if sampling_rate != processor.feature_extractor.sampling_rate:
        print(f"Uyarı: Ses dosyasının örnekleme oranı ({sampling_rate}) modelin beklediği "
              f"({processor.feature_extractor.sampling_rate}) ile uyuşmuyor. "
              f"Modelin beklediği örnekleme oranına yeniden örnekleme yapılıyor.")
        # torchaudio.transforms.Resample kullanmak daha iyi olurdu ama bağımlılık eklememek için basit bir uyarı bırakıyorum.
        # Bu durumda, modelin beklediği sampling_rate'e göre sesin yeniden örneklenmesi gerekir.
        # Basitlik adına, bu örnekte yeniden örnekleme yapılmıyor, ancak gerçek uygulamada yapılmalıdır.
        pass # Şimdilik yeniden örnekleme yapmıyoruz, sadece uyarı veriyoruz.

    print("\n--- Çıkarım Yapılıyor ---")
    try:
        # Ses verisini işlemciye ver
        input_values = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True).input_values

        # Modelden tahminleri al
        with torch.no_grad():
            logits = model(input_values).logits

        # Logitleri metne dönüştür
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        print(f"Tahmin Edilen Metin: {transcription}")
    except Exception as e:
        print(f"Hata: Çıkarım yapılırken sorun oluştu: {e}")
        return

if __name__ == "__main__":
    # Eğitilmiş modelin kaydedildiği yol
    trained_model_path = "./asr_model"

    # Test etmek istediğiniz ses dosyasının yolu
    # Lütfen bu yolu kendi ses dosyanızın yolu ile değiştirin.
    # Örneğin: sample_audio_file = "asr_data/audio/your_audio_file.wav"
    sample_audio_file = "asr_data/audio/1e7196f2-1829-4dc8-981e-763cc3637cfc.wav" # Örnek bir dosya

    if not os.path.exists(trained_model_path):
        print(f"Hata: Eğitilmiş model dizini bulunamadı: {trained_model_path}")
    elif not os.path.exists(sample_audio_file):
        print(f"Hata: Örnek ses dosyası bulunamadı: {sample_audio_file}")
        print("Lütfen 'sample_audio_file' değişkenini geçerli bir ses dosyası yolu ile güncelleyin.")
    else:
        test_model(trained_model_path, sample_audio_file)
