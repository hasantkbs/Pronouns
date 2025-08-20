import os
import json
import shutil

def prepare_speech_commands_dataset():
    base_dataset_path = "external_datasets/speech_commands_v0.02/"
    target_user_path = "users/speech_commands_test"
    meta_file_path = os.path.join(target_user_path, "meta.json")

    meta_data = {"user_id": "speech_commands_test", "kelimeler": []}
    
    print(f"--- '{base_dataset_path}' dizininden veri hazırlanıyor ---")

    # Iterate through each word directory
    for word_label in os.listdir(base_dataset_path):
        word_dir_path = os.path.join(base_dataset_path, word_label)
        
        # Skip if it's not a directory or if it's a hidden file/directory
        if not os.path.isdir(word_dir_path) or word_label.startswith('.'):
            continue

        print(f"  '{word_label}' kelimesi işleniyor...")
        
        # Iterate through each audio file in the word directory
        for i, audio_file_name in enumerate(os.listdir(word_dir_path)):
            if audio_file_name.endswith(".wav"):
                source_audio_path = os.path.join(word_dir_path, audio_file_name)
                
                # Create a unique name for the copied file to avoid conflicts
                # Format: label_originalfilename.wav (e.g., "yes_0a7c2a8d_nohash_0.wav")
                # Or, if original filename already contains label, just use it.
                # For simplicity, let's just copy and rename if needed.
                
                # To match the original system's naming convention (word_number.wav)
                # we'll use a simple counter for each word.
                new_audio_file_name = f"{word_label}_{i+1}.wav"
                destination_audio_path = os.path.join(target_user_path, new_audio_file_name)
                
                try:
                    shutil.copy(source_audio_path, destination_audio_path)
                    meta_data["kelimeler"].append({"etiket": word_label, "dosya": new_audio_file_name})
                except Exception as e:
                    print(f"    Hata: '{source_audio_path}' kopyalanırken bir sorun oluştu: {e}")
    
    if not meta_data["kelimeler"]:
        print("Uyarı: Hiç .wav dosyası bulunamadı veya veri seti yapısı beklenenden farklı.")
        return

    with open(meta_file_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)

    print(f"\nVeri hazırlığı tamamlandı! '{meta_file_path}' dosyası oluşturuldu.")

if __name__ == "__main__":
    prepare_speech_commands_dataset()
