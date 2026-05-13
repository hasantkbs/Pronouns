# create_personalized_corpus.py
import os
from pathlib import Path

def create_corpus(user_id="FurkanV1"):
    corpus_path = "data/corpus_personalized.txt"
    
    # 1. Kelime setini oku
    word_set_path = "datasets/words_set/wordSet.txt"
    with open(word_set_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    
    # 2. Furkan'ın özel kelimelerini (metadata'dan) al
    metadata_path = f"data/users/{user_id}/metadata_words.csv"
    user_words = []
    if os.path.exists(metadata_path):
        import pandas as pd
        df = pd.read_csv(metadata_path)
        user_words = df["transcription"].unique().tolist()
    
    all_vocab = list(set(words + user_words))
    
    # 3. Yaygın cümleler ekle (Furkan'ın ihtiyacı olabilecek)
    common_phrases = [
        "su verir misin",
        "yemek yemek istiyorum",
        "televizyonu aç",
        "ışığı kapat",
        "nasılsın",
        "iyiyim teşekkür ederim",
        "yardım et",
        "kapıyı aç",
        "uyumak istiyorum",
        "lavaboya gitmek istiyorum",
        "acıktım",
        "susadım",
        "dışarı çıkalım",
        "annemi çağır",
        "babamı çağır",
        "ilaçlarımı ver",
        "saat kaç",
        "bugün günlerden ne",
        "hava çok güzel",
        "seni seviyorum"
    ]
    
    with open(corpus_path, "w", encoding="utf-8") as f:
        # Kelimeleri tek tek ekle (Ağırlık vermek için 5'er kez)
        for w in all_vocab:
            for _ in range(5):
                f.write(f"{w}\n")
        
        # Cümleleri ekle (Ağırlık vermek için 10'ar kez)
        for phrase in common_phrases:
            for _ in range(10):
                f.write(f"{phrase}\n")
        
        # Mevcut genel corpus'u da ekle (Eğer varsa)
        if os.path.exists("data/corpus.txt"):
            with open("data/corpus.txt", "r", encoding="utf-8") as old_f:
                f.write(old_f.read())
                
    print(f"Kişiselleştirilmiş corpus oluşturuldu: {corpus_path}")
    return corpus_path

if __name__ == "__main__":
    create_corpus()
