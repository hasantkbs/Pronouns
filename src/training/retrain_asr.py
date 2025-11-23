# -*- coding: utf-8 -*-
import os
import sys
from src.utils.utils import save_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
import torch
import re
from datasets import load_dataset, Audio, Features, Value
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
import dataclasses
from typing import Dict, List, Optional, Union

# --- Global Değişkenler ---
# processor artık global değil, fonksiyonlara argüman olarak geçecek

# --- Veri Kümesi Hazırlama ---

def update_audio_paths(batch, split_name):
    """
    Common Voice dataset'indeki ses dosyası yollarını günceller.
    """
    # Assuming audio files are in downloaded_data/train/audio/ or downloaded_data/validation/audio/
    # and the 'path' in the dataset is just the filename (e.g., "common_voice_tr_12345.mp3")
    base_audio_dir = os.path.join("downloaded_data", split_name, "audio")
    batch["audio"]["path"] = os.path.join(base_audio_dir, os.path.basename(batch["path"]))
    return batch

def prepare_dataset(batch, processor: Wav2Vec2Processor): # processor argüman olarak eklendi
    """
    Ses verisini işler ve metin transkriptlerini modelin anlayacağı formata getirir.
    """
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

def remove_special_characters(batch):
    """
    Metin transkriptlerindeki özel karakterleri ve rakamları temizler.
    """
    chars_to_remove_regex = r'[^a-zçğıöşü\s]'
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"].lower())
    return batch

# --- Veri Harmanlayıcı (Data Collator) ---

@dataclasses.dataclass
class DataCollatorCTCWithPadding:
    """
    Değişken uzunluktaki ses ve etiket dizilerini toplu halde işler.
    """
    processor: Wav2Vec2Processor # Düzeltildi
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

# --- Ana Eğitim Fonksiyonu ---

def main():
    # processor artık global değil
    
    # 1. İşlemciyi (Processor) Yükle
    print(f"'{config.MODEL_NAME}' için işlemci yükleniyor...")
    processor = Wav2Vec2Processor.from_pretrained(config.MODEL_NAME)

    # 2. Veri Setini Yükle
    # Veri setinin şemasını manuel olarak tanımla
    features = Features({
        "client_id": Value("string"),
        "path": Value("string"),
        "audio": Audio(sampling_rate=16000),
        "sentence": Value("string"),
        "up_votes": Value("int64"),
        "down_votes": Value("int64"),
        "age": Value("string"),
        "gender": Value("string"),
        "accent": Value("string"),
        "locale": Value("string"),
        "segment": Value("string"),
        "variant": Value("string"), # Yeni eklenen sütun
    })

    print("Yerel Parquet dosyaları için yollar tanımlanıyor...")
    train_parquet_path = "downloaded_data/train/0000.parquet"
    validation_parquet_path = "downloaded_data/validation/0000.parquet" # Düzeltilmiş yazım hatası

    print(f"Train yolu kontrol ediliyor: {train_parquet_path}")
    print(f"Validation yolu kontrol ediliyor: {validation_parquet_path}")

    if not os.path.exists(train_parquet_path) or not os.path.exists(validation_parquet_path):
        raise FileNotFoundError("Lütfen 'downloaded_data' klasörünüzdeki Parquet dosyalarının doğru yolda ve isimde olduğundan emin olun. Beklenen yollar: downloaded_data/train/0000.parquet ve downloaded_data/validation/0000.parquet")

    # data_files değişkeni tanımlandı
    data_files = {
        "train": train_parquet_path,
        "validation": validation_parquet_path,
    }

    print("Parquet dosyaları yükleniyor (load_dataset çağrısı)...")
    common_voice_dict = load_dataset("parquet", data_files=data_files, features=features)
    print("Parquet dosyaları başarıyla yüklendi.")
    train_dataset = common_voice_dict["train"]
    eval_dataset = common_voice_dict["validation"]

    # Ses dosyalarının doğru yollarını ayarlama
    print("Ses dosyası yolları güncelleniyor...")
    train_dataset = train_dataset.map(lambda batch: update_audio_paths(batch, "train"))
    eval_dataset = eval_dataset.map(lambda batch: update_audio_paths(batch, "validation")) # 'validation' yazım hatası düzeltildi

    # 3. Veri Setini Ön İşle
    print("Veri seti temizleniyor ve ön işleniyor...")
    train_dataset = train_dataset.map(remove_special_characters)
    eval_dataset = eval_dataset.map(remove_special_characters)
    
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=16000))

    # prepare_dataset'e processor argümanı eklendi
    train_dataset = train_dataset.map(lambda batch: prepare_dataset(batch, processor), remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(lambda batch: prepare_dataset(batch, processor), remove_columns=eval_dataset.column_names)
    
    print("Veri seti eğitime hazır.")

    # 4. Data Collator'ı Oluştur
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # 5. Modeli Yükle
    print(f"'{config.MODEL_NAME}' modeli yükleniyor...")
    model = Wav2Vec2ForCTC.from_pretrained(
        config.MODEL_NAME,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    # 6. Eğitim Ayarlarını Tanımla
    training_args = TrainingArguments(
        output_dir=config.FINETUNE_OUTPUT_DIR,
        group_by_length=True,
        per_device_train_batch_size=config.FINETUNE_BATCH_SIZE,
        per_device_eval_batch_size=config.FINETUNE_BATCH_SIZE,
        evaluation_strategy="steps",
        num_train_epochs=config.NUM_FINETUNE_EPOCHS,
        fp16=True if torch.cuda.is_available() else False,
        save_steps=config.FINETUNE_EVAL_STEPS,
        eval_steps=config.FINETUNE_EVAL_STEPS,
        logging_steps=config.FINETUNE_LOGGING_STEPS,
        learning_rate=config.FINETUNE_LEARNING_RATE,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
    )

    # 7. Trainer'ı Oluştur
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,  # feature_extractor verilebilir
        # compute_metrics=compute_metrics, # compute_metrics burada tanımlı değil, train_asr.py'de var
    )

    # 8. Eğitimi Başlat
    
        print("\n--- Model Eğitimi Başlatılıyor ---")
        trainer.train()
    
        # 9. Eğitilmiş Modeli Kaydet
        print("Eğitim tamamlandı. Model kaydediliyor...")
        save_model(trainer.model, processor, config.FINETUNE_OUTPUT_DIR)
        print(f"Model başarıyla '{config.FINETUNE_OUTPUT_DIR}' klasörüne kaydedildi.")
    