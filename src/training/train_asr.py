#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from src.utils.reporting import generate_report
from src.utils.utils import save_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
import argparse
from typing import Dict, List, Union
from functools import partial

import numpy as np
import pandas as pd
import torch
import soundfile as sf
import evaluate
import audiomentations as A

from datasets import Dataset
from inspect import signature

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
)

# ---- Custom CTC data collator for transformers>=4.55 ----
class DataCollatorCTCWithPadding:
    '''
    CTC tabanlı ASR (Wav2Vec2 vb.) için collator.
    - input_values (audio) ve labels ayrı ayrı pad edilir.
    - label padding'leri CTC kaybı için -100'e çevrilir.
    '''
    def __init__(self, processor: Wav2Vec2Processor, padding: Union[bool, str] = True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # input_values
        input_features = [{'input_values': f['input_values']} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors='pt',
        )

        # labels
        label_features = [{'input_ids': f['labels']} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors='pt',
        )
        labels = labels_batch['input_ids'].masked_fill(labels_batch['attention_mask'].ne(1), -100)

        batch['labels'] = labels
        return batch
# ---------------------------------------------------------

def build_augment_pipeline(sampling_rate: int):
    """Builds the audio augmentation pipeline."""
    print(f"[Bilgi] Veri zenginleştirme (augmentation) pipeline'ı {sampling_rate}Hz için oluşturuluyor.", file=sys.stderr)
    return A.Compose([
        A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.4),
        A.TimeStretch(min_rate=0.85, max_rate=1.15, p=0.4, leave_length_unchanged=False),
        A.PitchShift(min_semitones=-3, max_semitones=3, p=0.4),
        A.TimeMask(min_band_part=0.05, max_band_part=0.2, p=0.4),
        A.BandStopFilter(p=0.4),
    ], p=0.8) # Pipeline'ı %80 ihtimalle uygula

def read_audio(file_path: str):
    """Load mono audio with soundfile."""
    speech_array, sampling_rate = sf.read(file_path)
    # Eğer stereo ise mono'ya indir (ortalama)
    if speech_array.ndim > 1:
        speech_array = np.mean(speech_array, axis=1)
    return speech_array, sampling_rate


def prepare_example(batch, processor: Wav2Vec2Processor, augmenter=None):
    """CSV satırını (file_path, transcript) -> (input_values, labels) çevirir."""
    try:
        file_path = batch["file_path"]
        text = batch["transcript"]

        speech_array, sampling_rate = read_audio(file_path)

        # Augmentation uygula (eğer varsa)
        if augmenter:
            speech_array = augmenter(samples=speech_array, sample_rate=sampling_rate)

        target_sr = processor.feature_extractor.sampling_rate
        if sampling_rate != target_sr:
            # Resample audio to target_sr
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sampling_rate,
                new_freq=target_sr
            )
            speech_array = resampler(torch.tensor(speech_array, dtype=torch.float)).numpy()
            sampling_rate = target_sr # Update sampling_rate after resampling
            # print(f"[Bilgi] {file_path} için sr {sampling_rate} -> {target_sr} olarak resample edildi.", file=sys.stderr)

        batch["input_values"] = processor(
            speech_array, sampling_rate=sampling_rate
        ).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        batch["labels"] = processor.tokenizer(text).input_ids

        return batch
    except Exception as e:
        import traceback
        print(f"[Hata] {batch.get('file_path', '?')} işlenirken sorun: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr) # Tam traceback'i yazdır
        # Trainer collator'u patlamasın diye boş bırakma; yerine kısa dummy değerler
        batch["input_values"] = [0.0]
        batch["labels"] = [processor.tokenizer.pad_token_id]
        batch["input_length"] = 1
        return batch


def load_dataset_from_csv(csv_path: str, processor: Wav2Vec2Processor, with_augmentation: bool = False, is_hard_cases: bool = False) -> Dataset:
    """CSV -> HF Dataset (haritalama ve filtreleme dahil)"""
    df = pd.read_csv(csv_path)
    
    if is_hard_cases:
        df = df.rename(columns={"ground_truth": "transcript"})

    needed_cols = {"file_path", "transcript"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV'de eksik kolon(lar): {missing}. "
                         f"Gerekli kolonlar: {needed_cols}")

    ds = Dataset.from_pandas(df, preserve_index=False)

    augmenter = None
    if with_augmentation:
        sampling_rate = processor.feature_extractor.sampling_rate
        augmenter = build_augment_pipeline(sampling_rate)

    # partial ile processor ve augmenter'ı fonksiyona bağla
    prepare_fn = partial(prepare_example, processor=processor, augmenter=augmenter)

    ds = ds.map(prepare_fn, num_proc=None) # num_proc=None for debugging, can be increased
    
    # Çok kısa kayıtları ve boşları at
    ds = ds.filter(lambda x: isinstance(x["input_values"], (list, np.ndarray)) and len(x["input_values"]) > 1000) # Min 1s audio
    return ds


def build_training_args_kwargs(
    output_dir: str,
    eval_dataset_exists: bool,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    gradient_accumulation_steps: int = 2,
    num_train_epochs: int = 15,
    fp16: bool = None,
    learning_rate: float = 1e-4,
    weight_decay: float = 5e-3,
    warmup_steps: int = 5,
    save_steps: int = 100,
    eval_steps: int = 100,
    logging_steps: int = 50,
    save_total_limit: int = 2,
):
    """
    TrainingArguments imzasını kontrol ederek sürüme uygun argümanları verir.
    - Yeni sürümler: evaluation_strategy
    - Çok eski sürümler: evaluate_during_training
    """
    if fp16 is None:
        fp16 = bool(torch.cuda.is_available())

    common_kwargs = dict(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        fp16=fp16,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        save_total_limit=save_total_limit,
    )

    ta_params = signature(TrainingArguments).parameters

    if "evaluation_strategy" in ta_params:
        common_kwargs["evaluation_strategy"] = "steps" if eval_dataset_exists else "no"
    elif "evaluate_during_training" in ta_params:
        common_kwargs["evaluate_during_training"] = bool(eval_dataset_exists)
    # Aksi halde hiçbir şey eklemeyiz; Trainer eval_dataset varsa bazı sürümlerde yine değerlendirir.

    return common_kwargs


def compute_metrics_builder(processor: Wav2Vec2Processor):
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        # label -100'leri pad_token_id ile doldurup decode edelim
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


def parse_args():
    ap = argparse.ArgumentParser(description="CTC tabanlı ASR eğitimi (Wav2Vec2)")
    ap.add_argument("--model_id", type=str, default="mpoyraz/wav2vec2-xls-r-300m-cv8-turkish",
                    help="Hugging Face model id (CTC tabanlı)")
    ap.add_argument("--train_csv", type=str, required=True,
                    help="Eğitim CSV yolu (file_path, transcript sütunları içermeli)")
    ap.add_argument("--eval_csv", type=str, default=None,
                    help="Opsiyonel: Değerlendirme CSV yolu")
    ap.add_argument("--output_dir", type=str, default="./asr_model",
                    help="Model çıktı klasörü")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--train_bs", type=int, default=2)
    ap.add_argument("--eval_bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-3)
    ap.add_argument("--warmup_steps", type=int, default=5)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--no_augment", action="store_true", help="Veri zenginleştirmeyi (augmentation) kapatır.")
    ap.add_argument("--log_hard_cases", action="store_true", help="Hatalı tahminleri bir CSV dosyasına kaydeder.")
    ap.add_argument("--train_on_hard_cases", action="store_true", help="Sadece hatalı tahminler üzerinde eğitim yapar.")
    return ap.parse_args()


def main():
    args = parse_args()

    # Sürüm görünürlüğü (tanı için faydalı)
    import transformers as _tf
    print(f"Transformers version: {_tf.__version__}", file=sys.stderr)

    print("\n--- Processor ve Model yükleniyor ---")
    processor = Wav2Vec2Processor.from_pretrained(args.model_id)
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True
    )
    # Feature encoder'ı dondurmak genellikle eğitimi stabilize eder
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()

    print("\n--- Dataset hazırlanıyor ---")
    if args.train_on_hard_cases:
        print("[Bilgi] Hatalı tahminler üzerinden eğitim yapılacak.")
        train_csv_path = os.path.join("data", "hard_cases.csv")
        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(f"'{train_csv_path}' bulunamadı. Lütfen önce --log_hard_cases ile bir eğitim çalıştırın.")
        train_dataset = load_dataset_from_csv(train_csv_path, processor, with_augmentation=not args.no_augment, is_hard_cases=True)
    else:
        train_dataset = load_dataset_from_csv(args.train_csv, processor, with_augmentation=not args.no_augment)

    eval_dataset = None
    if args.eval_csv and os.path.exists(args.eval_csv):
        eval_dataset = load_dataset_from_csv(args.eval_csv, processor, with_augmentation=False) # Eval setine augmentation uygulama

    print("\n--- TrainingArguments oluşturuluyor ---")
    ta_kwargs = build_training_args_kwargs(
        output_dir=args.output_dir,
        eval_dataset_exists=bool(eval_dataset),
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_total_limit=2,
    )
    training_args = TrainingArguments(**ta_kwargs)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    compute_metrics = compute_metrics_builder(processor)

    print("\n--- Trainer başlatılıyor ---")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,  # feature_extractor verilebilir
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )

    print("\n--- Model Eğitimi Başlıyor ---")
    trainer.train()

    print("\n--- Eğitim tamamlandı. Son model kaydediliyor. ---")
    save_model(trainer.model, processor, args.output_dir)
    print(f"Model başarıyla '{args.output_dir}' klasörüne kaydedildi.")

    # --- Hatalı Tahminleri Loglama ---
    if args.log_hard_cases and eval_dataset is not None:
        print("\n--- Hatalı Tahminler Loglanıyor ---")
        predictions = trainer.predict(eval_dataset)
        pred_ids = np.argmax(predictions.predictions, axis=-1)
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

        hard_cases = []
        for i in range(len(pred_str)):
            # Normalize strings for comparison
            if pred_str[i].strip().lower() != eval_dataset[i]['transcript'].strip().lower():
                hard_cases.append({
                    "file_path": eval_dataset[i]['file_path'],
                    "ground_truth": eval_dataset[i]['transcript'],
                    "prediction": pred_str[i]
                })

        if hard_cases:
            hard_cases_df = pd.DataFrame(hard_cases)
            hard_cases_csv_path = os.path.join("data", "hard_cases.csv")
            hard_cases_df.to_csv(hard_cases_csv_path, index=False, encoding="utf-8")
            print(f"{len(hard_cases)} hatalı tahmin '{hard_cases_csv_path}' dosyasına kaydedildi.")


    # --- Raporlama ---
    print("\n--- Rapor oluşturuluyor ---")

    # En son metrikleri al
    if eval_dataset is not None:
        print("Değerlendirme metrikleri hesaplanıyor...")
        eval_results = trainer.evaluate()
        wer = eval_results.get("eval_wer", "Hesaplanamadı")

        report_data = {
            "title": f"ASR Model Eğitim Raporu: {os.path.basename(args.output_dir)}",
            "metrics": {
                "Model ID": args.model_id,
                "Eğitim Veri Seti": args.train_csv,
                "Değerlendirme Veri Seti": args.eval_csv,
                "Epoch Sayısı": args.epochs,
                "Batch Size (Train)": args.train_bs,
                "Batch Size (Eval)": args.eval_bs,
                "Öğrenme Oranı": args.lr,
                "Veri Zenginleştirme": "Aktif" if not args.no_augment else "Pasif",
                "Son Değerlendirme WER": wer,
            },
            "notes": f"Model '{args.output_dir}' klasörüne kaydedildi."
        }
        generate_report(report_data)
    else:
        print("Değerlendirme veri seti bulunmadığı için rapor oluşturulmuyor.")



if __name__ == "__main__":
    main()
