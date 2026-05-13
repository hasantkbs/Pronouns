# -*- coding: utf-8 -*-
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
import csv

import config
from src.constants import RECORD_TYPE_WORD, DATASET_DIRS, METADATA_FILENAMES
from src.core.asr import ASRSystem
from src.core.synthesizer import WordSynthesizer
from src.core.nlu import NLU_System
from src.core.actions import run_action
from train_adapter import PersonalizedTrainer
from src.data.repository import UserDataRepository
from src.utils.utils import calculate_audio_quality, normalize_path_for_cross_platform

app = FastAPI(title="Pronouns AI API")

# Repository instance
repo = UserDataRepository()

# CORS ayarları - Farklı ağlardan erişim için
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global sistemler (Lazy loading için None)
asr_systems = {}  # user_id -> ASRSystem
nlu = NLU_System()


def _metadata_words_path(user_id: str) -> Path:
    user_path = Path(config.BASE_PATH) / user_id
    return user_path / METADATA_FILENAMES[RECORD_TYPE_WORD]


def _get_recorded_details_words(user_id: str) -> Dict[str, int]:
    """
    pandas'a ihtiyaç duymadan metadata_words.csv içinden kelime->adet çıkarır.
    """
    path = _metadata_words_path(user_id)
    if not path.exists():
        return {}

    details: Dict[str, int] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                w = (row.get("transcription") or "").strip()
                if not w:
                    continue
                details[w] = details.get(w, 0) + 1
    except Exception:
        return {}
    return details


def _append_metadata_words_row(user_id: str, row: Dict[str, Any]) -> None:
    """
    metadata_words.csv'ye satır ekler. Dosya yoksa header oluşturur.
    """
    path = _metadata_words_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file_path",
        "transcription",
        "repetition",
        "quality_score",
        "rms",
        "snr_db",
        "duration",
    ]

    file_exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def _dataset_words_dir() -> Path:
    return Path(DATASET_DIRS[RECORD_TYPE_WORD])


def _read_word_set_file(set_file: str) -> List[str]:
    """
    collect_data.py mantığına benzer şekilde kelime setini txt'den okur.
    Boş satırları atar.
    """
    words_dir = _dataset_words_dir()
    path = (words_dir / set_file).resolve()

    # Güvenlik: datasets/words_set dışına çıkmayı engelle
    if words_dir.resolve() not in path.parents and path != words_dir.resolve():
        raise ValueError("Geçersiz set_file yolu.")

    if not path.exists():
        raise FileNotFoundError(f"Set dosyası bulunamadı: {set_file}")

    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            v = line.strip()
            if v:
                lines.append(v)
    return lines


def _get_next_word_and_rep(user_id: str, set_file: str) -> Dict[str, Any]:
    """
    collect_data.py'deki resume/eksik kayıt mantığının API versiyonu:
    - Setteki kelimeleri sırayla gezer
    - metadata_words.csv'den o kelime için kaç kayıt var bakar
    - IDEAL_REPETITIONS tamamlanmamışsa o kelimeyi + sıradaki rep'i döndürür
    """
    words = _read_word_set_file(set_file)
    details = _get_recorded_details_words(user_id)
    ideal = int(config.IDEAL_REPETITIONS)

    for w in words:
        current = int(details.get(w, 0))
        if current < ideal:
            return {
                "done": False,
                "word": w,
                "rep": current + 1,
                "current_count": current,
                "ideal_repetitions": ideal,
                "set_file": set_file,
                "total_words": len(words),
            }

    return {
        "done": True,
        "word": None,
        "rep": None,
        "ideal_repetitions": ideal,
        "set_file": set_file,
        "total_words": len(words),
    }

def get_asr(user_id: str):
    if user_id not in asr_systems:
        from src.services.model_service import ModelService
        model_path = ModelService.find_personalized_model(user_id)
        asr_systems[user_id] = ASRSystem(model_name=model_path, user_id=user_id)
    return asr_systems[user_id]

@app.get("/words")
async def list_words(user_id: str = "FurkanV1"):
    """Kaydedilecek kelimeleri ve mevcut kayıt durumlarını döndürür."""
    # Varsayılan kelime setini oku
    word_file = Path("datasets/words_set/wordSet.txt")
    if not word_file.exists():
        return {"error": "Kelime seti bulunamadı."}
    
    with open(word_file, "r", encoding="utf-8") as f:
        all_words = [line.strip() for line in f if line.strip()]
    
    # Mevcut kayıt detaylarını al
    details = repo.get_recorded_details(user_id, RECORD_TYPE_WORD)
    
    result = []
    # Sadece kaydı eksik olan kelimeleri bulalım
    for word in all_words:
        count = details.get(word, 0)
        required = config.IDEAL_REPETITIONS
        
        if count < required:
            result.append({
                "word": word,
                "count": count,
                "required": required
            })
            
        if len(result) >= 50:
            break
    
    return {"words": result}

@app.post("/record")
async def upload_record(
    user_id: str = Form(...), 
    word: str = Form(...), 
    audio: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Mobil uygulamadan gelen ses kaydını saklar ve metadata günceller."""
    user_dir = repo.get_save_path(user_id, RECORD_TYPE_WORD) / word
    user_dir.mkdir(parents=True, exist_ok=True)
    
    # Kaçıncı tekrar olduğunu bul
    current_count = repo.get_recorded_count(user_id, RECORD_TYPE_WORD, word)
    rep = current_count + 1
    
    file_path = user_dir / f"rep{rep}.wav"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    
    # Kalite hesapla
    quality_info = calculate_audio_quality(str(file_path))
    
    # Metadata kaydet
    relative_path = normalize_path_for_cross_platform(str(file_path.absolute()), repo.get_user_path(user_id))
    
    metadata_entry = {
        "file_path": relative_path,
        "transcription": word,
        "repetition": rep,
        "quality_score": quality_info['quality_score'],
        "rms": quality_info['rms'],
        "snr_db": quality_info['snr_db'],
        "duration": quality_info['duration']
    }
    
    repo.save_metadata(user_id, RECORD_TYPE_WORD, [metadata_entry], append=True)
    
    # Self-learning: yeterli yeni kayıt varsa arka planda fine-tune başlat
    settings = _user_settings.get(user_id, dict(_DEFAULT_SETTINGS))
    if settings.get("self_learning", True):
        details = _get_recorded_details_words(user_id)
        total_samples = sum(details.values())
        ideal = int(config.IDEAL_REPETITIONS)
        # Her ideal_repetitions tamamlandığında otomatik eğitim tetikle
        if total_samples > 0 and total_samples % ideal == 0:
            trainer = PersonalizedTrainer(user_id=user_id)
            if background_tasks:
                background_tasks.add_task(trainer.run)
            else:
                import threading
                t = threading.Thread(target=trainer.run, daemon=True)
                t.start()

    return {
        "status": "success", 
        "word": word,
        "rep": rep,
        "quality": quality_info['quality_score'],
        "path": str(file_path),
        "metadata_row": metadata_entry
    }


@app.get("/collect/word-sets")
async def list_word_sets():
    """datasets/words_set içindeki set dosyalarını listeler."""
    d = _dataset_words_dir()
    if not d.exists():
        return {"sets": []}
    sets = sorted([p.name for p in d.glob("*.txt")])
    return {"sets": sets}


@app.get("/collect/next-word")
async def get_next_word(user_id: str, set_file: str = "wordSet.txt"):
    """
    Mobil tarafın 'collect_data.py' akışına benzer biçimde sıradaki kelimeyi alması için.
    """
    try:
        return _get_next_word_and_rep(user_id=user_id, set_file=set_file)
    except Exception as e:
        return {"error": str(e)}


@app.get("/collect/progress")
async def get_progress(user_id: str, set_file: str = "wordSet.txt"):
    """
    Set bazında ilerleme özeti.
    """
    try:
        words = _read_word_set_file(set_file)
        details = _get_recorded_details_words(user_id)
        ideal = int(config.IDEAL_REPETITIONS)
        completed = sum(1 for w in words if int(details.get(w, 0)) >= ideal)
        return {
            "set_file": set_file,
            "total_words": len(words),
            "completed_words": completed,
            "ideal_repetitions": ideal,
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/train")
async def start_training(user_id: str, background_tasks: BackgroundTasks):
    """Eğitimi arka planda başlatır."""
    trainer = PersonalizedTrainer(user_id=user_id)
    background_tasks.add_task(trainer.run)
    return {"status": "training_started", "user_id": user_id}

@app.post("/translate")
async def translate_speech(user_id: str = Form(...), audio: UploadFile = File(...)):
    """
    1. Sesi ASR ile metne çevirir.
    2. Metni NLU ile anlar.
    3. Yanıtı kullanıcının sesiyle sentezler.
    """
    # Geçici dosyaya kaydet
    temp_id = str(uuid.uuid4())
    temp_input = f"temp_in_{temp_id}.wav"
    with open(temp_input, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    
    # ASR
    asr = get_asr(user_id)
    text, confidence = asr.transcribe(temp_input)
    os.remove(temp_input)
    
    if not text:
        return {"error": "Konuşma anlaşılamadı."}
    
    # NLU & Action
    intent, entities = nlu.process_text(text)
    action_response = run_action(intent, entities)
    
    # Synthesis
    synth = WordSynthesizer(user_id=user_id)
    out_path, missing = synth.synthesize_sentence(action_response)
    
    return {
        "recognized_text": text,
        "intent": intent,
        "response_text": action_response,
        "audio_url": f"/download/{os.path.basename(out_path)}" if out_path else None,
        "missing_words": missing
    }

@app.get("/download/{filename}")
async def download_audio(filename: str):
    return FileResponse(filename, media_type="audio/wav")

@app.get("/apk")
async def download_apk():
    """APK dosyasını indirir."""
    apk_path = Path(config.BASE_PATH).parent / "apk" / "pronouns.apk"
    if not apk_path.exists():
        return {"error": "APK henüz mevcut değil. Lütfen önce derleyin."}
    return FileResponse(apk_path, media_type="application/vnd.android.package-archive", filename="pronouns.apk")


# ─── Ayarlar ─────────────────────────────────────────────────────────────────

# Basit in-memory ayar deposu (sunucu yeniden başlarsa sıfırlanır).
_user_settings: Dict[str, Dict[str, Any]] = {}

_DEFAULT_SETTINGS: Dict[str, Any] = {
    "model": "Furkan",
    "algorithm": "lora",
    "self_learning": True,
    "learning_rate": float(config.FINETUNE_LEARNING_RATE),
    "epochs": int(config.NUM_FINETUNE_EPOCHS),
    "batch_size": int(config.FINETUNE_BATCH_SIZE),
}


@app.get("/settings")
async def get_settings(user_id: str):
    """Kullanıcının model & fine-tune ayarlarını döner."""
    settings = _user_settings.get(user_id, dict(_DEFAULT_SETTINGS))
    return {"user_id": user_id, **settings}


@app.post("/settings")
async def save_settings(payload: Dict[str, Any]):
    """Kullanıcının model & fine-tune ayarlarını günceller."""
    user_id = payload.get("user_id", "")
    if not user_id:
        return {"error": "user_id zorunlu"}

    current = _user_settings.get(user_id, dict(_DEFAULT_SETTINGS))
    allowed = {"model", "algorithm", "self_learning", "learning_rate", "epochs", "batch_size"}
    for key in allowed:
        if key in payload:
            current[key] = payload[key]
    _user_settings[user_id] = current
    return {"status": "saved", "user_id": user_id, **current}


# ─── Model Bilgisi ───────────────────────────────────────────────────────────

@app.get("/model/info")
async def model_info(user_id: str):
    """Kullanıcının mevcut model bilgisini döner."""
    from src.services.model_service import ModelService
    model_path = ModelService.find_personalized_model(user_id)

    # Kaç ses kaydı var?
    meta_path = _metadata_words_path(user_id)
    sample_count = 0
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                sample_count = max(0, sum(1 for _ in f) - 1)  # header hariç
        except Exception:
            pass

    return {
        "user_id": user_id,
        "model": model_path,
        "sample_count": sample_count,
        "wer": None,
    }


# ─── Fine-Tune (İnce Ayar) ───────────────────────────────────────────────────

@app.post("/fine-tune")
async def fine_tune(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    """
    Kullanıcı tarafından tetiklenen fine-tune işlemi.
    Arka planda PersonalizedTrainer çalıştırır.
    Self-learning aktifse yeni kayıtlar otomatik modele dahil edilir.
    """
    user_id = payload.get("user_id", "")
    if not user_id:
        return {"error": "user_id zorunlu"}

    # Ayarları güncelle (varsa)
    settings = _user_settings.get(user_id, dict(_DEFAULT_SETTINGS))
    for key in ("algorithm", "learning_rate", "epochs", "batch_size"):
        if key in payload:
            settings[key] = payload[key]
    _user_settings[user_id] = settings

    trainer = PersonalizedTrainer(user_id=user_id)
    background_tasks.add_task(trainer.run)

    return {
        "status": "fine_tune_started",
        "user_id": user_id,
        "algorithm": settings.get("algorithm"),
        "epochs": settings.get("epochs"),
        "learning_rate": settings.get("learning_rate"),
        "batch_size": settings.get("batch_size"),
        "message": "Eğitim arka planda başlatıldı. Tamamlandığında model otomatik güncellenir.",
    }


# ─── Self-Learning Tetikleyici ────────────────────────────────────────────────

@app.post("/self-learn")
async def self_learn(user_id: str, background_tasks: BackgroundTasks):
    """
    Yeni ses kayıtları algılandığında (self-learning aktifse)
    otomatik fine-tune tetikler.
    """
    settings = _user_settings.get(user_id, dict(_DEFAULT_SETTINGS))
    if not settings.get("self_learning", True):
        return {"status": "skipped", "reason": "self_learning devre dışı"}

    trainer = PersonalizedTrainer(user_id=user_id)
    background_tasks.add_task(trainer.run)
    return {"status": "self_learn_started", "user_id": user_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
