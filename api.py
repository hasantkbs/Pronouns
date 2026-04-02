# -*- coding: utf-8 -*-
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
import os
import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import config
from src.core.asr import ASRSystem
from src.core.synthesizer import WordSynthesizer
from src.core.nlu import NLU_System
from src.core.actions import run_action
from train_adapter import PersonalizedTrainer
from src.data.repository import UserDataRepository
from src.utils.utils import calculate_audio_quality, normalize_path_for_cross_platform
from src.constants import RECORD_TYPE_WORD

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

def get_asr(user_id: str):
    if user_id not in asr_systems:
        from src.services.model_service import ModelService
        # FurkanV1 zorlaması yapılabilir veya dinamik bırakılabilir
        # Kullanıcı FurkanV1 dediği için varsayılan olarak onu arayacaktır.
        model_path = ModelService.find_personalized_model(user_id)
        asr_systems[user_id] = ASRSystem(model_name=model_path)
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
            
        # Performans için liste çok uzamasın (örn: 50 kelime yeterli)
        if len(result) >= 50:
            break
    
    return {"words": result}

@app.post("/record")
async def upload_record(
    user_id: str = Form(...), 
    word: str = Form(...), 
    audio: UploadFile = File(...)
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
    
    return {
        "status": "success", 
        "word": word,
        "rep": rep,
        "quality": quality_info['quality_score'],
        "path": str(file_path)
    }

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
