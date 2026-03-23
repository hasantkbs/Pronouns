# -*- coding: utf-8 -*-
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
import os
import uuid
import shutil
from pathlib import Path
import config
from src.core.asr import ASRSystem
from src.core.synthesizer import WordSynthesizer
from src.core.nlu import NLU_System
from src.core.actions import run_action
from train_adapter import PersonalizedTrainer

app = FastAPI(title="Pronouns AI API")

# Global sistemler (Lazy loading için None)
asr_systems = {}  # user_id -> ASRSystem
nlu = NLU_System()

def get_asr(user_id: str):
    if user_id not in asr_systems:
        from src.services.model_service import ModelService
        model_path = ModelService.find_personalized_model(user_id)
        asr_systems[user_id] = ASRSystem(model_name=model_path)
    return asr_systems[user_id]

@app.post("/record")
async def upload_record(user_id: str = Form(...), word: str = Form(...), rep: int = Form(...), audio: UploadFile = File(...)):
    """Mobil uygulamadan gelen ses kaydını saklar."""
    user_dir = Path(config.BASE_PATH) / user_id / "words" / word
    user_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = user_dir / f"rep{rep}.wav"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    
    # Metadata güncelleme (Opsiyonel: Burada repository kullanılabilir)
    return {"status": "success", "path": str(file_path)}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
