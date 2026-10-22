# -*- coding: utf-8 -*-
"""
Cümle kayıtlarında ASR doğruluğunu ölçer (alt proje B1).

Her cümle kaydı için iki çıktı karşılaştırılır:
  ham         : kullanıcının modeli, fuzzy düzeltme kapalı (canlı /translate ile aynı)
  düzeltilmiş : ham + SentenceCorrector (Furkan'ın gördüğü cümle)

Kullanım:
    python -m src.cli.evaluate_sentences --user FurkanV1 [--model <klasör adı>] [--device cpu]
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import jiwer
import soundfile as sf

import config
from src.utils.utils import normalize_turkish_text

TOTAL = "toplam"


def normalize(text: Optional[str]) -> str:
    """Büyük/küçük harf (İ/I) ve noktalama farkı hata sayılmasın."""
    return normalize_turkish_text(text or "")


def score(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    """Normalize edilmiş metin listeleri için WER, CER ve tam isabet oranı."""
    return {
        "wer": float(jiwer.wer(refs, hyps)),
        "cer": float(jiwer.cer(refs, hyps)),
        "exact": sum(r == h for r, h in zip(refs, hyps)) / len(refs),
    }


def load_rows(user_dir: Path) -> List[Dict[str, str]]:
    path = Path(user_dir) / "metadata.csv"
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            file_path = row["file_path"]
            audio = Path(file_path)
            if not audio.is_absolute():
                audio = Path(user_dir) / audio
            rows.append({
                "audio_path": str(audio),
                "file_path": file_path,
                "expected": row["transcription"],
                "set_file": row.get("set_file") or "bilinmiyor",
            })
    return rows


def evaluate(
    rows: List[Dict[str, str]],
    transcribe: Callable[[str], Optional[str]],
    correct: Callable[[str], str],
) -> Dict:
    results, skipped = [], []
    for row in rows:
        try:
            sf.info(row["audio_path"])
        except Exception as e:
            skipped.append({"file_path": row["file_path"], "reason": str(e)})
            continue
        raw = transcribe(row["audio_path"]) or ""
        corrected = correct(raw) or ""
        expected = normalize(row["expected"])
        results.append({
            "file_path": row["file_path"],
            "set_file": row["set_file"],
            "expected": row["expected"],
            "raw": raw,
            "corrected": corrected,
            "raw_wer": float(jiwer.wer(expected, normalize(raw))),
            "corrected_wer": float(jiwer.wer(expected, normalize(corrected))),
        })

    groups: Dict[str, List[Dict]] = {}
    for r in results:
        groups.setdefault(r["set_file"], []).append(r)
    if results:
        groups[TOTAL] = results

    summary = {}
    for name, items in groups.items():
        refs = [normalize(r["expected"]) for r in items]
        summary[name] = {
            "count": len(items),
            "raw": score(refs, [normalize(r["raw"]) for r in items]),
            "corrected": score(refs, [normalize(r["corrected"]) for r in items]),
        }
    return {"summary": summary, "rows": results, "skipped": skipped}


def worst(rows: List[Dict], n: int = 10) -> List[Dict]:
    """Ham çıktıda hatası en yüksek n kayıt (B1 modelin kendisini ölçer)."""
    return sorted(rows, key=lambda r: (r["raw_wer"], r["corrected_wer"]), reverse=True)[:n]


def write_report(
    report: Dict, reports_dir: Path, now: Optional[datetime] = None, prefix: str = "sentence_eval"
) -> Path:
    now = now or datetime.now()
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"{prefix}_{now.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def print_summary(report: Dict) -> None:
    print(f"\n{'Grup':<28}{'Adet':>6}  {'WER ham/düz':>14}  {'CER ham/düz':>14}  {'Tam ham/düz':>14}")
    for name, s in report["summary"].items():
        raw, cor = s["raw"], s["corrected"]
        print(
            f"{name:<28}{s['count']:>6}  "
            f"{raw['wer']:>6.1%}/{cor['wer']:<7.1%}  "
            f"{raw['cer']:>6.1%}/{cor['cer']:<7.1%}  "
            f"{raw['exact']:>6.1%}/{cor['exact']:<7.1%}"
        )
    print("\nEn kötü örnekler (ham WER'e göre):")
    for r in worst(report["rows"]):
        print(f"- [{r['raw_wer']:.0%}] beklenen: {r['expected']}")
        print(f"        ham: {r['raw']}")
        print(f"  düzeltilmiş: {r['corrected']}")
    if report["skipped"]:
        print(f"\nAtlanan kayıt: {len(report['skipped'])}")


def model_key(user_id: str, settings_model: Optional[str], model_name: Optional[str] = None) -> str:
    """Ölçülecek model klasörünün adı; --model verilmezse api.get_asr ile aynı seçim."""
    if model_name:
        return model_name
    return f"{user_id}_whisper" if settings_model == "whisper" else user_id


def _load_asr(user_id: str, model_name: Optional[str] = None):
    # Ağır importlar (transformers, peft) yalnızca gerçek ölçümde yüklenir
    from src.core.asr import ASRSystem
    from src.services.model_service import ModelService
    from src.services.settings_service import SettingsService

    settings = SettingsService().load_settings(user_id)
    key = model_key(user_id, settings.get("model"), model_name)
    model_path = ModelService.find_personalized_model(key)
    if model_path is None:
        raise SystemExit(f"Model bulunamadı: {key}")
    # user_id sözlüğü (metadata_words.csv) seçer; model klasöründen bağımsız
    return ASRSystem(model_name=model_path, user_id=user_id)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Cümle kayıtlarında ASR doğruluğunu ölçer.")
    parser.add_argument("--user", default="FurkanV1")
    parser.add_argument("--model", default=None,
                        help="data/models/personalized_models altındaki klasör adı (varsayılan: canlı model)")
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto",
                        help="cpu: API GPU'yu kullanırken bellek yetmezse")
    args = parser.parse_args(argv)

    if args.device == "cpu":
        # CUDA henüz başlatılmadı; torch GPU'yu görmez
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    rows = load_rows(Path(config.BASE_PATH) / args.user)
    if not rows:
        print("Cümle kaydı yok.")
        return 1

    from src.core.sentence_corrector import SentenceCorrector

    asr = _load_asr(args.user, args.model)
    corrector = SentenceCorrector(asr.user_vocabulary, Path(config.LEXICON_PATH))
    report = evaluate(
        rows,
        transcribe=lambda path: asr.transcribe(path, fuzzy_correct=False)[0],
        correct=corrector.correct,
    )
    report["user_id"] = args.user
    report["model_type"] = asr.model_type
    report["model"] = args.model

    print_summary(report)
    out = write_report(report, Path(config.REPORTS_DIR))
    print(f"\nRapor: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
