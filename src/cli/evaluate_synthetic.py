# -*- coding: utf-8 -*-
"""
Sentetik cümlelerde ve aynı kelimelerin tekil kayıtlarında ASR doğruluğunu
karşılaştırır (alt proje B2). Cümle WER'i kelime WER'inden en az 5 puan
kötüyse karar "egitime_gec", değilse "dur".

Kullanım:
    python -m src.cli.evaluate_synthetic --user FurkanV1 [--model <klasör adı>] [--full-words] [--device cpu]
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import config
from src.cli import evaluate_sentences as ev
from src.training import sentence_synth as ss

EVAL_SENTENCES = Path("datasets") / "sentence_sets" / "sentetik_olcum.txt"
WORDS = "kelime"
SENTENCES = "sentetik_cumle"
THRESHOLD = 0.05


def decide(word_wer: float, sentence_wer: float, threshold: float = THRESHOLD) -> str:
    # Yuvarlama: 0.20 - 0.15 kayan noktada 0.04999... çıkabilir
    return "egitime_gec" if round(sentence_wer - word_wer, 6) >= threshold else "dur"


def sentence_rows(manifest_rows: List[Dict[str, str]], user_dir: Path) -> List[Dict[str, str]]:
    return [
        {
            "audio_path": str(Path(user_dir) / r["file_path"]),
            "file_path": r["file_path"],
            "expected": r["transcription"],
            "set_file": SENTENCES,
        }
        for r in manifest_rows
    ]


def word_rows(takes_by_word: Dict[str, List[str]], words: Optional[set] = None) -> List[Dict[str, str]]:
    """eval.csv'nin tekil kelime kayıtları; words verilirse yalnızca o kelimeler."""
    return [
        {"audio_path": path, "file_path": path, "expected": word, "set_file": WORDS}
        for word, paths in sorted(takes_by_word.items())
        if words is None or word in words
        for path in paths
    ]


def build_report(result: Dict, skipped: List[str], model: Optional[str]) -> Dict:
    word_wer = result["summary"][WORDS]["raw"]["wer"]
    sentence_wer = result["summary"][SENTENCES]["raw"]["wer"]
    return {
        "model": model,
        "word_wer": word_wer,
        "sentence_wer": sentence_wer,
        "difference": sentence_wer - word_wer,
        "decision": decide(word_wer, sentence_wer),
        "skipped_sentences": skipped,
        **result,
    }


def print_decision(report: Dict) -> None:
    print(
        f"\nKelime WER: {report['word_wer']:.1%}  Cümle WER: {report['sentence_wer']:.1%}  "
        f"Fark: {report['difference']:+.1%}"
    )
    print(f"Karar: {report['decision']}")
    if report["skipped_sentences"]:
        print(f"Atlanan cümle (kaydı olmayan kelime): {len(report['skipped_sentences'])}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sentetik cümlelerde ASR doğruluğunu ölçer.")
    parser.add_argument("--user", default="FurkanV1")
    parser.add_argument("--model", default=None,
                        help="data/models/personalized_models altındaki klasör adı (varsayılan: canlı model)")
    parser.add_argument("--full-words", action="store_true",
                        help="kelime kıyası için eval.csv'nin tamamı")
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    args = parser.parse_args(argv)

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    user_dir = Path(config.BASE_PATH) / args.user
    if not (user_dir / "eval.csv").exists():
        print(f"eval.csv yok: {user_dir}")
        return 1

    manifest, skipped = ss.ensure_eval_set(user_dir, EVAL_SENTENCES)
    if not manifest:
        print("Üretilebilen sentetik cümle yok.")
        return 1
    takes = ss.load_takes(user_dir / "eval.csv", user_dir)
    words = None if args.full_words else {
        w for r in manifest for w in ss.sentence_words(r["transcription"])
    }
    rows = word_rows(takes, words) + sentence_rows(manifest, user_dir)

    from src.core.sentence_corrector import SentenceCorrector

    asr = ev._load_asr(args.user, args.model)
    corrector = SentenceCorrector(asr.user_vocabulary, Path(config.LEXICON_PATH))
    result = ev.evaluate(
        rows,
        transcribe=lambda path: asr.transcribe(path, fuzzy_correct=False)[0],
        correct=corrector.correct,
    )
    report = build_report(result, skipped, args.model)
    report["user_id"] = args.user
    report["model_type"] = asr.model_type
    report["full_words"] = args.full_words

    ev.print_summary(report)
    print_decision(report)
    out = ev.write_report(report, Path(config.REPORTS_DIR), prefix="synthetic_eval")
    print(f"\nRapor: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
