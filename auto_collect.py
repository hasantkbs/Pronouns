# -*- coding: utf-8 -*-
"""
Otonom Kelime Kayıt Sistemi
============================
Kullanıcı başlatıp bırakabilir; sistem kelimeleri sırayla okur, sesi
otomatik algılar, üç kademeli doğrulama uygular ve sadece geçerli
kayıtları kaydeder.

Üç kademeli doğrulama:
  1. RMS / süre / SNR filtresi  — boş ve gürültülü sesleri reddeder
  2. Konuşma oranı filtresi     — gerçek ses içermeyen kaydı reddeder
  3. ASR doğrulaması            — (config.AUTO_ASR_VERIFY=True ise)
                                  yanlış kelimeyi reddeder

Kullanım:
  python auto_collect.py <kullanici_id> <kelime_listesi.txt> [secenekler]

Örnekler:
  python auto_collect.py FurkanV1 datasets/words_set/temel_kelimeler.txt
  python auto_collect.py FurkanV1 datasets/words_set/temel_kelimeler.txt --reps 5
  python auto_collect.py FurkanV1 datasets/words_set/temel_kelimeler.txt --asr-verify
  python auto_collect.py FurkanV1 datasets/words_set/temel_kelimeler.txt --resume
"""

import os
import sys
import time
import argparse
import signal
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.utils.utils import record_audio_auto, validate_recording, normalize_path_for_cross_platform
from src.services.reporting_service import ReportingService


# ---------------------------------------------------------------------------
# Yardımcı gösterim fonksiyonları
# ---------------------------------------------------------------------------

def _clear_line():
    print("\r" + " " * 80 + "\r", end="", flush=True)


def _display_word(word: str, rep_num: int, total_reps: int, attempt: int, max_attempts: int):
    """Hedef kelimeyi terminal'de büyük ve net gösterir."""
    width = getattr(config, "AUTO_DISPLAY_WIDTH", 60)
    border = "=" * width
    word_centered = word.upper().center(width)
    status = f"Tekrar {rep_num}/{total_reps}  |  Deneme {attempt}/{max_attempts}"
    status_centered = status.center(width)

    print(f"\n{border}")
    print(word_centered)
    print("-" * width)
    print(status_centered)
    print(border)


def _display_result(valid: bool, reason: str, quality: dict, elapsed: float):
    if valid:
        qs = quality.get("quality_score", 0)
        snr = quality.get("snr_db", 0)
        dur = quality.get("duration", 0)
        print(f"  KABUL  |  Kalite: {qs:.0f}/100  SNR: {snr:.1f}dB  Sure: {dur:.2f}s  ({elapsed:.1f}s)")
    else:
        print(f"  RED    |  {reason}")


def _display_skip(word: str, reason: str):
    print(f"  ATLANDI: '{word}'  —  {reason}")


def _countdown(seconds: float, message: str = "Sonraki kelime"):
    """Geri sayım gösterir; kullanıcı 's' tuşuna basarak atlayabilir."""
    steps = int(seconds * 4)  # 250 ms adım
    for i in range(steps, 0, -1):
        remaining = i / 4
        print(f"\r  {message}: {remaining:.1f}s   ", end="", flush=True)
        time.sleep(0.25)
    _clear_line()


def _listening_indicator(running_flag: list):
    """Arka plan thread'i: Kayıt devam ederken animasyon gösterir."""
    import threading
    symbols = ["|", "/", "-", "\\"]
    i = 0
    while running_flag[0]:
        print(f"\r  Dinleniyor {symbols[i % 4]}  ", end="", flush=True)
        i += 1
        time.sleep(0.15)
    _clear_line()


# ---------------------------------------------------------------------------
# İlerleme takibi
# ---------------------------------------------------------------------------

def _load_progress(metadata_path: Path) -> dict[str, int]:
    """CSV'den her kelime için mevcut kayıt sayısını döndürür."""
    if not metadata_path.exists():
        return {}
    try:
        df = pd.read_csv(metadata_path, encoding="utf-8")
        if "transcription" not in df.columns:
            return {}
        return df.groupby("transcription").size().to_dict()
    except Exception:
        return {}


def _append_to_metadata(metadata_path: Path, row: dict):
    """Tek bir satırı metadata CSV'ye güvenli biçimde ekler."""
    df_new = pd.DataFrame([row])
    if metadata_path.exists() and metadata_path.stat().st_size > 0:
        try:
            df_existing = pd.read_csv(metadata_path, encoding="utf-8")
            df_out = pd.concat([df_existing, df_new], ignore_index=True)
        except pd.errors.EmptyDataError:
            df_out = df_new
    else:
        df_out = df_new
    df_out.drop_duplicates(
        subset=["file_path", "transcription", "repetition"], inplace=True
    )
    df_out.to_csv(metadata_path, index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# Ana toplayıcı sınıfı
# ---------------------------------------------------------------------------

class AutoCollector:
    """
    Otonom kelime kayıt oturumu.

    Parametreler
    ------------
    user_id     : Kullanıcı kimliği
    word_list   : Kaydedilecek kelimeler
    reps        : Her kelime için hedef tekrar sayısı
    asr_verify  : ASR doğrulaması kullanılsın mı
    resume      : Mevcut ilerlemeyi yükle ve kaldığı yerden devam et
    """

    def __init__(
        self,
        user_id: str,
        word_list: list[str],
        reps: int = None,
        asr_verify: bool = None,
        resume: bool = True,
    ):
        self.user_id = user_id
        self.word_list = word_list
        self.target_reps = reps or config.IDEAL_REPETITIONS
        self.asr_verify = asr_verify if asr_verify is not None else config.AUTO_ASR_VERIFY
        self.resume = resume

        self.words_dir = Path(config.BASE_PATH) / user_id / "words"
        self.metadata_path = Path(config.BASE_PATH) / user_id / "metadata_words.csv"
        self.words_dir.mkdir(parents=True, exist_ok=True)

        self.asr_system = None
        if self.asr_verify:
            self._init_asr()

        # Oturum istatistikleri
        self.stats = {
            "session_start": datetime.now().isoformat(),
            "words_completed": 0,
            "words_skipped": 0,
            "total_attempts": 0,
            "accepted": 0,
            "rejected_quality": 0,
            "rejected_silence": 0,
            "rejected_asr": 0,
            "quality_scores": [],
        }

        # Ctrl+C ile temiz çıkış
        self._interrupted = False
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, sig, frame):
        print("\n\nOturum kullanici tarafindan durduruldu.")
        self._interrupted = True

    def _init_asr(self):
        """ASR doğrulaması için modeli yükler."""
        from src.core.asr import ASRSystem
        from src.services.model_service import ModelService
        model_path = ModelService.find_personalized_model(self.user_id)
        print("ASR modeli yukleniyor (dogrulama icin)...")
        self.asr_system = ASRSystem(model_name=model_path)

    # ------------------------------------------------------------------
    # Ana oturum döngüsü
    # ------------------------------------------------------------------

    def run(self):
        progress = _load_progress(self.metadata_path) if self.resume else {}

        # Tamamlanmamış kelimeleri filtrele
        pending = [w for w in self.word_list if progress.get(w, 0) < self.target_reps]

        total_words = len(self.word_list)
        already_done = total_words - len(pending)

        print("\n" + "=" * 60)
        print("  OTONOM KELIME KAYIT OTURUMU")
        print("=" * 60)
        print(f"  Kullanici     : {self.user_id}")
        print(f"  Toplam kelime : {total_words}")
        print(f"  Tamamlanmis   : {already_done}")
        print(f"  Kaydedilecek  : {len(pending)}")
        print(f"  Tekrar/kelime : {self.target_reps}")
        print(f"  ASR dogrulama : {'ACIK' if self.asr_verify else 'KAPALI'}")
        print(f"  Zaman asimi   : {config.AUTO_WORD_TIMEOUT_SEC}s/kelime")
        print("=" * 60)
        print("  Durdurmak icin Ctrl+C'ye basin.")
        print("=" * 60)
        time.sleep(2)

        for word_idx, word in enumerate(pending):
            if self._interrupted:
                break

            current_reps = progress.get(word, 0)
            needed_reps = self.target_reps - current_reps

            word_completed = self._record_word(
                word=word,
                start_rep=current_reps + 1,
                needed_reps=needed_reps,
                word_idx=word_idx,
                total_pending=len(pending),
            )

            if word_completed:
                self.stats["words_completed"] += 1
                progress[word] = self.target_reps
            else:
                self.stats["words_skipped"] += 1

            if not self._interrupted and word_idx < len(pending) - 1:
                _countdown(config.AUTO_INTER_WORD_PAUSE_SEC, "Sonraki kelime")

        self._print_summary()

        if config.AUTO_SAVE_REPORT:
            self._save_report()

    # ------------------------------------------------------------------
    # Tek kelime kayıt döngüsü
    # ------------------------------------------------------------------

    def _record_word(
        self,
        word: str,
        start_rep: int,
        needed_reps: int,
        word_idx: int,
        total_pending: int,
    ) -> bool:
        """
        Bir kelime için gerekli tekrar sayısını kayıt eder.

        Returns True eğer tüm tekrarlar başarıyla tamamlandıysa.
        """
        max_retries = config.AUTO_MAX_RETRIES
        word_dir = self.words_dir / word
        word_dir.mkdir(parents=True, exist_ok=True)

        accepted_this_word = 0

        for rep_num in range(start_rep, start_rep + needed_reps):
            if self._interrupted:
                return False

            attempt = 0
            rep_accepted = False

            while attempt < max_retries and not self._interrupted:
                attempt += 1
                self.stats["total_attempts"] += 1

                _display_word(
                    word=word,
                    rep_num=rep_num,
                    total_reps=self.target_reps,
                    attempt=attempt,
                    max_attempts=max_retries,
                )
                print(f"  Kelime: {word_idx + 1}/{total_pending}")

                tmp_path = str(word_dir / f"_tmp_rep{rep_num}.wav")
                t0 = time.time()

                # Arka planda animasyon göster
                import threading
                running = [True]
                anim = threading.Thread(target=_listening_indicator, args=(running,), daemon=True)
                anim.start()

                recorded = record_audio_auto(file_path=tmp_path)

                running[0] = False
                anim.join(timeout=1)

                elapsed = time.time() - t0

                if recorded is None:
                    self.stats["rejected_silence"] += 1
                    _display_result(False, f"Ses algilanamadi ({elapsed:.1f}s beklendi)", {}, elapsed)
                    # Zaman aşımı = kelimenin tamamını atla
                    if elapsed >= config.AUTO_SPEECH_WAIT_SEC - 0.5:
                        _display_skip(word, "zaman asimi")
                        return False
                    continue

                # Üç kademeli doğrulama
                validation = validate_recording(
                    file_path=recorded,
                    expected_word=word,
                    asr_system=self.asr_system if self.asr_verify else None,
                )

                if not validation["valid"]:
                    reason = validation["reason"]
                    q = validation["quality"]
                    _display_result(False, reason, q, elapsed)

                    # Red nedenine göre istatistik güncelle
                    if "sessiz" in reason.lower() or "kisa" in reason.lower() or "oran" in reason.lower():
                        self.stats["rejected_silence"] += 1
                    elif "asr" in reason.lower():
                        self.stats["rejected_asr"] += 1
                    else:
                        self.stats["rejected_quality"] += 1

                    # Geçersiz geçici dosyayı sil
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    continue

                # --- Kayıt geçerli: kalıcı dosyaya taşı ---
                final_path = word_dir / f"rep{rep_num}.wav"
                os.replace(tmp_path, str(final_path))

                q = validation["quality"]
                _display_result(True, "", q, elapsed)
                self.stats["accepted"] += 1
                self.stats["quality_scores"].append(q.get("quality_score", 0))

                # Metadata'ya ekle
                rel_path = normalize_path_for_cross_platform(
                    str(final_path.absolute()),
                    Path(config.BASE_PATH) / self.user_id,
                )
                _append_to_metadata(
                    self.metadata_path,
                    {
                        "file_path": rel_path,
                        "transcription": word,
                        "repetition": rep_num,
                        "quality_score": q.get("quality_score", 0),
                        "rms": q.get("rms", 0),
                        "snr_db": q.get("snr_db", 0),
                        "duration": q.get("duration", 0),
                    },
                )

                rep_accepted = True
                accepted_this_word += 1
                break  # Bu tekrar tamamlandı; bir sonraki tekrara geç

            if not rep_accepted:
                # Tüm denemeler tükendi; kelimeyi atla
                _display_skip(word, f"{max_retries} denemede gecerli kayit alinamadi")
                return False

        return accepted_this_word == needed_reps

    # ------------------------------------------------------------------
    # Özet ve rapor
    # ------------------------------------------------------------------

    def _print_summary(self):
        s = self.stats
        total = s["total_attempts"]
        accepted = s["accepted"]
        rejection_rate = (total - accepted) / max(total, 1) * 100

        avg_q = (
            sum(s["quality_scores"]) / len(s["quality_scores"])
            if s["quality_scores"]
            else 0
        )

        print("\n" + "=" * 60)
        print("  OTURUM OZETI")
        print("=" * 60)
        print(f"  Tamamlanan kelime : {s['words_completed']}")
        print(f"  Atlanan kelime    : {s['words_skipped']}")
        print(f"  Toplam deneme     : {total}")
        print(f"  Kabul edilen      : {accepted}")
        print(f"  Red orani         : {rejection_rate:.1f}%")
        print(f"    - Kalite/Gurultu: {s['rejected_quality']}")
        print(f"    - Bos/Sessiz    : {s['rejected_silence']}")
        print(f"    - ASR uyumsuz   : {s['rejected_asr']}")
        print(f"  Ort. kalite skoru : {avg_q:.1f}/100")
        print("=" * 60)

    def _save_report(self):
        try:
            reporting = ReportingService()
            report_data = {
                **self.stats,
                "target_reps": self.target_reps,
                "asr_verify": self.asr_verify,
                "total_words": len(self.word_list),
            }
            report_data.pop("quality_scores", None)
            report_file = reporting.log_general_operation(
                "auto_collect", self.user_id, report_data
            )
            print(f"  Rapor kaydedildi: {report_file}")
        except Exception as e:
            print(f"  Rapor kaydedilemedi: {e}")


# ---------------------------------------------------------------------------
# CLI giriş noktası
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Otonom kelime kayıt sistemi — ENTER gerektirmez, otomatik doğrular."
    )
    parser.add_argument("user_id", help="Kullanici kimligi")
    parser.add_argument("word_file", help="Kaydedilecek kelimelerin bulundugu .txt dosyasi")
    parser.add_argument(
        "--reps",
        type=int,
        default=None,
        help=f"Her kelime icin tekrar sayisi (varsayilan: {config.IDEAL_REPETITIONS})",
    )
    parser.add_argument(
        "--asr-verify",
        action="store_true",
        default=config.AUTO_ASR_VERIFY,
        help="ASR ile kelime dogrulamasi etkinlestir (varsayilan: config.AUTO_ASR_VERIFY)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ilerlemeyi sifirdan baslat (mevcut kayitlari saymaz)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Ses esigi (0-1; varsayilan: config.AUTO_SOUND_THRESHOLD)",
    )
    args = parser.parse_args()

    # Kelime dosyasını oku
    word_file = Path(args.word_file)
    if not word_file.exists():
        print(f"Hata: Kelime dosyasi bulunamadi: {word_file}")
        sys.exit(1)

    with open(word_file, encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    if not words:
        print("Hata: Kelime dosyasi bos.")
        sys.exit(1)

    # config üzerinden threshold override
    if args.threshold is not None:
        config.AUTO_SOUND_THRESHOLD = args.threshold

    collector = AutoCollector(
        user_id=args.user_id,
        word_list=words,
        reps=args.reps,
        asr_verify=args.asr_verify,
        resume=not args.no_resume,
    )
    collector.run()


if __name__ == "__main__":
    main()
