
# -*- coding: utf-8 -*-

class NLU_System:
    """
    Metin girdisini analiz ederek kullanıcının niyetini (intent) ve
    parametrelerini (entities) anlayan sistem.
    Türkçe günlük konuşma ve sistem komutları için genişletilmiş kural seti.
    """

    def __init__(self):
        print("NLU Sistemi baslatiliyor...")
        self.intent_rules = {

            # --- Karsilasma ve sohbet ---
            "greeting": {
                "keywords": ["merhaba", "selam", "hey", "hi", "hello", "günaydın", "iyi günler", "iyi aksamlar"],
                "priority": 1,
            },
            "how_are_you": {
                "keywords": ["naber", "nasılsın", "nasil gidiyor", "iyi misin", "ne haber", "nasılsınız"],
                "priority": 2,
            },
            "farewell": {
                "keywords": ["hoşça kal", "görüşürüz", "güle güle", "iyi geceler", "iyi aksamlar", "bay bay"],
                "priority": 2,
            },
            "thanks": {
                "keywords": ["teşekkür", "teşekkürler", "sağol", "sağolun", "eyvallah", "mersi"],
                "priority": 2,
            },
            "sorry": {
                "keywords": ["özür", "affedersin", "pardon", "kusura bakma"],
                "priority": 2,
            },
            "yes_response": {
                "keywords": ["evet", "tamam", "olur", "peki", "tabii", "elbette", "kesinlikle"],
                "priority": 1,
            },
            "no_response": {
                "keywords": ["hayır", "yok", "olmaz", "istemiyorum", "hayır hayır"],
                "priority": 1,
            },

            # --- Günlük aktivite ---
            "what_are_you_doing": {
                "keywords": ["ne yapıyorsun", "ne yapıyorsunuz", "ne yapıyorsun bu gün", "bugün ne yapıyorsun"],
                "priority": 2,
            },
            "daily_activity": {
                "keywords": ["bugün", "gün", "yapıyorsun", "yapıyorsunuz", "plan", "program"],
                "priority": 3,
            },
            "combined_greeting": {
                "keywords": ["naber bu gün ne yapıyorsun", "nasılsın bugün ne yapıyorsun"],
                "priority": 4,
            },

            # --- Sorgulama ---
            "ask_time": {
                "keywords": ["saat", "kaç", "saat kaç", "şu an saat"],
                "priority": 1,
            },
            "ask_date": {
                "keywords": ["tarih", "bugün kaçıncı", "hangi gün", "bugün ne", "gün ay yıl"],
                "priority": 1,
            },
            "ask_weather": {
                "keywords": ["hava", "hava durumu", "yağmur", "güneş", "sıcaklık", "derece"],
                "priority": 1,
            },

            # --- Sistem komutları ---
            "take_note": {
                "keywords": ["not", "al", "kaydet", "hatırlat"],
                "entity": {"trigger": "diye", "name": "note_content"},
                "priority": 2,
            },
            "list_files": {
                "keywords": ["dosya", "listele", "klasör", "göster", "hangi dosyalar"],
                "priority": 1,
            },
            "show_time": {
                "keywords": ["saat", "kaç"],
                "priority": 1,
            },
            "show_current_directory": {
                "keywords": ["çalışma", "dizini", "klasör nerede", "neredeyim"],
                "priority": 1,
            },
            "who_are_you": {
                "keywords": ["kimsin", "sen kimsin", "adın ne", "adın", "ne yaparsın"],
                "priority": 1,
            },

            # --- Yardım ---
            "help": {
                "keywords": ["yardım", "yardımcı ol", "nasıl kullanırım", "ne yapabilirim", "komutlar"],
                "priority": 1,
            },

            # --- Çıkış ---
            "exit": {
                "keywords": ["çık", "kapat", "durdur", "görüşürüz", "kapat programı", "bitir"],
                "priority": 1,
            },
        }

    def process_text(self, text: str) -> tuple[str, dict]:
        """
        Verilen metni analiz eder ve niyeti ile parametrelerini döndürür.

        Returns
        -------
        tuple[str, dict]
            (intent, entities) — örnek: ('take_note', {'note_content': 'yarın toplantı'})
        """
        normalized = text.lower().strip()
        words = normalized.split()

        if not words:
            return "unknown", {}

        best_intent = None
        best_score = 0.0
        best_entities: dict = {}

        for intent, rules in self.intent_rules.items():
            score = 0.0
            priority = rules.get("priority", 1)

            for keyword in rules["keywords"]:
                if keyword in normalized:
                    score = priority * 2 if len(keyword.split()) > 1 else float(priority)
                    break

            if score == 0.0:
                matches = sum(1 for kw in rules["keywords"] if kw in normalized)
                if matches:
                    score = (matches / len(rules["keywords"])) * priority

            if score <= 0.0:
                continue

            entities: dict = {}
            if "entity" in rules:
                rule = rules["entity"]
                trigger = rule["trigger"]
                if trigger in words:
                    idx = words.index(trigger)
                    value = " ".join(words[idx + 1:])
                    if value:
                        entities[rule["name"]] = value

            if score > best_score:
                best_score = score
                best_intent = intent
                best_entities = entities

        if best_intent:
            print(f"Niyet: '{best_intent}' (skor: {best_score:.2f})")
            for key, val in best_entities.items():
                print(f"  Parametre: {key} = '{val}'")
            return best_intent, best_entities

        print("Anlasılır bir niyet bulunamadi.")
        return "unknown", {}


if __name__ == "__main__":
    nlu = NLU_System()
    tests = [
        "naber bu gün ne yapıyorsun",
        "merhaba nasıl gidiyor",
        "bana dosyaları listele lütfen",
        "notlarıma yarın saat 10'da toplantı var diye not al",
        "saat kaç acaba",
        "programı kapat",
        "teşekkürler",
        "hoşça kal",
        "bugün hava nasıl",
        "yardım eder misin",
        "evet tamam",
        "hayır istemiyorum",
    ]
    for s in tests:
        intent, entities = nlu.process_text(s)
        print(f"  '{s}' -> {intent} {entities}\n")
