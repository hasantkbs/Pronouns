
# -*- coding: utf-8 -*-

class NLU_System:
    """
    Metin girdisini analiz ederek kullanıcının niyetini (intent) ve 
    parametrelerini (entities) anlayan sistem.
    """
    def __init__(self):
        print("NLU Sistemi başlatılıyor...")
        # Niyet kurallarını yeniden yapılandır - Türkçe günlük konuşma için genişletildi
        self.intent_rules = {
            # Günlük selamlaşma ve sohbet
            'greeting': {
                'keywords': ['merhaba', 'selam', 'hey', 'hi', 'hello'],
                'priority': 1
            },
            'how_are_you': {
                'keywords': ['naber', 'nasılsın', 'nasıl gidiyor', 'iyi misin'],
                'priority': 2
            },
            'what_are_you_doing': {
                'keywords': ['ne yapıyorsun', 'ne yapıyorsunuz', 'ne yapıyorsun bu gün', 'bugün ne yapıyorsun'],
                'priority': 2
            },
            'daily_activity': {
                'keywords': ['bugün', 'gün', 'yapıyorsun', 'yapıyorsunuz'],
                'priority': 3
            },
            'combined_greeting': {
                'keywords': ['naber bu gün ne yapıyorsun', 'nasılsın bugün ne yapıyorsun'],
                'priority': 4
            },
            
            # Sistem komutları
            'take_note': {
                'keywords': ['not', 'al'],
                'entity': {'trigger': 'diye', 'name': 'note_content'},
                'priority': 1
            },
            'list_files': {
                'keywords': ['dosya', 'listele'],
                'priority': 1
            },
            'show_time': {
                'keywords': ['saat', 'kaç'],
                'priority': 1
            },
            'show_current_directory': {
                'keywords': ['çalışma', 'dizini'],
                'priority': 1
            },
            'who_are_you': {
                'keywords': ['kimsin', 'sen kimsin'],
                'priority': 1
            },
            'exit': {
                'keywords': ['çık', 'kapat', 'durdur', 'görüşürüz'],
                'priority': 1
            },
        }

    def process_text(self, text: str) -> tuple[str, dict]:
        """
        Verilen metni analiz eder ve niyeti ile parametrelerini döndürür.

        Args:
            text (str): ASR tarafından üretilmiş metin.

        Returns:
            tuple[str, dict]: (intent, entities) formatında bir demet.
                              Örnek: ('take_note', {'note_content': 'yarın toplantı var'})
        """
        normalized_text = text.lower().strip()
        words = normalized_text.split()
        
        # Boş metin kontrolü
        if not words:
            return 'unknown', {}
        
        # En yüksek öncelikli niyeti bul
        best_intent = None
        best_score = 0
        best_entities = {}
        
        for intent, rules in self.intent_rules.items():
            score = 0
            priority = rules.get('priority', 1)
            
            # Tam cümle eşleşmesi kontrolü (yüksek öncelik)
            for keyword in rules['keywords']:
                if keyword in normalized_text:
                    # Tam cümle eşleşmesi varsa daha yüksek skor ver
                    if len(keyword.split()) > 1:
                        score = priority * 2
                    else:
                        score = priority
                    break
            
            # Anahtar kelime eşleşmesi (düşük öncelik)
            if score == 0:
                keyword_matches = 0
                for keyword in rules['keywords']:
                    if keyword in normalized_text:
                        keyword_matches += 1
                
                if keyword_matches > 0:
                    score = (keyword_matches / len(rules['keywords'])) * priority
            
            if score > 0:
                # Parametre (entity) çıkarma mantığı
                entities = {}
                if 'entity' in rules:
                    entity_rule = rules['entity']
                    trigger_word = entity_rule['trigger']
                    
                    if trigger_word in words:
                        trigger_index = words.index(trigger_word)
                        # Tetikleyici kelimeden sonraki tüm kelimeleri birleştir
                        entity_value = " ".join(words[trigger_index + 1:])
                        if entity_value:
                            entities[entity_rule['name']] = entity_value
                
                # Daha yüksek skor varsa güncelle
                if score > best_score:
                    best_score = score
                    best_intent = intent
                    best_entities = entities
        
        if best_intent:
            print(f"Niyet bulundu: '{best_intent}' (skor: {best_score:.2f})")
            if best_entities:
                for key, value in best_entities.items():
                    print(f"  -> Parametre bulundu: {key} = '{value}'")
            return best_intent, best_entities
        
        print("Anlaşılır bir niyet bulunamadı.")
        return 'unknown', {}

if __name__ == '__main__':
    nlu_system = NLU_System()
    print("\n--- NLU Testi Başlatılıyor ---")

    test_sentences = [
        "naber bu gün ne yapıyorsun",
        "nasılsın bugün ne yapıyorsun",
        "nasılsın bugün",
        "merhaba nasıl gidiyor",
        "bana dosyaları listele lütfen",
        "notlarıma yarın saat 10'da toplantı var diye not al",
        "saat kaç acaba",
        "programı kapat",
        "sen kimsin",
        "bugün hava nasıl",
        "hey nasılsın",
        "selam naber"
    ]

    for sentence in test_sentences:
        print(f"\nCümle: '{sentence}'")
        intent, entities = nlu_system.process_text(sentence)
        print(f"  -> Bulunan Niyet: '{intent}', Parametreler: {entities}")

    print("\nTest tamamlandı.")
