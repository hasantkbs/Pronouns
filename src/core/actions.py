
# -*- coding: utf-8 -*-
import os
import datetime
import random

# --- Eylem Fonksiyonları ---

def greeting():
    """Selamlama mesajı döndürür."""
    print("Eylem: Selamlama...")
    greetings = [
        "Merhaba! Size nasıl yardımcı olabilirim?",
        "Selam! Bugün nasılsınız?",
        "Hey! Hoş geldiniz!",
        "Merhaba! Nasıl gidiyor?"
    ]
    return random.choice(greetings)

def how_are_you():
    """Nasılsın sorusuna cevap verir."""
    print("Eylem: Nasılsın sorusuna cevap veriliyor...")
    responses = [
        "İyiyim, teşekkür ederim! Siz nasılsınız?",
        "Çok iyiyim! Sizin gününüz nasıl geçiyor?",
        "Harika! Size nasıl yardımcı olabilirim?",
        "İyiyim, umarım siz de iyisinizdir!"
    ]
    return random.choice(responses)

def what_are_you_doing():
    """Ne yapıyorsun sorusuna cevap verir."""
    print("Eylem: Ne yapıyorsun sorusuna cevap veriliyor...")
    responses = [
        "Şu anda sizin sesli komutlarınızı dinliyorum ve size yardım etmeye çalışıyorum!",
        "Sizinle sohbet ediyorum ve komutlarınızı yerine getirmeye hazırım.",
        "Sesli asistan olarak görev yapıyorum. Size nasıl yardımcı olabilirim?",
        "Sizin komutlarınızı anlamaya ve yanıt vermeye çalışıyorum!"
    ]
    return random.choice(responses)

def daily_activity():
    """Günlük aktivite sorusuna cevap verir."""
    print("Eylem: Günlük aktivite sorusuna cevap veriliyor...")
    responses = [
        "Bugün sizinle sohbet ediyorum ve komutlarınızı yerine getiriyorum!",
        "Günüm çok güzel geçiyor, sizinle konuşmak harika!",
        "Bugün size yardım etmek için buradayım!",
        "Günümü sizinle geçirmekten mutluyum!"
    ]
    return random.choice(responses)

def combined_greeting():
    """Naber bu gün ne yapıyorsun gibi birleşik selamlaşma sorusuna cevap verir."""
    print("Eylem: Birleşik selamlaşma sorusuna cevap veriliyor...")
    responses = [
        "İyiyim, teşekkür ederim! Bugün sizinle sohbet ediyorum ve komutlarınızı yerine getiriyorum. Size nasıl yardımcı olabilirim?",
        "Çok iyiyim! Şu anda sizin sesli komutlarınızı dinliyorum. Günüm harika geçiyor, sizinle konuşmak çok güzel!",
        "Harika! Bugün size yardım etmek için buradayım. Sesli asistan olarak görev yapıyorum ve sizinle sohbet etmekten mutluyum!",
        "İyiyim, umarım siz de iyisinizdir! Bugün günümü sizinle geçiriyorum, komutlarınızı anlamaya ve yanıt vermeye çalışıyorum!"
    ]
    return random.choice(responses)

def take_note(note_content: str = None):
    """Verilen içeriği bir not dosyasına kaydeder."""
    print(f"Eylem: Nota kaydediliyor... İçerik: '{note_content}'")
    if not note_content:
        return "Ne not almamı istersiniz? Lütfen '... diye not al' şeklinde belirtin."
    
    # Ensure the data/users directory exists
    notes_dir = "data/users"
    os.makedirs(notes_dir, exist_ok=True)
    notes_file_path = os.path.join(notes_dir, "notes.txt")

    try:
        with open(notes_file_path, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp}: {note_content}\n")
        return f"'...{note_content[-20:]}' içeriğiyle not alındı."
    except Exception as e:
        return f"Nota kaydederken bir hata oluştu: {e}"

def show_current_directory():
    """Mevcut çalışma dizinini döndürür."""
    print("Eylem: Mevcut dizin gösteriliyor...")
    try:
        return f"Şu anki çalışma dizini: {os.getcwd()}"
    except Exception as e:
        return f"Dizin bilgisi alınırken bir hata oluştu: {e}"

def list_files():
    """Mevcut dizindeki dosya ve klasörleri listeler."""
    print("Eylem: Dosyalar listeleniyor...")
    try:
        files = os.listdir('.')
        if not files:
            return "Bu dizinde hiç dosya yok."
        return "\n".join(files)
    except Exception as e:
        return f"Dosyaları listelerken bir hata oluştu: {e}"

def show_time():
    """Mevcut saati döndürür."""
    print("Eylem: Saat gösteriliyor...")
    now = datetime.datetime.now()
    return f"Şu an saat {now.strftime('%H:%M')}."

def who_are_you():
    """Asistanın kimliğini döndürür."""
    print("Eylem: Kimlik bilgisi veriliyor...")
    return "Ben Wav2Vec2 tabanlı bir konuşma anlama sistemiyim. Konuşma bozukluğu olan bireyler için özel olarak tasarlanmış bir sesli asistanım."

def exit_program():
    """Çıkış mesajı döndürür."""
    print("Eylem: Çıkış yapılıyor...")
    return "Görüşmek üzere! İyi günler!"

def handle_unknown():
    """Anlaşılamayan niyetler için bir mesaj döndürür."""
    print("Eylem: Anlaşılamayan komut işleniyor...")
    responses = [
        "Üzgünüm, ne istediğinizi anlayamadım. Lütfen tekrar deneyin.",
        "Bu komutu anlayamadım. Başka bir şekilde söyleyebilir misiniz?",
        "Anlayamadığım bir komut verdiniz. Size nasıl yardımcı olabilirim?",
        "Bu komutu tanımıyorum. Lütfen farklı bir şekilde ifade edin."
    ]
    return random.choice(responses)


# --- Niyet ve Eylem Eşleştirme ---
ACTION_MAP = {
    'greeting': greeting,
    'how_are_you': how_are_you,
    'what_are_you_doing': what_are_you_doing,
    'daily_activity': daily_activity,
    'combined_greeting': combined_greeting,
    'take_note': take_note,
    'list_files': list_files,
    'show_time': show_time,
    'show_current_directory': show_current_directory,
    'who_are_you': who_are_you,
    'exit': exit_program,
    'unknown': handle_unknown,
}

def run_action(intent: str, entities: dict = None):
    """
    Verilen niyete ve parametrelere göre eylemi çalıştırır.

    Args:
        intent (str): NLU sisteminden gelen niyet.
        entities (dict): NLU sisteminden gelen parametreler.

    Returns:
        str: Eylem fonksiyonunun ürettiği sonuç metni.
    """
    entities = entities or {}
    action_function = ACTION_MAP.get(intent, handle_unknown)
    
    # Fonksiyonu, parametrelerle birlikte çalıştır
    return action_function(**entities)


if __name__ == '__main__':
    print("--- Eylem Testi Başlatılıyor ---")

    # Test senaryoları
    test_cases = [
        {'intent': 'greeting', 'entities': {}},
        {'intent': 'how_are_you', 'entities': {}},
        {'intent': 'what_are_you_doing', 'entities': {}},
        {'intent': 'daily_activity', 'entities': {}},
        {'intent': 'combined_greeting', 'entities': {}},
        {'intent': 'list_files', 'entities': {}},
        {'intent': 'show_time', 'entities': {}},
        {'intent': 'take_note', 'entities': {'note_content': 'bu bir test notudur'}},
        {'intent': 'take_note', 'entities': {}}, # Parametresiz test
        {'intent': 'non_existent_intent', 'entities': {}},
    ]

    for case in test_cases:
        print(f"\nNiyet: '{case['intent']}', Parametreler: {case['entities']}")
        result = run_action(case['intent'], case['entities'])
        print(f"  -> Sonuç: \n{result}")

    print("\nTest tamamlandı.")
