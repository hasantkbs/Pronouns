# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
import pytest

# Proje ana dizinindeki modüllere ulaşabilmek için sys.path'e ekleme yapıyoruz
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Artık config ve utils'i import edebiliriz
import config
from utils import EmbeddingGenerator


@pytest.fixture(scope="module")
def embedding_generator():
    """
    Testler boyunca sadece bir kere model yüklemek için
    paylaşılan bir EmbeddingGenerator nesnesi oluşturur.
    """
    # Not: Bu, testler her çalıştığında modeli indireceği için zaman alabilir.
    print("\nTest için EmbeddingGenerator hazırlanıyor (model indirilebilir)...")
    generator = EmbeddingGenerator(model_name=config.MODEL_NAME)
    return generator


def test_embedding_generator_initialization(embedding_generator):
    """
    Test: EmbeddingGenerator sınıfı doğru bir şekilde başlatılıyor mu?
    """
    assert embedding_generator is not None
    assert isinstance(embedding_generator.model, torch.nn.Module)
    assert isinstance(embedding_generator.processor, object)  # Wav2Vec2Processor
    print("EmbeddingGenerator başlatma testi başarılı.")


def test_get_embedding_from_data(embedding_generator):
    """
    Test: Sahte ses verisinden embedding üretimi çalışıyor mu ve
    doğru formatta bir çıktı veriyor mu?
    """
    # 1. Hazırlık: 1 saniyelik boş bir ses sinyali oluştur
    sampling_rate = config.ORNEKLEME_ORANI
    dummy_audio = np.zeros(sampling_rate, dtype=np.float32)

    # 2. Eylem: Embedding'i üret
    embedding = embedding_generator.get_embedding_from_data(dummy_audio, sampling_rate)

    # 3. Doğrulama: Çıktının özelliklerini kontrol et
    assert embedding is not None
    assert isinstance(embedding, torch.Tensor)
    assert embedding.ndim == 1, "Embedding tek boyutlu bir vektör olmalı"
    assert embedding.shape[0] == config.EMBEDDING_DIM, f"Embedding boyutu {config.EMBEDDING_DIM} olmalı"
    print("Veriden embedding üretme testi başarılı.")
