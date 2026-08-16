import { useEffect, useState } from 'react';
import {
  getModelInfo,
  getSettings,
  getVoiceProfileStatus,
  rebuildVoiceProfile,
  saveSettings,
  type ModelInfoResponse,
  type SettingsResponse,
  type VoiceProfileStatus,
} from '../lib/api';

const MODEL_OPTIONS = ['Furkan', 'base', 'large'];
const ALGO_OPTIONS = ['lora', 'qlora', 'full'];

export default function AyarlarPage() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const [settings, setSettings] = useState<SettingsResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [voiceProfile, setVoiceProfile] = useState<VoiceProfileStatus | null>(null);
  const [voiceProfileBusy, setVoiceProfileBusy] = useState(false);

  useEffect(() => {
    void loadAll();
  }, []);

  async function loadAll() {
    setLoading(true);
    setError(null);
    try {
      const [s, info] = await Promise.all([getSettings(), getModelInfo()]);
      setSettings(s);
      setModelInfo(info);
    } catch (e) {
      setError(`Ayarlar yüklenemedi: ${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
    try {
      setVoiceProfile(await getVoiceProfileStatus());
    } catch {
      // ses profili durumu opsiyonel
    }
  }

  async function handleSave() {
    if (!settings) return;
    setSaving(true);
    try {
      const updated = await saveSettings({
        model: settings.model,
        algorithm: settings.algorithm,
        self_learning: settings.self_learning,
        learning_rate: settings.learning_rate,
        epochs: settings.epochs,
        batch_size: settings.batch_size,
      });
      setSettings(updated);
      setMessage('Ayarlar kaydedildi.');
    } catch (e) {
      setMessage(`Hata: ${(e as Error).message}`);
    } finally {
      setSaving(false);
    }
  }

  async function handleRebuildVoiceProfile() {
    setVoiceProfileBusy(true);
    try {
      const result = await rebuildVoiceProfile();
      setVoiceProfile(await getVoiceProfileStatus());
      setMessage(`Ses profili güncellendi (${result.sample_count} kayıt).`);
    } catch (e) {
      setMessage(`Ses profili güncellenemedi: ${(e as Error).message}`);
    } finally {
      setVoiceProfileBusy(false);
    }
  }

  if (loading) return <p>Yükleniyor...</p>;
  if (error) return <p role="alert">{error}</p>;
  if (!settings) return null;

  return (
    <main>
      <h1>Ayarlar</h1>
      {modelInfo && (
        <p>
          Model: {modelInfo.model} • WER: {modelInfo.wer} • Veri: {modelInfo.sample_count} ses
        </p>
      )}

      <label>
        Model
        <select
          value={settings.model}
          onChange={(e) => setSettings({ ...settings, model: e.target.value })}
        >
          {MODEL_OPTIONS.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </label>

      <label>
        Algoritma
        <select
          value={settings.algorithm}
          onChange={(e) => setSettings({ ...settings, algorithm: e.target.value })}
        >
          {ALGO_OPTIONS.map((a) => (
            <option key={a} value={a}>
              {a}
            </option>
          ))}
        </select>
      </label>

      <label>
        <input
          type="checkbox"
          checked={settings.self_learning}
          onChange={(e) => setSettings({ ...settings, self_learning: e.target.checked })}
        />
        Kendi kendine öğrenme
      </label>

      <label>
        Öğrenme oranı
        <input
          type="number"
          step="0.00001"
          value={settings.learning_rate}
          onChange={(e) => setSettings({ ...settings, learning_rate: Number(e.target.value) })}
        />
      </label>

      <label>
        Epoch
        <input
          type="number"
          value={settings.epochs}
          onChange={(e) => setSettings({ ...settings, epochs: Number(e.target.value) })}
        />
      </label>

      <label>
        Batch size
        <input
          type="number"
          value={settings.batch_size}
          onChange={(e) => setSettings({ ...settings, batch_size: Number(e.target.value) })}
        />
      </label>

      <button onClick={handleSave} disabled={saving}>
        {saving ? 'Kaydediliyor...' : 'Kaydet'}
      </button>

      <section>
        <h2>Ses Profili</h2>
        <p>
          {voiceProfile
            ? voiceProfile.exists
              ? `${voiceProfile.sample_count} kayıttan oluşturuldu${voiceProfile.created_at ? ` • ${voiceProfile.created_at}` : ''}`
              : 'Henüz ses profili oluşturulmadı.'
            : 'Durum yükleniyor...'}
        </p>
        <button onClick={handleRebuildVoiceProfile} disabled={voiceProfileBusy}>
          {voiceProfileBusy ? 'Güncelleniyor...' : 'Ses Profilini Güncelle'}
        </button>
      </section>

      {message && <p>{message}</p>}
    </main>
  );
}
