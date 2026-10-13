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
import { withCurrent } from '../lib/uiFormat';
import AppBar from '../components/AppBar';
import { FilledButton, OutlinedButton, ResultTile, SectionHeader, SegmentedButton } from '../components/ui';

const MODEL_OPTIONS = ['Furkan', 'base', 'large'];
const ALGO_OPTIONS = ['lora', 'qlora', 'full'];

// handleSave / handleRebuildVoiceProfile hata mesajlari bu kaliplarla baslar
const isErrorMessage = (m: string) => m.startsWith('Hata:') || m.startsWith('Ses profili güncellenemedi');

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

  return (
    <main className="page settings">
      <AppBar title="Model Ayarları" backTo="/" />
      <div className="page-body">
        {loading ? (
          <div className="spinner spinner--large" role="status" aria-label="Yükleniyor" />
        ) : error ? (
          <ResultTile label="Hata" value={error} tone="error" />
        ) : settings ? (
          <>
            {modelInfo && (
              <ResultTile
                label="Model"
                tone="settings"
                value={`Model: ${modelInfo.model} • WER: ${modelInfo.wer} • Veri: ${modelInfo.sample_count} ses`}
              />
            )}

            <SectionHeader>Model Tercihi</SectionHeader>
            <SegmentedButton
              label="Model Tercihi"
              className="segmented--settings"
              options={withCurrent(MODEL_OPTIONS, settings.model).map((m) => ({ value: m, label: m }))}
              value={settings.model}
              onChange={(model) => setSettings({ ...settings, model })}
            />

            <SectionHeader>Fine-Tune Algoritması</SectionHeader>
            <SegmentedButton
              label="Fine-Tune Algoritması"
              className="segmented--settings"
              options={withCurrent(ALGO_OPTIONS, settings.algorithm).map((a) => ({ value: a, label: a.toUpperCase() }))}
              value={settings.algorithm}
              onChange={(algorithm) => setSettings({ ...settings, algorithm })}
            />

            <SectionHeader>Kendi Kendine Öğrenme</SectionHeader>
            <label className="switch-row">
              <span style={{ flex: 1 }}>
                Self-learning aktif
                <br />
                <span className="muted small">
                  {settings.self_learning ? 'Yeni kayıtlar otomatik modele eklenir' : 'Manuel eğitim gerektirir'}
                </span>
              </span>
              <input
                type="checkbox"
                role="switch"
                checked={settings.self_learning}
                onChange={(e) => setSettings({ ...settings, self_learning: e.target.checked })}
              />
            </label>

            <SectionHeader>Hiperparametreler</SectionHeader>
            <label className="field">
              <span className="field-label">Learning Rate</span>
              <input
                type="number"
                step="0.00001"
                value={settings.learning_rate}
                onChange={(e) => setSettings({ ...settings, learning_rate: Number(e.target.value) })}
              />
            </label>
            <label className="field">
              <span className="field-label">Epoch Sayısı</span>
              <input
                type="number"
                value={settings.epochs}
                onChange={(e) => setSettings({ ...settings, epochs: Number(e.target.value) })}
              />
            </label>
            <label className="field">
              <span className="field-label">Batch Size</span>
              <input
                type="number"
                value={settings.batch_size}
                onChange={(e) => setSettings({ ...settings, batch_size: Number(e.target.value) })}
              />
            </label>

            <FilledButton
              icon={saving ? 'hourglass_top' : 'save'}
              className="button--settings"
              onClick={handleSave}
              disabled={saving}
            >
              {saving ? 'Kaydediliyor...' : 'Ayarları Kaydet'}
            </FilledButton>

            <SectionHeader>Ses Profili</SectionHeader>
            <ResultTile
              label="Durum"
              tone="settings"
              value={
                voiceProfile
                  ? voiceProfile.exists
                    ? `${voiceProfile.sample_count} kayıttan oluşturuldu${voiceProfile.created_at ? ` • ${voiceProfile.created_at}` : ''}`
                    : 'Henüz ses profili oluşturulmadı.'
                  : 'Durum yükleniyor...'
              }
            />
            <OutlinedButton
              icon={voiceProfileBusy ? 'hourglass_top' : 'record_voice_over'}
              className="button--settings"
              onClick={handleRebuildVoiceProfile}
              disabled={voiceProfileBusy}
            >
              {voiceProfileBusy ? 'Güncelleniyor...' : 'Ses Profilini Güncelle'}
            </OutlinedButton>

            {message && <ResultTile label="Durum" value={message} tone={isErrorMessage(message) ? 'error' : 'settings'} />}
          </>
        ) : null}
      </div>
    </main>
  );
}
