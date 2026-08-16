export interface AudioCaptureHandle {
  sampleRate: number;
  stop: () => void;
}

export async function startAudioCapture(
  onChunk: (chunk: Int16Array) => void,
  bufferSize: 1024 | 2048 | 4096 = 4096,
): Promise<AudioCaptureHandle> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });

  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);

  processor.onaudioprocess = (event) => {
    const float32 = event.inputBuffer.getChannelData(0);
    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      const sample = Math.max(-1, Math.min(1, float32[i]));
      int16[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }
    onChunk(int16);
  };

  // ScriptProcessorNode only fires onaudioprocess while connected to a
  // destination; we discard the (silent) output, we only need the input.
  source.connect(processor);
  processor.connect(audioContext.destination);

  const stop = () => {
    processor.disconnect();
    source.disconnect();
    stream.getTracks().forEach((track) => track.stop());
    void audioContext.close();
  };

  return { sampleRate: audioContext.sampleRate, stop };
}
