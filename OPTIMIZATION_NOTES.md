# Optimization Notes

## Main change in v4

The pipeline now supports a configurable four-ASR ensemble:

```text
whisper + qwen + cohere + granite
```

This is not just four transcripts dumped into a sidecar. The code uses them in a production-safe way:

1. Whisper generates word timestamps and acts as the timing anchor.
2. Qwen, Cohere, and Granite provide additional transcripts.
3. A per-channel consensus transcript is built.
4. PII detection runs on the final transcript and, by default, every enabled ASR transcript.
5. Transcript-only model entities are projected onto the Whisper timestamp timeline.
6. Unmapped PII triggers full-channel masking by default.

## Speed optimizations kept from v2

- Single decode of source audio to stereo float32 48 kHz.
- Reuse decoded 48 kHz buffer for masking.
- Create 16 kHz mono ASR arrays in memory.
- Avoid Wav2Vec2 forced alignment.
- Use faster-whisper word timestamps instead of a second alignment model.
- Copy original audio for no-PII files when source is already stereo 48 kHz Opus.
- Atomic file writes.
- Output validation by ffprobe.
- SQLite checkpointing.
- JSONL reporting with bounded CSV memory.

## Multi-ASR performance tradeoff

Four ASR models improve PII recall, but they are expensive. There is no free optimization that makes four large models as fast as one model.

Recommended operating modes:

### Large GPU

```yaml
asr:
  model_residency: keep_loaded
```

This avoids repeated model loading and is the fastest configuration if VRAM is sufficient.

### Small GPU

```yaml
asr:
  model_residency: unload_after_file
```

This reduces out-of-memory risk, but repeated model load/unload is slower.

### Throughput smoke test

```yaml
asr:
  engines:
    whisper:
      enabled: true
    qwen:
      enabled: false
    cohere:
      enabled: false
    granite:
      enabled: false
```

This is useful for validating audio I/O, masking, sidecars, and output format before enabling all four engines.

## Why Whisper is still required

Qwen, Cohere, and Granite usually return transcript text but not reliable word timestamps. Audio masking needs exact time ranges. Therefore:

```text
Whisper = timestamp anchor
Other ASR models = transcript recall and consensus
```

If Whisper misses a PII token but another model catches it, the code aligns that other transcript back to Whisper's timestamp words. If alignment fails, the default safety policy masks the full detected channel.

## Best speed settings

```yaml
asr:
  input_audio_strategy: single_decode
  model_residency: keep_loaded
  pii_detection_transcript_scope: final_and_all_engines
  engines:
    whisper:
      use_batched_pipeline: true
      batch_size: 8
      beam_size: 1
      vad_filter: true
      word_timestamps: true
    qwen:
      batch_size: 2
    cohere:
      batch_size: 2
    granite:
      batch_size: 2

pii:
  batch_size: 16
  chunk_chars: 1800
  chunk_overlap_chars: 240

masking:
  mode: silence
  copy_input_if_no_pii: true

runtime:
  ffmpeg_threads: 1
  copy_unmasked_when_no_pii: true
  unmasked_copy_method: hardlink_or_copy
  atomic_output: true
```

## Best safety settings

```yaml
asr:
  pii_detection_transcript_scope: final_and_all_engines

masking:
  target_channels: detected_channel
  unmapped_entity_policy: mask_full_channel
  pad_sec: 0.12
  min_duration_sec: 0.30
```

Use `target_channels: both` only when compliance requires more conservative masking. It will remove more non-PII speech.

## Parallelism guidance

Use sharding across machines or GPUs:

```bash
python -m pii_audio_masking_pipeline.run --config config.yaml --stage process --shard-count 8 --shard-index 0
```

For best throughput:

```text
one worker per GPU
model_residency: keep_loaded
all four ASR models enabled only on machines with enough VRAM
```

Do not run multiple four-model workers on a single small GPU. That will usually reduce throughput through context switching or cause out-of-memory failures.

## What was intentionally removed

The notebook's Wav2Vec2 forced-alignment step is not used in the production path. It is too slow for this task and duplicates work that faster-whisper already provides through word timestamps.

## Validation performed in the package environment

```text
Python compile: passed
pytest with plugin autoload disabled: 14 passed
CLI import path: passed
config.example.yaml validation: passed
```

End-to-end validation on AMC production audio still needs to be run in your environment because the sandbox does not contain `/mnt/amc-data`, your local ASR model folders, or your exact FFmpeg and GPU setup.
