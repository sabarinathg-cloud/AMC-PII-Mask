# Optimization Notes

This version is optimized for speed and efficiency while preserving conservative PII masking behavior.

## Implemented optimizations

1. **Single decode path**
   - Each input is decoded once to stereo 48 kHz float32 when `asr.input_audio_strategy: single_decode`.
   - The same buffer is reused for ASR preparation and masking.

2. **No forced alignment**
   - Wav2Vec2 forced alignment is removed from the production path.
   - faster-whisper word timestamps are used as the timestamp anchor.

3. **Four-ASR ensemble**
   - Whisper, Qwen, Cohere, and Granite are configurable.
   - Each engine can be enabled or disabled independently.
   - PII detection runs over the final consensus transcript and every enabled engine transcript by default.

4. **File-level micro-batching**
   - `runtime.file_batch_size` controls how many files are prepared together.
   - Qwen, Cohere, and Granite receive cross-file channel batches.
   - PII detection receives all transcript texts from the micro-batch in one call.

5. **Whisper internal batching**
   - Whisper uses faster-whisper `BatchedInferencePipeline` with `batch_size`.
   - Whisper is not cross-file batched because it is the word-timestamp anchor.

6. **Batched neural PII detection**
   - GLiNER uses `batch_predict_entities` when available.
   - Piiranha and spaCy receive batched texts.
   - Long transcripts are chunked with overlap.

7. **No-PII fast copy**
   - If no PII spans are found and the input is already stereo 48 kHz Opus, the original audio is copied or hardlinked.
   - No decode or re-encode is performed at finalization for clean files.

8. **Memory guard**
   - `runtime.file_batch_max_decoded_audio_gb` prevents holding too much decoded audio in RAM.
   - Long calls can fall back to per-file processing.

9. **Atomic writes and validation**
   - Output writes are atomic.
   - ffprobe validation enforces Opus, 48 kHz, 2 channels, and duration sanity.

10. **Shard-level scale-out**
   - `runtime.shard_count` and `runtime.shard_index` support one worker per GPU or machine.

## Recommended production settings

```yaml
asr:
  input_audio_strategy: single_decode
  model_residency: keep_loaded
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

runtime:
  file_batch_size: 2
  file_batch_max_decoded_audio_gb: 2.0
  copy_unmasked_when_no_pii: true
  atomic_output: true
```

## Tuning strategy

- Short clips: increase `runtime.file_batch_size` first, then Qwen/Cohere/Granite batch sizes.
- Long full calls: keep `runtime.file_batch_size` small because decoded audio buffers are retained until finalization.
- Small GPUs: set `asr.model_residency: unload_after_file`, but expect slower throughput.
- Accuracy-sensitive audit runs: keep all ASR engines enabled and use `pii_detection_transcript_scope: final_and_all_engines`.
- Maximum speed smoke tests: run only Whisper with `--enable-asr-engines whisper`.

## Known hard limit

True cross-file Whisper batching is not used. faster-whisper's supported optimization is internal segment batching through `BatchedInferencePipeline`. Since Whisper provides word timestamps for masking, it remains the anchor and runs per channel. The other ASR engines are transcript-only and are batched across files.
