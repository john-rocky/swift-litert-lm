# Verified on device

Gemma 4 E2B (`gemma-4-E2B-it.litertlm`) on a physical **iPhone 18 Pro (iOS 27.2)** with LiteRT-LM
v0.19.0, measured on 2026-11-05 with the `LiteRTDemo` self-tests in Debug builds. The tok/s and TTFT
values are LiteRT-LM's own counters.

One generation on a new `LiteRTChat` per modality, with the default prewarm and sampler
(`LITERT_G0_TEST`, two runs). Each chat was ready in 0.6 s. Wall is the time around `respond`.

| Modality | Output | Decode | Prefill | TTFT | Wall | Footprint |
|---|---|---|---|---|---|---|
| Text (GPU) | one-sentence answer | ~80 tok/s | 475–496 tok/s | 0.04–0.05 s | 0.4 s | 468–469 MB |
| Image (CPU encoder) | "Apples" for `apple.png` | — ¹ | 3226–3470 tok/s | 0.09–0.10 s | 1.2–1.4 s | 1214–1384 MB |
| Audio (CPU encoder) | "Have a wonderful day." | 85–88 tok/s | 1836–2052 tok/s | 0.05 s | 0.3 s | 264–269 MB |

¹ image decode tok/s is statistically meaningless here (one-word answer).

Text on one engine, without `LiteRTChat` (`LITERT_BENCH`, two runs): GPU, `maxNumTokens` 2048,
temperature 0, the same prompt and a 587-token answer each turn, a new conversation per turn.

| Run | Engine init | Decode, turn 1 | Decode, turns 2–4 | Prefill, turns 2–4 | TTFT | Footprint |
|---|---|---|---|---|---|---|
| 1 | 0.5 s | 84.9 tok/s | 88.4, 88.4, 88.5 tok/s | 818, 823, 817 tok/s | 0.04 s | 455 MB |
| 2 | 0.4 s | 88.4 tok/s | 87.0, 88.5, 88.4 tok/s | 815, 826, 822 tok/s | 0.04 s | 455 MB |

`LiteRTChat(.gemma4_E2B)`, which loads text and image, held 506–508 MB in a two-turn chat
(`LITERT_CHAT_CHECK`, three runs). Its answers had 2 and 6 tokens, too few for a decode number. One
`.all` chat with text, image and audio in turn reached 1532–1533 MB (`LITERT_MMCHAT`, two runs).

The ready and init times above had the GPU cache from earlier runs in place; the first load after
install was not timed. The bench ran before the wrapper patch in
[LiteRT-LM v0.19.0](litert-lm-v0.19.0.md). The text row of the first table ran on both sides of it, at
80.4 and 79.9 tok/s.

**On decode speed.** On v0.19.0 the first turn on an engine ran about as fast as later turns.
`LiteRTChat` runs a small **prewarm** during setup (`prewarm: true`); on v0.13.1 it made the first
message warm. On v0.19.0, in the same bench runs, the first message decoded at 62.6 and 68.7 tok/s
with the prewarm, and at 86.9 and 87.8 tok/s without it. The run without it came first both times
and the order was not swapped, so the cause is not settled. The bundled MTP speculative-decoding
drafter was **not** a useful lever on v0.13.1 — enabling it (`speculativeDecoding: true`) was flaky
("Failed to create engine") with no measured speedup — so it's off by
default. (Probe it yourself: `LITERT_BENCH=1`.)

Logs, in the maintainer's `lane/` directory and not in this repository: `lane/device/g0.log` and
`lane/device/r3/obs_fix_g0.log` (first table); `lane/device/bench_v019.log` and
`lane/device/bench_v019_2.log` (second table, prewarm); `lane/device/r3/final/chat.log`,
`lane/device/r3/chat.log` and `lane/device/r3/obs_fix_rel_chat.log` (the two-turn chat, the last a
Release build); `lane/device/mmchat_v019.log` and `lane/device/r3/obs_fix_mmchat.log` (the `.all`
chat).

The iPhone 17 Pro numbers on v0.13.1 are in the
[README of tag 0.2.0](https://github.com/john-rocky/swift-litert-lm/tree/0.2.0#verified-on-device-g0).

Three device findings on v0.13.1, baked into the catalog so the API "just works":

- **Audio encoder must run on CPU.** The `.litertlm` marks the audio sections
  `section_backend_constraint: cpu`; passing a GPU audio backend makes the engine
  refuse to initialize.
- **Vision encoder must run on CPU on iOS.** The Metal GPU delegate fails to
  prepare the encoder's `STABLEHLO_COMPOSITE` op (createConversation → INTERNAL
  error); CPU/XNNPACK compiles it fine.
- **Bring towers up one at a time.** Initializing text + vision + audio + the
  speculative drafter *simultaneously* overruns the GPU weight-conversion budget
  (`std::bad_alloc`). Each tower in isolation is small (≤ ~1.6 GB), so Easy mode
  brings up only what you ask for.

**FM mode, also verified on device** (`LITERT_FM=1`, iPhone 17 Pro, LiteRT-LM v0.13.1): `respond` /
`streamResponse`, guided generation (`respond(generating:)` → a structured
`@Generable` result), tool calling (the model calls your `Tool`, FM runs it, the
result feeds back into the answer), and audio / video understanding through the
Foundation Models API. A note on audio: Gemma 4's audio tower is built for
transcription, translation, and audio understanding (per Google's docs) — not as
a conversational voice assistant — so audio is treated as an understanding task.
