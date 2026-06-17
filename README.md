# swift-litert-lm

**Run Gemma 4 (and other LiteRT-LM models) on iPhone — the easy way, and as an Apple Foundation Models backend.**

<p align="center">
  <img src="https://github.com/user-attachments/assets/91567f7d-6916-4cca-a399-3c385c82ab51" alt="LiteRTDemo on iPhone 17 Pro" width="280">
</p>

> Gemma 4 E2B running fully on-device — text, photos, microphone audio, and video, in one chat, on the Metal GPU.

This package puts two faces on Google's official [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime:

- **Easy mode** — `LiteRTChat(.gemma4_E2B)`: auto-download, memory-safe, Metal GPU, multimodal. The fastest path from zero to a Gemma 4 chat on a real device.
- **FM mode** — `LiteRTLanguageModel`: a backend for Apple's **Foundation Models** API. Drive LiteRT through `LanguageModelSession(model:)` with the full FM surface — `respond` / `streamResponse`, `@Generable` guided generation, `Tool` calling, and **image / audio / video understanding that Apple's own system model does not offer.** All device-verified on iPhone 17 Pro.

Apple opened Foundation Models to custom backends in iOS 27 and blessed exactly two on-device ones — `CoreAILanguageModel` (its own) and `MLXLanguageModel`. **LiteRT is conspicuously absent.** `LiteRTLanguageModel` makes Google's runtime the natural third first-class FM backend on iPhone.

## Quickstart (Easy mode)

```swift
import LiteRTFoundation

// Downloads the model on first run, brings up the Metal GPU backend, streams tokens.
let chat = try await LiteRTChat(.gemma4_E2B) { progress in
    print("Downloading model… \(Int(progress.fraction * 100))%")
}

for try await token in chat.stream("Explain quantum computing in one sentence.") {
    print(token, terminator: "")
}
```

Multimodal (text + image):

```swift
let chat = try await LiteRTChat(.gemma4_E2B, modalities: .textImage)
let answer = try await chat.respond("What's in this photo?", image: jpegData)
```

Text + image + audio:

```swift
let chat = try await LiteRTChat(.gemma4_E2B, modalities: .all)
let answer = try await chat.respond("Transcribe and summarize.", audio: .file(wavURL))
```

### FM mode — LiteRT as an Apple Foundation Models backend (iOS 27+)

```swift
import FoundationModels
import LiteRTFoundation

let model   = try await LiteRTLanguageModel(.gemma4_E2B)
let session = LanguageModelSession(model: model)        // Apple's exact API

// Text — respond / streamResponse.
let answer = try await session.respond(to: "Explain on-device AI in one sentence.")

// Guided generation — @Generable structured output.
let colors = try await session.respond(generating: PrimaryColors.self) { "List three primary colors." }

// Tools — the model calls your FM Tool; FM runs it and feeds the result back.
let withTool = LanguageModelSession(model: model, tools: [WeatherTool()])
let weather  = try await withTool.respond(to: "What's the weather in Tokyo?")

// Audio + video understanding — Apple's transcript has no audio/video, so LiteRT
// carries them through the custom-segment hook (a world-first through the FM API).
let spoken = try await session.respond {
    LiteRTAudioSegment(data: wavBytes)
    "Answer the question in this audio."
}
let frames = try await VideoFrameSampler.sampleFrames(from: videoURL, count: 4)
let clip   = try await session.respond {
    LiteRTVideoSegment(frames: frames)
    "Describe what happens in this video."
}
```

> **Metal GPU note:** the GPU backend only engages in an Xcode-signed app on a *physical* device. The iOS Simulator falls back to CPU.

## Sample app — `Samples/LiteRTDemo`

A clone-and-run SwiftUI chat in the ChatGPT/Claude style: message bubbles, live
token streaming, and three attachments — **photo**, **microphone audio**, and
**video** (sampled to frames). Runs Gemma 4 E2B fully on the device's Metal GPU.

Tap **FM API** in the header for a second screen that runs the *same* model
through Apple's Foundation Models API — `LanguageModelSession(model:)`,
`respond(generating:)` rendering a typed `@Generable`, and a live **tool-calling**
round-trip — so you can see on-device that LiteRT is a real FM backend, not a
lookalike. (Entering FM mode frees the chat engine and FM brings up its own, so
only one model is ever resident.)

```bash
brew install xcodegen          # if needed
cd Samples/LiteRTDemo
./fetch-test-assets.sh         # optional: pulls/generates the demo + self-test fixtures
xcodegen generate
open LiteRTDemo.xcodeproj       # set your DEVELOPMENT_TEAM, run on a device
```

The fixtures (a sample image, audio, and video) are kept out of git — the
interactive chat doesn't need them; the script provides them for the scripted
demo (`LITERT_DEMO`) and the headless self-tests below.

The same target also hosts headless self-tests for device verification (set an
env var on launch): `LITERT_G0_TEST` (text+image+audio), `LITERT_MMCHAT`
(one `.all` chat), `LITERT_BENCH` (decode speed), `LITERT_FM` (FM guided + audio
+ video + tools), `LITERT_DEMO` (auto-runs a couple of chat turns).

## What Easy mode owns for you

- **Model download** — a chunked, resumable, single-flight HTTP downloader tuned for the iPhone + Hugging Face dual-CDN path (pooled independent `URLSession`s, per-chunk wall-clock deadlines, `waitsForConnectivity = false`). No silent 0-byte stalls.
- **Memory safety** — refuses to load a model that would jetsam the app on the current device (override with `allowUnsafeMemory: true`), and caps Gemma 4's per-image visual-token budget to bound the GPU working set.
- **GPU + multimodal bring-up** — `EngineConfig(backend: .gpu, visionBackend:…, audioBackend:…)` wired correctly, with only the towers you ask for.

## Model catalog

| Model | Modalities | Size | Notes |
|---|---|---|---|
| `.gemma4_E2B` | text · image · audio | ~2.6 GB | Default hero. The E2B variant whose vision path works on iOS. |

## Why we vendor the wrapper instead of depending on the upstream package

The native runtime ships as prebuilt `xcframework`s attached to the official LiteRT-LM GitHub releases; we depend on those binary artifacts directly (they download cleanly over HTTPS). We deliberately **do not** add `google-ai-edge/litert-lm` as a SwiftPM git dependency: that repo LFS-tracks Android/Linux/Windows prebuilt libraries (`prebuilt/*/*.so`, `.dylib`, …) which are irrelevant to Apple platforms but make `swift package resolve` fragile (upstream issue #2407). Vendoring the small Apache-2.0 Swift wrapper under `Sources/LiteRTLM` keeps the on-ramp bulletproof. See `NOTICE`.

## Requirements

- iOS 16+ / macOS 13+ for Easy mode. (FM mode requires the iOS 27 SDK.)
- A physical device for Metal GPU inference.

## Verified on device (G0)

Gemma 4 E2B running on a physical **iPhone 17 Pro (iOS 27)** via the `LiteRTDemo`
self-test — text + image + audio, all working:

| Modality | Output | Decode | Prefill | TTFT | Footprint |
|---|---|---|---|---|---|
| Text (GPU) | correct one-sentence answer | **~50 tok/s** (warm; ~44 cold) | ~300–1000 tok/s | ~0.1 s (warm) | ~530 MB |
| Image (CPU encoder) | "Apple" for `apple.png` ✓ | — ¹ | ~780 tok/s | 0.6 s | ~1.57 GB |
| Audio (CPU encoder) | "Have a wonderful day." ✓ | ~53 tok/s | ~1090 tok/s | 0.1 s | ~400 MB |

¹ image decode tok/s is statistically meaningless here (one-word answer). Peak
footprint stayed at ~1.57 GB — far under the iPhone jetsam ceiling.

**On decode speed.** Decode is warm-up- and context-dependent. The *first*
generation after load is cold (~44 tok/s) while the GPU decode kernels compile;
from the second turn it settles at **~50 tok/s** on iPhone 17 Pro (peak ~53) —
in line with independent benchmarks and Google's 56.5 tok/s model-card figure.
`LiteRTChat` runs a small **prewarm** during setup (`prewarm: true`) so your
*first* message is already warm. Decode is weight-bandwidth-bound and tapers slightly as the KV
cache grows. The bundled MTP speculative-decoding drafter is **not** a useful
lever here — enabling it (`speculativeDecoding: true`) is flaky on this runtime
build ("Failed to create engine") with no measured speedup — so it's off by
default. (Probe it yourself: `LITERT_BENCH=1`.)

Three device findings, now baked into the catalog so the API "just works":

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

**FM mode, also verified on device** (`LITERT_FM=1`, iPhone 17 Pro): `respond` /
`streamResponse`, guided generation (`respond(generating:)` → a structured
`@Generable` result), tool calling (the model calls your `Tool`, FM runs it, the
result feeds back into the answer), and audio / video understanding through the
Foundation Models API. A note on audio: Gemma 4's audio tower is built for
transcription, translation, and audio understanding (per Google's docs) — not as
a conversational voice assistant — so audio is treated as an understanding task.

## Roadmap

- [x] Easy mode spine: catalog, downloader, `LiteRTChat` (text · image · audio)
- [x] **G0** — Gemma 4 E2B text + image + audio on a physical iPhone (tok/s + memory recorded above)
- [x] Easy-mode sample app (`Samples/LiteRTDemo`, clone-and-run) + headless G0 self-test
- [x] Multimodal chat verified on device: one `.all` engine handles text + image + audio in one conversation
- [x] **G1** — `LiteRTLanguageModel` / `LiteRTExecutor`: `LanguageModelSession(model:)` drives LiteRT-LM end-to-end via the real FM API (`respond` + `streamResponse`), device-verified on iPhone 17 Pro
- [x] Image through the FM API (`Transcript.AttachmentSegment` → LiteRT vision)
- [x] **Audio through the FM API** via `LiteRTAudioSegment` (`Transcript.CustomSegment`) — **device-verified** (transcribed audio through `LanguageModelSession`; a world-first)
- [x] **G2** — guided generation: `@Generable` / `GenerationSchema` over the custom executor — **device-verified** (`respond(generating:)` → structured result; schema-in-prompt + JSON extraction, hard `llguidance` is a follow-up)
- [x] **Video through the FM API** via `LiteRTVideoSegment` + `VideoFrameSampler` (app-side frames) — **device-verified**
- [x] **Tool calling** (FM `Tool` → LiteRT): the executor emits `ToolCalls` events, FM runs the app's tool, the result feeds back — **device-verified**
- [x] Chat sample: ChatGPT-style bubbles + photo / **microphone audio** / **video** attachments (device)
- [ ] Hard `llguidance`-constrained decoding (guided gen + tools are prompt-driven today)

## License

Apache-2.0. Includes software developed by Google LLC (the vendored LiteRT-LM Swift wrapper and the prebuilt runtime binaries). See `LICENSE` and `NOTICE`.
