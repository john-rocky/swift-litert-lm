# swift-litert-lm

**Run Gemma 4 (and other LiteRT-LM models) on iPhone — the easy way, and as an Apple Foundation Models backend.**

This package puts two faces on Google's official [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime:

- **Easy mode** — `LiteRTChat(.gemma4_E2B)`: auto-download, memory-safe, Metal GPU, multimodal. The fastest path from zero to a Gemma 4 chat on a real device.
- **FM mode** — `LiteRTLanguageModel`: a backend for Apple's **Foundation Models** API, so you can drive LiteRT through `LanguageModelSession(model:)` with streaming today (and `@Generable` / tools / audio / video on the roadmap) — *understanding Apple's own system model does not offer.* Text generation through the real FM API is **device-verified** (`respond` + `streamResponse` on iPhone 17 Pro).

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

FM mode — LiteRT as an Apple Foundation Models backend (iOS 27+):

```swift
import FoundationModels
import LiteRTFoundation

let model   = try await LiteRTLanguageModel(.gemma4_E2B)
let session = LanguageModelSession(model: model)        // Apple's exact API
let answer  = try await session.respond(to: "Explain on-device AI in one sentence.")
for try await snapshot in session.streamResponse(to: "Name three colors.") { /* … */ }
```

> **Metal GPU note:** the GPU backend only engages in an Xcode-signed app on a *physical* device. The iOS Simulator falls back to CPU.

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
| Text (GPU) | correct one-sentence answer | ~25–42 tok/s | ~45–64 tok/s | 0.3–0.4 s | ~530 MB |
| Image (CPU encoder) | "Apple" for `apple.png` ✓ | — ¹ | ~780 tok/s | 0.6 s | ~1.57 GB |
| Audio (CPU encoder) | "Have a wonderful day." ✓ | ~53 tok/s | ~1090 tok/s | 0.1 s | ~400 MB |

¹ image decode tok/s is statistically meaningless here (one-word answer). Peak
footprint stayed at ~1.57 GB — far under the iPhone jetsam ceiling.

**On decode speed.** Decode is warm-up- and context-dependent. The *first*
generation after load is cold (~33–44 tok/s) while the GPU decode kernels
compile; once warm it settles at **~50 tok/s** on iPhone 17 Pro — in line with
independent benchmarks and Google's 56.5 tok/s model-card figure. `LiteRTChat`
runs a small **prewarm** during setup (`prewarm: true`) so your *first* message
is already fast. Decode is weight-bandwidth-bound and tapers slightly as the KV
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

## Roadmap

- [x] Easy mode spine: catalog, downloader, `LiteRTChat` (text · image · audio)
- [x] **G0** — Gemma 4 E2B text + image + audio on a physical iPhone (tok/s + memory recorded above)
- [x] Easy-mode sample app (`Samples/LiteRTDemo`, clone-and-run) + headless G0 self-test
- [x] Multimodal chat verified on device: one `.all` engine handles text + image + audio in one conversation
- [x] **G1** — `LiteRTLanguageModel` / `LiteRTExecutor`: `LanguageModelSession(model:)` drives LiteRT-LM end-to-end via the real FM API (`respond` + `streamResponse`), device-verified on iPhone 17 Pro
- [ ] Audio through the Foundation Models API via `Transcript.CustomSegment`
- [ ] Video through the Foundation Models API via app-side frame sampling
- [ ] **G2** — guided generation (`@Generable` / tools) over the custom executor

## License

Apache-2.0. Includes software developed by Google LLC (the vendored LiteRT-LM Swift wrapper and the prebuilt runtime binaries). See `LICENSE` and `NOTICE`.
