# swift-litert-lm

**Run Gemma 4 (and other LiteRT-LM models) on iPhone — the easy way, and as an Apple Foundation Models backend.**

This package puts two faces on Google's official [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime:

- **Easy mode** — `LiteRTChat(.gemma4_E2B)`: auto-download, memory-safe, Metal GPU, multimodal. The fastest path from zero to a Gemma 4 chat on a real device.
- **FM mode** — `LiteRTLanguageModel`: a backend for Apple's **Foundation Models** API, so you can drive LiteRT through `LanguageModelSession(model:)` with `@Generable`, tools, and streaming — *plus audio and video understanding that Apple's own system model does not offer.* _(in progress; see Roadmap.)_

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

## Roadmap

- [x] Easy mode spine: catalog, downloader, `LiteRTChat` (text · image · audio)
- [ ] **G0** — Gemma 4 E2B text + image + audio on a physical iPhone with Metal GPU (record tok/s + memory)
- [ ] Easy-mode sample app (clone-and-run)
- [ ] **G1** — `LiteRTLanguageModel` / `LiteRTExecutor`: drive a non-Apple executor through `LanguageModelSession`
- [ ] Audio through the Foundation Models API via `Transcript.CustomSegment`
- [ ] Video through the Foundation Models API via app-side frame sampling
- [ ] **G2** — guided generation (`@Generable` / tools) over the custom executor

## License

Apache-2.0. Includes software developed by Google LLC (the vendored LiteRT-LM Swift wrapper and the prebuilt runtime binaries). See `LICENSE` and `NOTICE`.
