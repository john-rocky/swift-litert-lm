# What Easy mode handles

This package puts two faces on Google's official [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime:

- **Easy mode** — `LiteRTChat(.gemma4_E2B)`: auto-download, memory-safe, Metal GPU, multimodal. The fastest path from zero to a Gemma 4 chat on a real device.
- **FM mode** — `LiteRTLanguageModel`: a backend for Apple's **Foundation Models** API. Drive LiteRT through `LanguageModelSession(model:)` with the full FM surface — `respond` / `streamResponse`, `@Generable` guided generation, `Tool` calling, and **image / audio / video understanding that Apple's own system model does not offer.** All device-verified on iPhone 17 Pro.

Apple opened Foundation Models to custom backends in iOS 27 — any type conforming to its `LanguageModel` protocol can drive a `LanguageModelSession`. Apple's own conformers are `SystemLanguageModel` (on-device) and `PrivateCloudComputeLanguageModel` (Private Cloud Compute). `LiteRTLanguageModel` adds Google's runtime as a third-party **on-device** backend — Gemma 4 today, and any `.litertlm` you bring.

## First run

One dependency:

```swift
.package(url: "https://github.com/john-rocky/swift-litert-lm", from: "0.3.0")   // product: LiteRTFoundation
```

This runs LiteRT-LM v0.19.0 (C API 1.0.0). Tag 0.2.0 runs LiteRT-LM v0.15.0. What v0.19.0 changes for an app is in [LiteRT-LM v0.19.0](litert-lm-v0.19.0.md).

Five lines to a streaming Gemma 4 chat:

```swift
import LiteRTFoundation

let chat = try await LiteRTChat(.gemma4_E2B)   // first run downloads ~2.6 GB, later runs reuse it
for try await token in chat.stream("Explain quantum computing in one sentence.") {
    print(token, terminator: "")
}
```

Check that it resolves and builds on your Mac before you open Xcode:

```bash
git clone https://github.com/john-rocky/swift-litert-lm && cd swift-litert-lm && swift build
```

Measured on an iPhone 18 Pro (iOS 27.2) with LiteRT-LM v0.19.0: ~88 tok/s decode (warm), ~507 MB footprint for `LiteRTChat(.gemma4_E2B)` ([full table](verified-on-device.md)).

## Quickstart, in full

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

```

> **Audio / video through the FM API — removed in Xcode 27 beta 5.** Apple's
> transcript has no audio/video segment; earlier 27.0 betas exposed a
> `Transcript.CustomSegment` hook that `LiteRTAudioSegment` / `LiteRTVideoSegment`
> rode on, and beta 5 dropped it from the SDK. Audio and video stay available in
> Easy mode (`LiteRTChat`); the FM path carries text and image attachments.

> **Metal GPU note:** the GPU backend only engages in an Xcode-signed app on a *physical* device. The iOS Simulator falls back to CPU.

> **AnyLanguageModel instead of Foundation Models:** [anylanguagemodel-litert-lm](https://github.com/john-rocky/anylanguagemodel-litert-lm) puts this package's runtime behind [AnyLanguageModel](https://github.com/huggingface/AnyLanguageModel)'s `LanguageModelSession` for apps that target iOS 17+ or macOS 14+, and its `@Generable` output goes through LiteRT-LM's constrained decoding.

## What Easy mode owns for you

- **Model download** — a chunked, resumable, single-flight HTTP downloader tuned for the iPhone + Hugging Face dual-CDN path (pooled independent `URLSession`s, per-chunk wall-clock deadlines, `waitsForConnectivity = false`). No silent 0-byte stalls.
- **Memory safety** — refuses to load a model that would jetsam the app on the current device (override with `allowUnsafeMemory: true`), and caps Gemma 4's per-image visual-token budget to bound the GPU working set.
- **GPU + multimodal bring-up** — `EngineConfig(backend: .gpu, visionBackend:…, audioBackend:…)` wired correctly, with only the towers you ask for.

## Model catalog

| Model | Modalities | Size | Notes |
|---|---|---|---|
| `.gemma4_E2B` | text · image · audio | ~2.6 GB | Default hero. The E2B variant whose vision path works on iOS. |

## Projects using swift-litert-lm

- [VivaDicta](https://github.com/n0an/VivaDicta) — an iOS and watchOS
  speech-to-text app. Version 3.12.0 runs Gemma 4 E2B and E4B on-device through
  `LiteRTChat`, from a fork of this package
  ([source](https://github.com/n0an/VivaDicta/blob/v3.12.0/Modules/LocalLLM/Sources/LocalLLM/LiteRTModelManager.swift)).
- [EyePal](https://github.com/yyyug/EyePal) — an iOS assistive app for people
  with visual impairments. It runs Gemma on-device through this package's
  `Engine` and `Conversation` to caption images and answer questions about them
  ([source](https://github.com/yyyug/EyePal/blob/97dec77adafbfe629c6abfaa60e490336a5a9fc1/EyePal/Services/Gemma/GemmaTextRecognitionService.swift)).

Separately, the Foundation Models adapter, `LiteRTLanguageModel`, was ported
into Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) under
`swift/apple_fm/`
([source](https://github.com/google-ai-edge/LiteRT-LM/blob/d17a52fd5c2b4ce1280959309af175caebaac2c3/swift/apple_fm/LiteRTLanguageModel.swift)).
