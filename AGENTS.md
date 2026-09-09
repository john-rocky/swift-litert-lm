# AGENTS.md — for coding agents putting an open LLM into an Apple app

Read this first if someone asked you to run Gemma, a Hugging Face model, or a fine-tuned
model on an iPhone or a Mac from Swift. This file routes you to the right API and the file
that shows it; the README has the code.

**What this is.** A Swift package over Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)
runtime: one dependency line, a streaming chat on the Metal GPU, in-app model download,
text + image + audio. It is also a backend for Apple's Foundation Models API (iOS 27).
Community package, not affiliated with Google or Apple. Use the `LiteRTFoundation` product;
`LiteRTLM` is the vendored runtime wrapper, only for code that needs the engine directly.

## Route

| Task | API | Where |
|---|---|---|
| Streaming chat with Gemma 4 on a device | `LiteRTChat(.gemma4_E2B)` then `chat.stream(_:)` | README "Quickstart (Easy mode)" |
| The same with a photo, or with audio | `LiteRTChat(.gemma4_E2B, modalities: .textImage)` or `.all`, then `respond(_:image:)` / `respond(_:audio:)` | README "Quickstart (Easy mode)" |
| Tool calling, `@Generable`, or an existing `LanguageModelSession` on an open model | `LiteRTLanguageModel(.gemma4_E2B)` then `LanguageModelSession(model:)`; Apple's API is unchanged | README "FM mode" |
| Any `.litertlm` on Hugging Face, no catalog entry | `LiteRTChat(huggingFaceRepo:fileName:)` | README "Shipping the model" |
| A `.litertlm` already on disk (bundled, sideloaded, fine-tuned) | `LiteRTChat(modelFileURL:)` or `LiteRTLanguageModel(modelFileURL:)` | README "Shipping the model" |
| A fine-tuned Hugging Face model, end to end | Convert with [hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm), then load it here | [docs/recipe-hf-finetune-to-iphone.md](docs/recipe-hf-finetune-to-iphone.md), measured 2026-09-05; data in [docs/recipe.json](docs/recipe.json) |
| A complete app to copy | Chat with attachments, an FM API screen, a model picker | [Samples/LiteRTDemo](Samples/LiteRTDemo) |
| Check the package resolves before opening Xcode | `git clone https://github.com/john-rocky/swift-litert-lm && cd swift-litert-lm && swift build` | README top |

## Rules that fail on a real device when broken

1. **Pin a tag, 0.2.0 or later.** `.package(url: "https://github.com/john-rocky/swift-litert-lm", from: "0.2.0")`.
   Tag 0.1.1, and `main` before 09f04f1, do not compile for iOS with Xcode 26.1.1 or Xcode 27.0 beta 5.
2. **Metal GPU only engages in an Xcode-signed app on a physical device.** The iOS Simulator
   falls back to CPU. Do not report simulator speed as device speed.
3. **Do not commit or bundle the model.** `LiteRTChat` downloads the `.litertlm` on first launch
   into `Application Support/LiteRTModels` (resumable, single-flight, excluded from iCloud backup).
   Bundle only models of tens of MB; Gemma 4 E2B is about 2.6 GB.
4. **Audio and video do not go through the FM API.** Xcode 27 beta 5 removed the transcript hook.
   Use Easy mode (`LiteRTChat`) for audio and video; FM mode carries text and image.
5. **One turn per `LiteRTChat` for Qwen3-template models** on the pinned runtime
   ([LiteRT-LM#3443](https://github.com/google-ai-edge/LiteRT-LM/issues/3443)). Make a new
   `LiteRTChat` per turn; the recipe has the measurement and the workaround.
6. **Do not hand-write `Engine` code or depend on the upstream Swift package** when `LiteRTChat`
   covers the task. The runtime wrapper is vendored here on purpose (README "Why we vendor").
7. **Numbers stay with their device.** iPhone 17 Pro, iOS 27: about 50 tok/s decode and about
   530 MB text footprint (README "Verified on device"). Do not extrapolate to other devices.
8. Easy mode needs iOS 16+ / macOS 13+; FM mode needs the iOS 27 SDK.

## Not this package

- Android → [hfmodels-android](https://github.com/john-rocky/hfmodels-android).
- Converting a model to `.litertlm` → [hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm).
- Apple's own Core AI runtime (`.aimodel`) → [coreai-kit](https://github.com/john-rocky/coreai-kit).

Maintainer: john-rocky (GitHub). Issues: https://github.com/john-rocky/swift-litert-lm/issues
