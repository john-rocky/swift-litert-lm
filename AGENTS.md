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

1. **Pin a tag, 0.3.0 or later.** `.package(url: "https://github.com/john-rocky/swift-litert-lm", from: "0.3.0")`.
   Tag 0.1.1, and `main` before 09f04f1, do not compile for iOS with Xcode 26.1.1 or Xcode 27.0 beta 5.
   Tag 0.2.0 runs LiteRT-LM v0.15.0; tag 0.3.0 runs v0.19.0 (C API 1.0.0).
2. **Metal GPU only engages in an Xcode-signed app on a physical device.** The iOS Simulator
   falls back to CPU. Do not report simulator speed as device speed.
3. **Do not commit or bundle the model.** `LiteRTChat` downloads the `.litertlm` on first launch
   into `Application Support/LiteRTModels` (resumable, single-flight, excluded from iCloud backup).
   Bundle only models of tens of MB; Gemma 4 E2B is about 2.6 GB.
4. **Audio and video do not go through the FM API.** Xcode 27 beta 5 removed the transcript hook.
   Use Easy mode (`LiteRTChat`) for audio and video; FM mode carries text and image.
5. **One turn per `LiteRTChat` for a bundle that embeds the Hugging Face Qwen3 template**
   ([LiteRT-LM#3443](https://github.com/google-ai-edge/LiteRT-LM/issues/3443)). On a Mac, the second
   message on the recipe's bundle fails on v0.15.0 and on v0.19.0. The litert-community files
   `Qwen3-0.6B.litertlm` and `Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm` take a second turn on v0.19.0.
   Make a new `LiteRTChat` per turn; the recipe has the measurement and the workaround.
6. **Do not hand-write `Engine` code or depend on the upstream Swift package** when `LiteRTChat`
   covers the task. The runtime wrapper is vendored here on purpose (README "Why we vendor"), and
   `Sources/LiteRTLM/Conversation.swift` carries a local patch (`NOTICE`): on an iPhone 18 Pro
   (iOS 27.2) with v0.19.0, an FM session answered a second turn with it, and without it had not
   returned when the runs were stopped, one to seven minutes in (README "LiteRT-LM v0.19.0").
7. **Numbers stay with their device and runtime.** iPhone 18 Pro, iOS 27.2, LiteRT-LM v0.19.0: about
   88 tok/s decode (warm) and about 507 MB footprint for `LiteRTChat(.gemma4_E2B)` (README "Verified
   on device"). Do not extrapolate to other devices or runtime versions.
8. Easy mode needs iOS 16+ / macOS 13+; FM mode needs the iOS 27 SDK.
9. **After a turn throws, or after `cancel()`, make a new `LiteRTChat`.** Later turns on that chat
   throw too. Release the old one first: each `LiteRTChat` holds its own engine. On a Mac with v0.15.0
   and v0.19.0, three more turns threw in each case. On an iPhone 18 Pro (iOS 27.2) with v0.19.0, the
   next turn threw in each case and a new chat answered.
10. **A string-only chat template can lose the prompt on v0.19.0, with no error.** If a bundle
    answers something unrelated, read its embedded template for `message.content is string`
    (README "LiteRT-LM v0.19.0").
11. **Two names need a module prefix.** With `import FoundationModels` and `import LiteRTFoundation`,
    write `FoundationModels.Tool`. With `import LiteRTFoundation` and `import LiteRTLM`, write
    `LiteRTFoundation.Modality`; with `import LiteRTFoundation` alone, plain `Modality` works.

## Not this package

- Android → [hfmodels-android](https://github.com/john-rocky/hfmodels-android).
- Converting a model to `.litertlm` → [hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm).
- Apple's own Core AI runtime (`.aimodel`) → [coreai-kit](https://github.com/john-rocky/coreai-kit).

Maintainer: john-rocky (GitHub). Issues: https://github.com/john-rocky/swift-litert-lm/issues
