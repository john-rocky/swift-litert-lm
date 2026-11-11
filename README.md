# swift-litert-lm

Gemma 4 on iPhone as a Swift package. On 2026-11-05, Gemma 4 E2B decoded at about 88 tokens per second once warm, on an iPhone 18 Pro with LiteRT-LM v0.19.0. Add one dependency and your app downloads the model, then chats on the Metal GPU with text, images and audio. Other `.litertlm` models load the same way, and the package can also act as a backend for Apple's Foundation Models API on iOS 27.

<p align="center">
  <img src="https://github.com/user-attachments/assets/91567f7d-6916-4cca-a399-3c385c82ab51" alt="LiteRTDemo on iPhone 17 Pro" width="280">
</p>
<p align="center"><i>Gemma 4 E2B on an iPhone, with text, photos, microphone audio and video frames in one chat.</i></p>

This is a community package over Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime. It downloads the model, checks that the device has the memory for it and sets up the GPU. Fetching LiteRT-LM as a SwiftPM dependency needs `git-lfs`, and this package does not. It is not affiliated with Google or Apple.

## Quickstart (Easy mode)

The simplest way in is `LiteRTChat`. In Xcode, paste the repo URL into **File → Add Package Dependencies…** and add the `LiteRTFoundation` product. In a `Package.swift`:

```swift
.package(url: "https://github.com/john-rocky/swift-litert-lm", from: "0.3.1")
```

Then stream a Gemma 4 chat. The first run downloads about 2.6 GB from Hugging Face, and later runs reuse it:

```swift
import LiteRTFoundation

let chat = try await LiteRTChat(.gemma4_E2B) { progress in
    print("Downloading model… \(Int(progress.fraction * 100))%")
}

for try await token in chat.stream("Explain quantum computing in one sentence.") {
    print(token, terminator: "")
}
```

Run it on a physical device with at least 7 GB of RAM. Below that, `LiteRTChat` refuses to load the model instead of letting iOS kill the app, and the iOS Simulator falls back to the CPU.

On an iPhone 18 Pro with iOS 27.2 and LiteRT-LM v0.19.0, Gemma 4 E2B decodes at about 88 tok/s once warm.

For a photo, or for text, image and audio together:

```swift
let chat = try await LiteRTChat(.gemma4_E2B, modalities: .textImage)
let answer = try await chat.respond("What's in this photo?", image: jpegData)
```

```swift
let chat = try await LiteRTChat(.gemma4_E2B, modalities: .all)
let answer = try await chat.respond("Transcribe and summarize.", audio: .file(wavURL))
```

Video goes in as frames. `VideoFrameSampler` samples them, and `respond` takes them as `images:`.

## FM mode: a backend for Apple's Foundation Models API

Apple opened Foundation Models to custom backends in iOS 27. `LiteRTLanguageModel` drives LiteRT-LM through `LanguageModelSession(model:)`, with `@Generable` output, tool calling and image attachments:

```swift
import FoundationModels
import LiteRTFoundation

let model   = try await LiteRTLanguageModel(.gemma4_E2B)
let session = LanguageModelSession(model: model)

// Plain text.
let answer = try await session.respond(to: "Explain on-device AI in one sentence.")

// Structured output. PrimaryColors is your own @Generable type.
let colors = try await session.respond(generating: PrimaryColors.self) {
    "List three primary colors."
}

// Tools. WeatherTool is your own Tool, and the framework runs it for the model.
let withTool = LanguageModelSession(model: model, tools: [WeatherTool()])
let weather  = try await withTool.respond(to: "What's the weather in Tokyo?")
```

Audio and video only work in Easy mode, because the Foundation Models API has no way to pass them.

Before iOS 27, [anylanguagemodel-litert-lm](https://github.com/john-rocky/anylanguagemodel-litert-lm) gives you the same `LanguageModelSession` API on iOS 17+ or macOS 14+, through [AnyLanguageModel](https://github.com/huggingface/AnyLanguageModel).

Google [merged this adapter](https://github.com/google-ai-edge/LiteRT-LM/commit/c765ade83170d9ea10d19a73f8f5998e94a85cfd) into [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM/blob/d17a52fd5c2b4ce1280959309af175caebaac2c3/swift/apple_fm/LiteRTLanguageModel.swift) in July 2026.

## Sample app

`Samples/LiteRTDemo` is a SwiftUI chat with photo, microphone and video attachments. Its **FM API** screen runs the same model through Foundation Models, including a live tool call:

<p align="center">
  <img src="https://github.com/user-attachments/assets/0a25f1a0-f173-473a-9cbf-2ce3e987a308" alt="Tool calling through the Foundation Models API, on-device" width="280">
</p>

```bash
cd Samples/LiteRTDemo
open LiteRTDemo.xcodeproj       # set your DEVELOPMENT_TEAM, run on a device
```

## Shipping the model

Gemma 4 E2B is about 2.6 GB, so don't commit it or bundle it in your app. By default, the package downloads the `.litertlm` on first launch into `Application Support/LiteRTModels`, excluded from iCloud backup, and reuses it afterward.

Any `.litertlm` on Hugging Face works:

```swift
let chat = try await LiteRTChat(
    huggingFaceRepo: "litert-community/gemma-4-E4B-it-litert-lm",
    fileName: "gemma-4-E4B-it.litertlm")   // text-only by default; pass modalities: .all if it has them
```

To load a file that is already on the device:

```swift
// Easy mode.
let chat = try await LiteRTChat(modelFileURL: url, modalities: .all)

// FM mode.
let model = try LiteRTLanguageModel(modelFileURL: url)
let session = LanguageModelSession(model: model)
```

## Requirements

- iOS 16+ / macOS 13+ for Easy mode. FM mode requires the iOS 27 SDK.
- An iPhone or a Mac for Metal GPU inference, not the Simulator.
- No memory entitlement: the sample app has none.
- Version 0.3.1 runs LiteRT-LM v0.19.1. Version 0.3.0 runs LiteRT-LM v0.19.0.

## Documentation

- [Fine-tuned model to iPhone](docs/recipe-hf-finetune-to-iphone.md): convert with hf-to-litertlm, add it to an app, verify
- [Other ways to ship the model](docs/shipping-the-model.md): your own host, bundling, On-Demand Resources
- [LiteRT-LM v0.19.0](docs/litert-lm-v0.19.0.md): what changes for an app, and what to do after a turn throws
- [Verified on device](docs/verified-on-device.md): the full table and how it was measured
- [Sample app details](docs/sample-app.md): self-tests and the demo-video mode
- [What Easy mode handles](docs/easy-mode.md): the downloader, memory checks and the model catalog
- [Why the wrapper is vendored](docs/why-vendor.md)
- [Roadmap](docs/roadmap.md)

## Projects using swift-litert-lm

- [VivaDicta](https://github.com/n0an/VivaDicta): speech-to-text for iOS and watchOS, with Gemma 4 on-device from [a fork of this package](https://github.com/n0an/VivaDicta/blob/v3.12.0/Modules/LocalLLM/Sources/LocalLLM/LiteRTModelManager.swift)
- [EyePal](https://github.com/yyyug/EyePal): an assistive iOS app that [captions images and answers questions about them](https://github.com/yyyug/EyePal/blob/97dec77adafbfe629c6abfaa60e490336a5a9fc1/EyePal/Services/Gemma/GemmaTextRecognitionService.swift) for people with visual impairments

## License

Apache-2.0. Includes software developed by Google LLC (the vendored LiteRT-LM Swift wrapper and the prebuilt runtime binaries). See `LICENSE` and `NOTICE`. Each model keeps its own license.
