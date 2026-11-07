# swift-litert-lm

**Gemma 4 on iPhone as a Swift package** — Metal GPU, text + image + audio, in-app model download. Also a backend for Apple's Foundation Models API (iOS 27).

One dependency:

```swift
.package(url: "https://github.com/john-rocky/swift-litert-lm", from: "0.3.0")   // product: LiteRTFoundation
```

This runs LiteRT-LM v0.19.0 (C API 1.0.0). Tag 0.2.0 runs LiteRT-LM v0.15.0. What v0.19.0 changes for an app is in [LiteRT-LM v0.19.0](#litert-lm-v0190).

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

Measured on an iPhone 18 Pro (iOS 27.2) with LiteRT-LM v0.19.0: ~88 tok/s decode (warm), ~507 MB footprint for `LiteRTChat(.gemma4_E2B)` ([full table](#verified-on-device-g0)).

<p align="center">
  <img src="https://github.com/user-attachments/assets/91567f7d-6916-4cca-a399-3c385c82ab51" alt="LiteRTDemo on iPhone 17 Pro" width="280">
</p>

> Gemma 4 E2B running fully on-device — text, photos, microphone audio, and video, in one chat, on the Metal GPU.

This package puts two faces on Google's official [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime:

- **Easy mode** — `LiteRTChat(.gemma4_E2B)`: auto-download, memory-safe, Metal GPU, multimodal. The fastest path from zero to a Gemma 4 chat on a real device.
- **FM mode** — `LiteRTLanguageModel`: a backend for Apple's **Foundation Models** API. Drive LiteRT through `LanguageModelSession(model:)` with the full FM surface — `respond` / `streamResponse`, `@Generable` guided generation, `Tool` calling, and **image / audio / video understanding that Apple's own system model does not offer.** All device-verified on iPhone 17 Pro.

Apple opened Foundation Models to custom backends in iOS 27 — any type conforming to its `LanguageModel` protocol can drive a `LanguageModelSession`. Apple's own conformers are `SystemLanguageModel` (on-device) and `PrivateCloudComputeLanguageModel` (Private Cloud Compute). `LiteRTLanguageModel` adds Google's runtime as a third-party **on-device** backend — Gemma 4 today, and any `.litertlm` you bring.

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

```

> **Audio / video through the FM API — removed in Xcode 27 beta 5.** Apple's
> transcript has no audio/video segment; earlier 27.0 betas exposed a
> `Transcript.CustomSegment` hook that `LiteRTAudioSegment` / `LiteRTVideoSegment`
> rode on, and beta 5 dropped it from the SDK. Audio and video stay available in
> Easy mode (`LiteRTChat`); the FM path carries text and image attachments.

> **Metal GPU note:** the GPU backend only engages in an Xcode-signed app on a *physical* device. The iOS Simulator falls back to CPU.

> **AnyLanguageModel instead of Foundation Models:** [anylanguagemodel-litert-lm](https://github.com/john-rocky/anylanguagemodel-litert-lm) puts this package's runtime behind [AnyLanguageModel](https://github.com/huggingface/AnyLanguageModel)'s `LanguageModelSession` for apps that target iOS 17+ or macOS 14+, and its `@Generable` output goes through LiteRT-LM's constrained decoding.

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

<p align="center">
  <img src="https://github.com/user-attachments/assets/0a25f1a0-f173-473a-9cbf-2ce3e987a308" alt="Tool calling through the Foundation Models API, on-device" width="280">
</p>

> Tool calling on-device: the model extracts the place from your prompt, your
> Swift function calls Apple's CoreLocation (`CLGeocoder`), and the real
> coordinates land on a MapKit map — orchestrated by the Foundation Models
> runtime over the LiteRT backend.

```bash
cd Samples/LiteRTDemo
open LiteRTDemo.xcodeproj       # set your DEVELOPMENT_TEAM, run on a device
```

The `.xcodeproj` is committed, so it's clone-and-run. It's generated from
`project.yml` by [xcodegen](https://github.com/yonaskolb/XcodeGen); regenerate
after editing the spec with `xcodegen generate`.

The fixtures (a sample image, audio, and video) are kept out of git — the
interactive chat doesn't need them. For the scripted demo (`LITERT_DEMO`) and
the headless self-tests below, fetch them and re-add them to the project:

```bash
brew install xcodegen           # if needed
./fetch-test-assets.sh          # pulls / generates the fixtures
xcodegen generate               # re-adds them to the project
```

The same target also hosts headless self-tests for device verification (set an
env var on launch): `LITERT_G0_TEST` (text+image+audio), `LITERT_MMCHAT`
(one `.all` chat), `LITERT_BENCH` (decode speed), `LITERT_FM` (FM images + guided
+ tools), `LITERT_DEMO` (auto-runs a couple of chat turns),
`LITERT_FAILURES` (missing or broken model file, RAM check, over-long input, cancel;
no fixtures needed), `LITERT_CHAT_CHECK` (a two-turn chat; no fixtures needed),
`LITERT_UI_CHECK` (an over-long input, then a short turn, on the chat screen; no fixtures needed).

### One-take showcase (the demo-video mode)

An auto-playing, screen-recording-oriented tour driven entirely through the
Foundation Models API: live token streaming with a decode tok/s badge, vision,
an agent scene where ONE request has Gemma chain four real iOS tools by itself
(UIDevice battery → a real notification banner → an ActivityKit Live Activity →
AVSpeechSynthesizer), and guided generation rendered straight into Swift
Charts. It ends by sending you Home, where Gemma's countdown ticks in the
Dynamic Island.

To run it: build the app on a device, launch once online (the Gemma 4 E2B
model downloads, and the notification permission prompt gets out of the way),
then turn ON Airplane Mode and go **FM API → Record a demo** — two takes are
available (`offline`, and an "evening run" `story` take). Tap Start; a 3-2-1
countdown leads in and everything after is automatic. `LITERT_SHOWCASE=1`
(or `=story`) boots straight into a take.

Recording notes: capturing via QuickTime (File → New Movie Recording, camera +
microphone = the iPhone) keeps the Dynamic Island visible and records the
spoken cheer into the file — but iOS paints its canonical status bar (9:41,
full bars) over captured video, so crop it in the edit if that matters. An
on-device screen recording shows the real status bar instead, but its
indicator suppresses the app's own Live Activity in the island. Optional:
`./fetch-test-assets.sh` pulls a `showcase.jpg` for the story take's vision
scene (any photo you drop at `Resources/showcase.jpg` wins).

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

## What Easy mode owns for you

- **Model download** — a chunked, resumable, single-flight HTTP downloader tuned for the iPhone + Hugging Face dual-CDN path (pooled independent `URLSession`s, per-chunk wall-clock deadlines, `waitsForConnectivity = false`). No silent 0-byte stalls.
- **Memory safety** — refuses to load a model that would jetsam the app on the current device (override with `allowUnsafeMemory: true`), and caps Gemma 4's per-image visual-token budget to bound the GPU working set.
- **GPU + multimodal bring-up** — `EngineConfig(backend: .gpu, visionBackend:…, audioBackend:…)` wired correctly, with only the towers you ask for.

## Model catalog

| Model | Modalities | Size | Notes |
|---|---|---|---|
| `.gemma4_E2B` | text · image · audio | ~2.6 GB | Default hero. The E2B variant whose vision path works on iOS. |

## Shipping the model

The model is **not** committed to git or baked into the app. By default
`LiteRTChat` / `LiteRTLanguageModel` **download the `.litertlm` on first launch**
from the catalog's Hugging Face `resolve` URL, store it in
`Application Support/LiteRTModels` (persistent, excluded from iCloud backup), and
reuse it afterward. The bundled downloader is chunked, resumable, single-flight,
and idempotent — an already-present file is reused, never re-fetched. For
multi-GB weights this is the right default: a small app binary, App Store
friendly, and you can ship a new model without an app update.

**Use your own model** — add a `LiteRTModel` catalog case (Hugging Face repo,
file name, approximate size, minimum RAM) and call `LiteRTChat(.yourModel)`; the
downloader does the rest. Point `downloadURL` at any host (your own CDN / S3) if
you don't use Hugging Face.

**Run any Hugging Face `.litertlm`** — point at any community repo (no catalog
entry); it downloads on first use:

```swift
let chat = try await LiteRTChat(
    huggingFaceRepo: "litert-community/gemma-4-E4B-it-litert-lm",
    fileName: "gemma-4-E4B-it.litertlm")   // text-only by default; pass modalities: .all if it has them
```

**Load a local file directly** — for swapping your own (e.g. fine-tuned) models
in and out while experimenting, skip the catalog and download entirely:

```swift
// Easy mode — any on-disk .litertlm (bundled, pushed via devicectl, or imported
// through the Files app). No download.
let chat = try await LiteRTChat(modelFileURL: url, modalities: .all)

// FM mode — same, as a Foundation Models backend.
let model = try LiteRTLanguageModel(modelFileURL: url)
let session = LanguageModelSession(model: model)
```

The sample app has a **model picker** in the header that exercises all three —
the bundled Gemma 4 E2B, any Hugging Face repo, or a local file.

**Recipe: a fine-tuned Hugging Face model → your iPhone app** — [docs/recipe-hf-finetune-to-iphone.md](docs/recipe-hf-finetune-to-iphone.md): convert it with hf-to-litertlm, add it to an existing app, stop and release, verify (Mac-verified 2026-09-05; iPhone row pending).

**Other options, and when they fit:**

| Option | When to use |
|---|---|
| **Runtime download** (default) | Large weights like this 2.6 GB model. Recommended. |
| **Bundle in the app** | Only small models (tens of MB) — a multi-GB `.ipa` hits App Store size / cellular-download limits. Since the downloader skips when the file is already in the storage dir, a model you pre-copy there "just works" through the Easy API (or use `EngineConfig(modelPath:)` with a `Bundle.main` path directly). |
| **On-Demand Resources** | Apple-hosted, not counted in initial app size; per-tag size limits make a 2.6 GB model awkward (you'd split it). |
| **Sideload / manual** | Dev or enterprise only — push the file into the app container and point `storageDirectory` at it. |

## Why we vendor the wrapper instead of depending on the upstream package

The native runtime ships as prebuilt `xcframework`s attached to the official LiteRT-LM GitHub releases; we depend on those binary artifacts directly (they download cleanly over HTTPS). We deliberately **do not** add `google-ai-edge/litert-lm` as a SwiftPM git dependency: that repo LFS-tracks Android/Linux/Windows prebuilt libraries (`prebuilt/*/*.so`, `.dylib`, …) which are irrelevant to Apple platforms but make `swift package resolve` fragile (upstream issue #2407). Vendoring the small Apache-2.0 Swift wrapper under `Sources/LiteRTLM` keeps the on-ramp bulletproof. See `NOTICE`.

Xcode 27.0 stops with `Git LFS is not installed` when it fetches LiteRT-LM as a SwiftPM git dependency and `git-lfs` is not on Xcode's PATH. This package fetches the release assets instead, so it builds without `git-lfs`.

## Requirements

- iOS 16+ / macOS 13+ for Easy mode. (FM mode requires the iOS 27 SDK.)
- A physical device for Metal GPU inference.

## LiteRT-LM v0.19.0

On a Mac, Gemma 4 E2B on v0.19.0 answered in Easy mode with text, with two images in one chat, and
with audio. In FM mode it answered a two-turn session, a `@Generable` request, a tool call and image
attachments.

On an iPhone 18 Pro (iOS 27.2) it answered in Easy mode a two-turn chat and one chat with text, an
image and audio. A missing model file, three broken files, the RAM check and an over-long input each
threw an error with a message, and a new chat answered afterwards. In FM mode a
`LanguageModelSession` answered a `@Generable` request, a tool call, image attachments, and a second
turn after an image (6 of 6 runs) and after text (3 of 3), in Debug and Release builds. A new session
answered after a cancelled stream (2 of 2).

These second turns, and the new session after a cancelled stream, depend on a patch to the vendored
wrapper (listed in `NOTICE`). Without it, the same runs had not returned when they were stopped, one
to seven minutes in. On a Mac, with a delay added to reproduce it, the turn came back after the
runtime's 10-minute timeout. Tag 0.3.0 carries the patch. Tag 0.2.0 does not carry the patch: on v0.15.0 the one iPhone run
returned, and the same delay on a Mac stopped it.

**After a turn throws, make a new chat.** Once a turn has thrown, or after `cancel()`, later turns
on that `LiteRTChat` throw too. Release it, then create a new `LiteRTChat`: each one holds its own
engine. The sample app does this in `send()`. On a Mac with LiteRT-LM v0.15.0 and v0.19.0, three more
turns threw in each case. On an iPhone 18 Pro (iOS 27.2) with v0.19.0, the next turn threw in each
case and a new chat answered.

**A string-only chat template can lose the prompt, with no error.** On v0.19.0,
`qwen3_0_6b_mixed_int4.litertlm` from
[litert-community/Qwen3-0.6B](https://huggingface.co/litert-community/Qwen3-0.6B) renders an empty
user turn, and the model answers something unrelated. Its template sets the content to an empty
string unless `message.content is string`, and v0.19.0 passes the content as typed parts. v0.15.0
renders the prompt from the same file. `Qwen3-0.6B.litertlm` and
`Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm` from that repo answered correctly on v0.19.0. All
measured on a Mac. If a bundle answers something unrelated, read its embedded template first.

**Two names need a module prefix.** In a file with `import FoundationModels` and
`import LiteRTFoundation`, write `FoundationModels.Tool`: plain `Tool` is ambiguous with
`LiteRTLM.Tool`. The v0.19.0 wrapper adds a `Modality` enum of its own, so a file that also has
`import LiteRTLM` must write `LiteRTFoundation.Modality`. With `import LiteRTFoundation` alone, plain
`Modality` works.

## Verified on device (G0)

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
[LiteRT-LM v0.19.0](#litert-lm-v0190). The text row of the first table ran on both sides of it, at
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

## Roadmap

- [x] Easy mode spine: catalog, downloader, `LiteRTChat` (text · image · audio)
- [x] **G0** — Gemma 4 E2B text + image + audio on a physical iPhone (tok/s + memory recorded above)
- [x] Easy-mode sample app (`Samples/LiteRTDemo`, clone-and-run) + headless G0 self-test
- [x] Multimodal chat verified on device: one `.all` engine handles text + image + audio in one conversation
- [x] **G1** — `LiteRTLanguageModel` / `LiteRTExecutor`: `LanguageModelSession(model:)` drives LiteRT-LM end-to-end via the real FM API (`respond` + `streamResponse`), device-verified on iPhone 17 Pro
- [x] Image through the FM API (`Transcript.AttachmentSegment` → LiteRT vision)
- [~] **Audio through the FM API** via `LiteRTAudioSegment` (`Transcript.CustomSegment`) — was device-verified on earlier 27.0 betas; **withdrawn** when Xcode 27 beta 5 removed `Transcript.CustomSegment` from the SDK
- [x] **G2** — guided generation: `@Generable` / `GenerationSchema` over the custom executor — **device-verified** (`respond(generating:)` → structured result; field guide + skeleton instance in the prompt, JSON extraction, schema echo fails loudly; hard `llguidance` is a follow-up)
- [~] **Video through the FM API** via `LiteRTVideoSegment` + `VideoFrameSampler` (app-side frames) — same: withdrawn with beta 5 (frames still work through `LiteRTChat`)
- [x] **Tool calling** (FM `Tool` → LiteRT): the executor emits `ToolCalls` events, FM runs the app's tool, the result feeds back — **device-verified**
- [x] Chat sample: ChatGPT-style bubbles + photo / **microphone audio** / **video** attachments (device)
- [ ] Hard `llguidance`-constrained decoding (guided gen + tools are prompt-driven today)

## License

Apache-2.0. Includes software developed by Google LLC (the vendored LiteRT-LM Swift wrapper and the prebuilt runtime binaries). See `LICENSE` and `NOTICE`.
