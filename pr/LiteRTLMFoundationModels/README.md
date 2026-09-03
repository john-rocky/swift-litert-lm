# LiteRTLMFoundationModels — lean Apple Foundation Models backend (upstream-PR candidate)

This is the **minimal** Apple Foundation Models backend for LiteRT-LM, carved out
of `swift-litert-lm` so it can be proposed upstream to
[`google-ai-edge/litert-lm`](https://github.com/google-ai-edge/litert-lm)'s Swift
API. It compiles against **only the LiteRT-LM core Swift wrapper** (`LiteRTLM` —
`Engine` / `Conversation` / `Message` / …) plus Apple's `FoundationModels`.

`swift build` here proves exactly that: no app dependencies, and the only core
change is the two-line `Hashable, Sendable` conformance on `Backend` /
`EngineConfig` described under Design.

## Build requirements

**Xcode 27.0 beta 5 or later.** The FoundationModels API churned between the
June 27.0 betas and beta 5:

- Beta 5 removed `Transcript.CustomSegment` (and `Transcript.Segment.custom`),
  the hook the audio / video segments rode on; those segments are gone with it.
- Beta 5 made `LanguageModelCapabilities.init` unlabeled (`init(_:)`, was
  `init(capabilities:)`).

- `GenerationOptions.SamplingMode.Kind` renamed its cases
  (`top` → `randomTopK`, `nucleus` → `randomProbabilityThreshold`) —
  source-breaking; this package uses the beta 3 names.
- `LanguageModelExecutorGenerationChannel.send` changed from a generic
  (`some Event`) to a concrete `Event` parameter — source-compatible but
  ABI-breaking, so **binaries built against earlier 27.0 betas fail to launch
  on a beta 3 OS** (dyld cannot resolve the old symbol). Rebuild anything that
  links FoundationModels.

## What it is

`LiteRTLanguageModel` conforms to the iOS 27 `LanguageModel` protocol, so a
LiteRT-LM model drives a stock `LanguageModelSession` — alongside Apple's own
conformers `SystemLanguageModel` (on-device) and `PrivateCloudComputeLanguageModel`.

```swift
import FoundationModels
import LiteRTLM
import LiteRTLMFoundationModels

let cfg     = try EngineConfig(modelPath: path, backend: .gpu)   // existing core API
let model   = LiteRTLanguageModel(engineConfig: cfg)             // <- the adapter
let session = LanguageModelSession(model: model)                 // Apple's exact API

let answer  = try await session.respond(to: "Explain on-device AI in one sentence.")
```

It provides:

- `respond` / `streamResponse` (text)
- image attachments (FM's native `AttachmentSegment`)
- `@Generable` guided generation (field guide + skeleton instance in the prompt → JSON extraction)
- `Tool` calling (emits `ToolCalls` events; FM runs the tool and re-invokes)
- `LanguageModelError.unsupportedCapability(.vision)` for an image attachment
  on a model built without a vision backend (declared capabilities are enforced,
  not silently ignored)

## Design (why it's mergeable)

- **Core-only deps, one two-line core change.** Uses just the existing public
  `Engine` / `EngineConfig` / `Conversation` / `Message` / `SamplerConfig` /
  `Backend` / `ExperimentalFlags` / `Tool` API. The single change required of the
  core is that `Backend` and `EngineConfig` become `Hashable, Sendable` — Apple's
  `LanguageModelExecutor` declares `associatedtype Configuration: Hashable & Sendable`,
  so the adapter cannot carry an `EngineConfig` without it. Both conformances are
  synthesized; no behaviour changes.
- **Zero impact on other platforms.** Everything is wrapped in
  `#if canImport(FoundationModels)` + `@available(iOS 27.0, macOS 27.0, *)`, so on
  Linux / Android / Windows it compiles to nothing. For the actual PR these files
  become a gated module/target inside the upstream Swift package (no manifest
  product needed — or an optional `LiteRTLM-FoundationModels` target if preferred).
- **Transcript bridge.** The FM API is transcript-based (each turn hands the
  executor the full conversation); LiteRT-LM is stateful. We rebuild a fresh
  LiteRT `Conversation` from the transcript per turn — correct and simple; an
  incremental KV fast-path is a later optimization.
- **One engine per distinct configuration.** FM builds one executor per session —
  a plain session and a tool-enabled session over the same model yield two — and
  each engine loads multi-GB weights, so without sharing the second session OOMs.
  `EngineCache` collapses them: `Configuration` wraps `EngineConfig` whole, so
  every engine setting flows through and equality covers all of them.
  `LiteRTLanguageModel.releaseCachedEngines()` frees them.

## Tests

`swift test` needs **no model file**: `LiteRTExecutor.init` only builds a
`LazyEngine`, and the weights are read on the first `respond`. So engine sharing,
configuration identity, and capability derivation are all observable against a
model path that does not exist — which makes them safe to run in CI.

## Non-invasive / good-citizen

Beyond keeping core changes to that one synthesized conformance, the adapter is
careful not to overstep the existing API:

- **Honors the caller's `GenerationOptions`.** `temperature`, `.greedy`,
  `.random(top:)`, and `.random(probabilityThreshold:)` are mapped to LiteRT's
  `SamplerConfig` instead of being overridden (only structured guided/tool output
  is forced near-deterministic, since it's parsed as JSON).
- **Leaves process-global `ExperimentalFlags` alone** unless the caller explicitly
  passes a `visualTokenBudget`. An app that doesn't set one sees its flags
  untouched.

One design point worth confirming with maintainers (kept explicit, not hidden):

- **`EngineCache` is a process-wide singleton.** FM constructs executors itself,
  through a *synchronous* `init(configuration:)` that receives only a `Hashable`
  value — there is no way to hand a session an already-initialized `Engine`, and
  no way to build one inside that initializer (`Engine.initialize()` is `async`).
  So the engine is deferred into a `LazyEngine` actor and shared through a store
  keyed by the configuration. A non-singleton / opt-in ownership model is
  possible, but it still has to be reachable from that synchronous init, so it
  would look like some form of keyed store.

`init(engineConfig:)` carries the caller's `EngineConfig` through verbatim,
including `cacheDir`, `loraRank`, `audioLoraRank` and `maxNumImages`. Any field
the core adds later flows through with no adapter change. The convenience
`init(modelPath:…)` is the only initializer that supplies a `cacheDir` default
(the app's Caches directory).

## Overhead vs the raw Swift API

Driving the engine through Foundation Models (`LanguageModelSession` → executor →
adapter → `Conversation`) is essentially free versus calling the core API
directly. Same model, same prompt, **greedy** decoding — deterministic, so the
output is byte-identical and the wall-time ratio is pure adapter overhead, not a
sampling difference:

| Platform | Model | Build | Overhead | Output |
|---|---|---|---|---|
| macOS (M-series) | Qwen3-0.6B | Release | +0.8% / −0.1% / +0.1% (×3) | identical |
| iPhone 17 Pro / iOS 27 | Gemma 4 E2B | Release | +1.0% | identical |

Decode runs on the same engine; the adapter only adds per-turn setup (transcript →
`Message`, conversation creation), which the optimizer reduces to noise. (Debug
builds show ~5% — that's the un-optimized glue; measure in Release.)

## What is intentionally *not* here

These stay in the `swift-litert-lm` app layer — they're conveniences, not runtime
concerns:

- the model **downloader** (Hugging Face fetch) and **model catalog**
- `LiteRTChat` (the Easy-mode chat facade)
- `VideoFrameSampler` (AVFoundation frame extraction)

## Honest scope

- Guided generation and tool calling are **prompt-driven** (a field guide plus
  a placeholder instance in the prompt — never the raw schema, which small
  models echo back — then JSON extraction), not hard constrained decoding. A
  reply that still echoes a field's schema fails with
  `LiteRTFMError.schemaEcho(field:)` instead of a vague decode error. Reliable
  for simple/medium schemas on small models; a hard `llguidance` path is future
  work.
- The lean adapter is verified for respond / guided / tools with Gemma 4 E2B on
  **both macOS** (the `fmtest` target) **and iPhone 17 Pro / iOS 27** (the sample
  app's `LITERT_LEAN_FM` self-test). Other models run through the same path but
  aren't individually verified.

## Suggested PR staging

1. **PR #1 (basic):** `respond` / `streamResponse` + image + guided + tools.
2. **Follow-up:** incremental KV fast-path;
   optional hard-constrained decoding.

## Build

```bash
swift build        # from this directory; resolves the parent package's LiteRTLM core
```
