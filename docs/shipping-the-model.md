# Shipping the model

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

**Recipe: a fine-tuned Hugging Face model → your iPhone app** — [docs/recipe-hf-finetune-to-iphone.md](recipe-hf-finetune-to-iphone.md): convert it with hf-to-litertlm, add it to an existing app, stop and release, verify (Mac-verified 2026-09-05; iPhone row pending).

**Other options, and when they fit:**

| Option | When to use |
|---|---|
| **Runtime download** (default) | Large weights like this 2.6 GB model. Recommended. |
| **Bundle in the app** | Only small models (tens of MB) — a multi-GB `.ipa` hits App Store size / cellular-download limits. Since the downloader skips when the file is already in the storage dir, a model you pre-copy there "just works" through the Easy API (or use `EngineConfig(modelPath:)` with a `Bundle.main` path directly). |
| **On-Demand Resources** | Apple-hosted, not counted in initial app size; per-tag size limits make a 2.6 GB model awkward (you'd split it). |
| **Sideload / manual** | Dev or enterprise only — push the file into the app container and point `storageDirectory` at it. |
