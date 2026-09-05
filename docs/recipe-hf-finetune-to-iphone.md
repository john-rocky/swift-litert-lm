# How do I run a fine-tuned Hugging Face model on iPhone?

**Short answer.** Convert the fine-tune to one `.litertlm` file with
[hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm) (`python scripts/convert.py <org>/<model>`),
host that file where your app can download it, and load it with `LiteRTChat` from this package.
The converter refuses what it cannot convert honestly and gates what it converts with 8 fixed questions.
The representative fine-tune below, a full fine-tune of Qwen3-0.6B, converted in 217 s, passed the gate 7/8,
and answered through this package's `LiteRTChat` on a Mac Studio (M4 Max) at 143 tok/s on the GPU.
Not yet run on an iPhone (row pending). The converted bundle is published as
[mlboydaisuke/Qwen3-0.6B-Code-Expert-LiteRT](https://huggingface.co/mlboydaisuke/Qwen3-0.6B-Code-Expert-LiteRT),
so the verify command in section 6 needs no conversion. Everything here was run on 2026-09-05; the
same facts in machine-readable form are in [`recipe.json`](recipe.json).

Two limits, up front:

- **One turn per `LiteRTChat` for Qwen3-template models today.** A second message on the same
  conversation fails on the runtime this package pins (v0.15.0); upstream issue
  [google-ai-edge/LiteRT-LM#3443](https://github.com/google-ai-edge/LiteRT-LM/issues/3443). Section 5
  has the measurement and the workaround (a new `LiteRTChat` per turn).
- **Use tag `0.2.0` or later of this package.** Tag `0.1.1`, and `main` before 09f04f1, fail to
  compile for iOS with Xcode 26.1.1 and with Xcode 27.0 beta 5. `0.2.0` (2026-09-05) compiles with
  both. Section 7 has the build matrix.

## 1. Which models this works for

The converter routes by `config.json` `model_type`. Quoted from the hf-to-litertlm README at
[5ffe9ee](https://github.com/john-rocky/hf-to-litertlm/blob/5ffe9ee/README.md#convert-a-finetune)
(Hub derivative counts recounted there on 2026-08-26):

| base family | bases | Hub finetunes + adapters | toolchain | path |
|---|---|---:|---|---|
| any dense arch the stock exporter handles | llama 3.x, qwen 2/2.5/3, smollm3, olmo2, phi, ministral, … | open-ended | default stack | stock export |
| MiniCPM5 | 1B | 53 + 54 | default stack | stock export (plain llama rail) |
| granite-4.1 (dense) | 3b | 20 + 15 | default stack | stock export; spurious-BOS guard fires (bos == eos) |
| Hy-MT2 (hunyuan_v1_dense) | 1.8B | 10 + 1 | default stack | stock export after a bitwise-equal rope bake; duplicate-BOS guard fires |
| Qwen3.5 | 0.8B / 2B / 4B | 1,214 + 928 | litert-torch *main* | stock export; CPU gate |
| LFM2.5 | 350M / 1.2B / 2.6B | 210 + 94 | released 0.9.3/0.9.4 | stock export + ExecutorMetadata retrofit |
| granite-4.0-h | 350m / 1b | 24 + 4 | pinned checkout | family recipe (`HYBRID_RECIPE`) |
| Falcon-H1 | 0.5B / 1.5B / 1.5B-Deep / 3B | 14 + 2 | pinned checkout | family recipe (`HYBRID_RECIPE`) |
| Zamba2 | 1.2B / 2.7B | 3 + 0; the only real one is pre-port-serialized | pinned checkout | routed; pre-port checkpoints are refused with re-serialization instructions |
| Nemotron-H | Nemotron-H-4B / Nemotron-3-Nano-4B | 19 + 14 | pinned checkout | family recipe (`HYBRID_RECIPE`) |

LoRA/PEFT adapter repos are merged into their base first. Refused at the entry gate, with a JSON
reason and exit code 2: gated repos, architectures that live in repo code (`auto_map` with a
`model_type` transformers does not register), pre-quantized weights (GPTQ, AWQ, bitsandbytes), and
pre-port Zamba2 checkpoints. Not covered by this recipe: mixture-of-experts models, diffusion
language models, and encoder-only or classifier heads (they are not text-generation bundles).
Vision-language derivatives use a separate script in the same repository.

For a phone, size decides more than architecture. The converter's default is int8 weights, so a
bundle is about one byte per parameter plus the tokenizer: 613 MB for the 596M-parameter model
below. Models of 3B parameters and more get the embedding split out so the main section stays under
the iOS mmap limit; that path is not exercised here.

### The representative fine-tune

Chosen with these filters: at most 1B parameters, not gated, Apache-2.0 or MIT, dense, at least
1,000 downloads in the last 30 days, and someone's fine-tune rather than a mirror or a re-upload.
Candidates from the Hub's `base_model:finetune:` filter over the small Qwen, SmolLM2, OLMo-2 and
sarashina bases, 2026-09-05:

| repo | base | license | 30-day downloads | decision |
|---|---|---|---:|---|
| `suayptalha/Qwen3-0.6B-Code-Expert` | Qwen/Qwen3-0.6B | apache-2.0 (card metadata; the card's LICENSE link points at a file the repo does not contain) | 1,431 | **chosen**: full fine-tune on a named public dataset (`nvidia/OpenCodeReasoning`), stock qwen3 config, unchanged since 2025-05 |
| `reaperdoesntknow/Qwen3-0.6B-Distilled-30B-A3B` | Qwen/Qwen3-0.6B | apache-2.0 | 3,944 | not chosen: repo changed on 2026-09-04, so a recorded sha256 would drift; untied lm_head (752M parameters) |
| `tabularisai/Qwen3-0.3B-distil` | Qwen/Qwen3-0.6B | apache-2.0 | 4,373 | not chosen: 14-layer pruned student, card says "Work in progress" |
| `SEN-AGI/Lura-1.0-500m` | Qwen/Qwen2.5-0.5B | apache-2.0 | 1,512 | not chosen: training data not named; created 2026-08-16 |
| `numind/NuExtract-1.5-tiny` | Qwen/Qwen2.5-0.5B | mit | 5,326 | not chosen: extraction task model; the generic 8-question gate cannot certify it (needs `--gate-script`) |

Excluded before the table: mirrors and template-only re-uploads (unsloth, mlx-community,
litert-community, PrimeIntellect, dnotitia), reranker / TTS / classifier heads, a mixture-of-experts
and a diffusion variant, a speculative-decoding draft model, gemma-based fine-tunes (Gemma license),
and everything above 1B or under cc-by-nc.

## 2. Convert (one command)

```bash
git clone https://github.com/john-rocky/hf-to-litertlm && cd hf-to-litertlm
pip install litert-torch ai-edge-quantizer "transformers==5.14.*" huggingface_hub litert-lm
python scripts/convert.py suayptalha/Qwen3-0.6B-Code-Expert
# -> out/Qwen3-0.6B-Code-Expert/model.litertlm, gate.json, convert_report.json
```

Exit 0 means converted and gated, 1 converted but the gate failed, 2 refused at the entry gate.
Every run writes `convert_report.json` with the decisions taken.

What the run did on 2026-09-05 (litert-torch 0.9.3, ai-edge-quantizer 0.8.0, transformers 5.14.1,
litert-lm-builder 0.15.0, source revision `02c021d`):

| | |
|---|---|
| export | 217 s, stock export, int8 weights, the repo's own chat template embedded verbatim |
| bundle | `model.litertlm`, 613,406,208 bytes, sha256 `b9f8587090b56934e4189a9d969193b5c2fcc120b44ae819692451c92e88f480`; published as [mlboydaisuke/Qwen3-0.6B-Code-Expert-LiteRT](https://huggingface.co/mlboydaisuke/Qwen3-0.6B-Code-Expert-LiteRT) |
| metadata | stop tokens `<\|im_end\|>` (151645) and `<\|endoftext\|>` (151643), context 4096 tokens, a `thought` channel bounded by `<think>` / `</think>` |
| gate | 7 of 8 correct, no degenerate answer, median decode 138 tok/s (Mac GPU). The miss: "Roses are red, violets are ___" answered "violet" |
| template lint | warning only: the template uses `split`, `strip`, `lstrip`, `rstrip`, which current runtimes render |

`--int4` selects the converter's int4 recipe; not run for this model.

## 3. Add it to an existing app

**Dependency.** Xcode: File → Add Package Dependencies → `https://github.com/john-rocky/swift-litert-lm`,
product `LiteRTFoundation`. Or in `Package.swift`:

```swift
dependencies: [
  .package(url: "https://github.com/john-rocky/swift-litert-lm", from: "0.2.0"),
],
// in the target:
.product(name: "LiteRTFoundation", package: "swift-litert-lm"),
```

Build status on 2026-09-05: tag `0.2.0` (09f04f1) compiles for iOS with Xcode 26.1.1 and with
Xcode 27.0 beta 5. Tag `0.1.1` and the `main` history before 09f04f1 do not: they use Foundation
Models API that beta 5 removed and the iOS 26 SDK never had. The Easy-mode sources this recipe uses
(`LiteRTChat`, the downloader, the vendored runtime wrapper) are unchanged between e82e3b1 and
0.2.0. The runtime binaries are the LiteRT-LM v0.15.0 xcframeworks the package pins. Minimum OS:
iOS 16.

**Swift.**

```swift
import LiteRTFoundation

// Load. First launch downloads the file into Application Support/LiteRTModels
// (excluded from iCloud backup) and every later launch reuses it.
var chat: LiteRTChat? = try await LiteRTChat(
  huggingFaceRepo: "<your-org>/<your-model>-litertlm", fileName: "model.litertlm",
  modalities: [])                        // text-only bundle: no vision or audio tower
// This recipe's bundle: huggingFaceRepo: "mlboydaisuke/Qwen3-0.6B-Code-Expert-LiteRT"

// Ask. Deltas are incremental: append them.
let generation = Task {
  for try await delta in chat!.stream("What is 17 + 25? Answer briefly.") { transcript.append(delta) }
}
```

The default sampler is top-k 40, top-p 0.95, temperature 0.8. Thinking is off on this path: the
runtime renders the Qwen3 template with an empty `<think></think>` pair and the model answers
directly. `LiteRTChat(modelFileURL:)` takes a `thinking:` argument; the `huggingFaceRepo:`
initializer does not.

**Model delivery, one pattern.** Upload the `.litertlm` to a public, ungated Hugging Face repo you
own and pass its id and file name as above. The downloader is chunked and resumable and sends no
token, so a private or gated repo does not work; point `storageDirectory:` elsewhere if you need
to. Do not bundle a 613 MB file into the app. During development, skip the download: push the file
into the app's Documents directory and load it with `LiteRTChat(modelFileURL:modalities: [])`.

```bash
xcrun devicectl device copy to --device <udid> --domain-type appDataContainer \
  --domain-identifier <bundle-id> --source model.litertlm --destination Documents/model.litertlm
```

## 4. Stop and release

Measured on the Mac through this package's `LiteRTChat` (section 7 has the machine):

**Stop.** Call `cancel()`. The model stopped within 0.5 s on both backends, the stream ended with
the error `CANCELLED: Task cancelled`, and the process burned no further CPU. Cancelling only the
Swift `Task` that reads the stream does not stop the model: after the reader left the loop at 10
deltas, the engine decoded the full 2,017-token essay it had started, 7.3 s more on the GPU
(7.8 s of CPU) and 30 s more on the CPU backend (85 s of CPU). The vendored wrapper's stream has
no termination hook that reaches the native `cancel_process`, so a Stop button must call
`chat.cancel()`.

```swift
try chat?.cancel()          // stops the model; the stream throws CANCELLED and `generation` ends
```

After `cancel()` the conversation is finished: the next message on it throws the same `CANCELLED`
error. Make a new `LiteRTChat` for the next turn (see section 5).

**Release.** Drop the last reference. `Engine` and `Conversation` free their native handles in
`deinit`; there is no `close()`.

```swift
chat = nil                  // release the weights; create a new LiteRTChat to load again
```

GPU: the process footprint went from 1,376 MB with the engine loaded to 226 MB two seconds after
`chat = nil`; a fresh load then took 0.1 s and answered. CPU backend: 1,105 MB to 1,052 MB in the
same test (the footprint stayed), and the next load did not add to it (1,103 MB). Release the old
`LiteRTChat` before creating the new one, or both engines are resident at once.

## 5. Second turn: not on the same conversation today

Every second message on the same `LiteRTChat` failed, with thinking off and with thinking on, on
the GPU and on the CPU backend:

```
failedToStartStream(status: 13)
INTERNAL: The new rendered template string does not start with the previous rendered template string.
```

The embedded template renders the trailing assistant turn with an empty `<think></think>` pair
while that turn is the last message, and without the pair once the next message is appended; the
engine requires the new render to extend the old one byte for byte. Upstream issue
[#3443](https://github.com/google-ai-edge/LiteRT-LM/issues/3443) (open, filed 2026-09-01 against
v0.16.0 with the litert-community Qwen3-0.6B bundle) names the embedded Qwen3 chat template as the
cause. Any Qwen3-template bundle is affected; the converter embeds the repo's template verbatim.

Workaround: one `LiteRTChat` per turn. Release the previous one, create a new one from the same
file (0.1 s on the Mac with a warm compile cache; not measured on an iPhone), and put the history
you want the model to see into the prompt yourself. Fine-tunes of non-thinking bases (Qwen2.5,
SmolLM2, OLMo-2) do not carry this template shape; not measured in this recipe.

## 6. Verify

**The bundle, on any Mac or Linux box** (no Xcode needed):

```bash
python3 -m venv lt && lt/bin/pip install "litert-lm==0.17.0"
lt/bin/litert-lm run --from-huggingface-repo mlboydaisuke/Qwen3-0.6B-Code-Expert-LiteRT model.litertlm \
  --prompt "What is 17 + 25? Answer briefly." --thinking false --temperature 0 --top-k 1
# or the file you converted yourself:
lt/bin/litert-lm run out/Qwen3-0.6B-Code-Expert/model.litertlm \
  --prompt "What is 17 + 25? Answer briefly." --thinking false --temperature 0 --top-k 1
```

Expected output, from the runs on 2026-09-05 (Mac Studio, M4 Max):

```
17 + 25 = 42
```

litert-lm 0.17.0 printed it in 1.1 s wall on the local file and in 67 s through the repo form, download
included (one earlier attempt of the repo form stalled in the download and was killed after 10 min; the
retry went through); 0.16.0 in 1.0 s. A file fetched with `hf download` had the recorded sha256. Keep `--thinking false` when you use
greedy sampling: with thinking left on and `--temperature 0 --top-k 1`, the same prompt looped on
"17 + 25." inside the thought channel for 128 s and never answered. The Qwen3 model card says not to
use greedy decoding in thinking mode. With the CLI's default sampler and thinking on (0.17.0), the
model thought for one paragraph and answered `17 + 25 = 42`.

**The Swift path, on the Mac.** macOS is a supported platform of this package, so the same
`LiteRTChat(modelFileURL:modalities: [])` call that the iPhone app makes runs on the host. The
numbers in sections 4 and 7 come from a 140-line SwiftPM executable that loads the bundle, streams,
cancels, releases and reloads; its steps are described in those sections.

**On an iPhone.** Not run yet. The intended check is the sample app's local-model self-test:
build `Samples/LiteRTDemo` on the device, push the file into its Documents directory with the
`devicectl` command in section 3, launch with `LITERT_LOCAL_TEST=1`, and read the `LOCAL:` lines
(output text, load time, decode and prefill tok/s). On 2026-09-05 the sample does not build: it
still references the `LiteRTAudioSegment` / `LiteRTVideoSegment` types that Xcode 27 beta 5
removed from the package. That fix, and the device row, are pending.

## 7. Verified / unverified

Machine: Mac Studio, Apple M4 Max, 128 GB, macOS 27.0 (26A5416b). Toolchain: Xcode 27.0 beta 5
(27A5237l) for the Swift builds; swift-litert-lm 09f04f1, tagged `0.2.0` on 2026-09-05 (Easy-mode
sources unchanged since e82e3b1); LiteRT-LM v0.15.0 binaries as pinned. Date: 2026-09-05.

| check | result |
|---|---|
| conversion, entry gate, export | pass, 217 s (section 2) |
| exit gate (8 questions, bar 6) | 7/8, no degenerate answer, median decode 138 tok/s on the GPU |
| `LiteRTChat` first turn after a fresh load, GPU | "17 + 25 = 42" in 2.2 s; decode 142.7 tok/s, prefill 469.2 tok/s; footprint 1,411 MB. The prebuilt verifier on the same file: 141.2 / 472.2 tok/s |
| `LiteRTChat` first turn after a fresh load, CPU | "17 + 25 = 42" in 11.4 s; decode 33.3 tok/s, prefill 95.2 tok/s; footprint 1,107 MB |
| stop by `cancel()` | model idle within 0.5 s, GPU and CPU; conversation unusable afterwards |
| stop by cancelling the reading `Task` only | model keeps decoding to the end (2,017 tokens on GPU, 1,643 on CPU) |
| release (`chat = nil`) | GPU 1,376 → 226 MB; CPU 1,105 → 1,052 MB; reload 0.1 s, answers |
| second turn on one conversation | fails, status 13, thinking off and on (#3443) |
| CLI 0.16.0 and 0.17.0, `--thinking false`, greedy | "17 + 25 = 42", 1.0 s / 1.1 s |
| CLI, thinking on, greedy | looped in the thought channel, no answer after 128 s |
| `xcodebuild -scheme LiteRTFoundation -destination generic/platform=iOS`, Xcode 27.0 beta 5 | `0.1.1` FAIL and e82e3b1 FAIL (`Transcript.CustomSegment` removed, `LanguageModelCapabilities` label); 09f04f1 = `0.2.0` OK |
| same, Xcode 26.1.1 (17B100) | `0.1.1` FAIL and e82e3b1 FAIL (`LanguageModel` not in the iOS 26 SDK); 09f04f1 = `0.2.0` OK |

Unverified:

- Any iPhone. The iPhone 17 Pro row is added when the device is available; the sample app must
  build first (section 6).
- iOS Simulator. The runtime xcframework ships an `ios-arm64-simulator` slice and the README says
  the Simulator falls back to CPU; nothing was run there.
- Runtime binaries other than v0.15.0 in the Swift package, and Xcode versions other than 26.1.1
  and 27.0 beta 5.
- Multi-turn through the `litert-lm` CLI: two prompts on stdin were answered in one run and not in
  another; not established either way.
- `convert.py --int4`, LoRA adapter repos (the merge path), and every fine-tune other than this one.
- First-generation cost on a fresh engine on an iPhone (2.2 s GPU / 11.4 s CPU on the Mac with
  `prewarm: false`; the default `prewarm: true` moves it into the initializer).

## 8. Provenance

- Converted and verified by: john-rocky
- Recipe: https://github.com/john-rocky/swift-litert-lm/blob/main/docs/recipe-hf-finetune-to-iphone.md
- Measurements: section 7 above (Mac Studio M4 Max, 2026-09-05)
- Commits: swift-litert-lm @ 09f04f1 (tag `0.2.0`); hf-to-litertlm @ 5ffe9ee
- Maintained at: https://github.com/john-rocky/swift-litert-lm/issues

Last verified: 2026-09-05
