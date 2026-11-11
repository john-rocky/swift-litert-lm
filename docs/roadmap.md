# Roadmap

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
