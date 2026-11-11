# Sample app: `Samples/LiteRTDemo`

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
