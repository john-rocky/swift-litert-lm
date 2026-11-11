# LiteRT-LM v0.19.0

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
