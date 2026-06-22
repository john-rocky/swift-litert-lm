// On-device overhead: Apple Foundation Models path (lean adapter) vs the raw
// LiteRT-LM Swift API. Launch with LITERT_COMPARE=1. **Build Release** for a
// meaningful number (Debug inflates the adapter glue).
//
// Same model, same prompt, greedy → identical output, so the wall-time ratio is
// the adapter+FM overhead, not a sampling difference. Engines are built one at a
// time (raw released before FM) to avoid double-loading the weights on device.

#if canImport(FoundationModels)

import Foundation
import FoundationModels
import LiteRTFoundation          // downloader/catalog + re-exported LiteRTLM core
import LiteRTLMFoundationModels  // the lean adapter under test

@available(iOS 27.0, macOS 27.0, *)
enum CompareSelfTest {
  static var isRequested: Bool { ProcessInfo.processInfo.environment["LITERT_COMPARE"] != nil }

  static func log(_ s: String) { print("CMP: \(s)"); fflush(stdout) }

  private static let prompt = "Explain on-device AI in two sentences."
  private static func now() -> Double { Date().timeIntervalSince1970 }

  static func run() async {
    log("start — FM adapter vs raw Swift API (greedy, gemma-4-E2B)")
    do {
      let path = try await LiteRTChat.ensureModel(.gemma4_E2B)
      let cfg = try EngineConfig(modelPath: path, backend: .gpu)

      // ── RAW path (core API), its own engine ──────────────────────────────
      var rawEngine: Engine? = Engine(engineConfig: cfg)
      try await rawEngine!.initialize()
      _ = try await rawGenerate(rawEngine!)            // warm up
      let raw = try await rawGenerate(rawEngine!)      // timed
      rawEngine = nil                                  // release before FM (memory)

      // ── FM path (lean adapter), its own engine ───────────────────────────
      let model = LiteRTLMFoundationModels.LiteRTLanguageModel(modelPath: path)
      _ = try await fmGenerate(model)                  // warm up
      let fm = try await fmGenerate(model)             // timed

      let same = raw.text.trimmingCharacters(in: .whitespacesAndNewlines)
        == fm.text.trimmingCharacters(in: .whitespacesAndNewlines)
      log("identical output: \(same)")
      log(String(format: "raw (core API): %.3f s", raw.seconds))
      log(String(format: "fm  (adapter) : %.3f s", fm.seconds))
      log(String(format: "overhead      : %+.1f%% (fm/raw = %.3f x)",
        (fm.seconds / raw.seconds - 1) * 100, fm.seconds / raw.seconds))
      log("DONE")
    } catch {
      log("FAILED: \(error.localizedDescription)")
      log("DONE")
    }
  }

  @available(iOS 27.0, macOS 27.0, *)
  private static func rawGenerate(_ engine: Engine) async throws -> (text: String, seconds: Double) {
    let conv = try await engine.createConversation(
      with: ConversationConfig(samplerConfig: try? SamplerConfig(topK: 1, topP: 1.0, temperature: 0.0)))
    let t0 = now()
    var out = ""
    for try await chunk in conv.sendMessageStream(Message(prompt)) { out += chunk.toString }
    return (out, now() - t0)
  }

  @available(iOS 27.0, macOS 27.0, *)
  private static func fmGenerate(_ model: LiteRTLMFoundationModels.LiteRTLanguageModel) async throws
    -> (text: String, seconds: Double)
  {
    let session = LanguageModelSession(model: model)
    let t0 = now()
    let r = try await session.respond(to: prompt, options: GenerationOptions(sampling: .greedy))
    return (r.content, now() - t0)
  }
}

#endif
