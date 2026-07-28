// LiteRTDemo — text decode benchmark probe.
//
// Measures text decode speed the way a real chat app experiences it: one warm
// engine, several short-chat turns in a row, reading LiteRT-LM's own per-turn
// counters (getBenchmarkInfo). The first turn is COLD (GPU shaders / weight
// conversion not yet warmed) and runs ~2× slower; from the second turn on the
// engine is at steady state. This is what explains "33 vs 55 tok/s".
//
// Mirrors the reference setup (ios-llm-benchmark): EngineConfig(.gpu,
// maxNumTokens: 2048) + a SamplerConfig on the conversation (the sampler is also
// what keeps benchmark mode from crashing in `output_buffer_dup`).
//
// Launch with LITERT_BENCH=1. Lines are tagged "BENCH:" for devicectl polling.

import Foundation
import LiteRTFoundation
import os

enum BenchSelfTest {
  private static let logger = Logger(subsystem: "com.example.litertdemo", category: "BENCH")

  static var isRequested: Bool { ProcessInfo.processInfo.environment["LITERT_BENCH"] != nil }

  static func log(_ message: String) {
    logger.log("\(message, privacy: .public)")
    print("BENCH: \(message)")
    fflush(stdout)
  }

  private static func mb(_ bytes: Int64) -> String {
    String(format: "%.0f MB", Double(bytes) / 1_048_576)
  }

  static func run() async {
    log("start — device=\(ProcessInfo.processInfo.operatingSystemVersionString)")

    let model = LiteRTModel.gemma4_E2B
    let path: String
    do {
      path = try await LiteRTChat.ensureModel(model)
    } catch {
      log("FATAL could not obtain model: \(error.localizedDescription)")
      log("DONE")
      return
    }

    // One warm engine, several short-chat turns. The reference benchmark + the
    // Gemma 4 E2B model card both report ~55–56 tok/s decode on iPhone 17 Pro;
    // that's the *warm* steady state, reached from the 2nd turn onward.
    ExperimentalFlags.optIntoExperimentalAPIs()
    ExperimentalFlags.enableBenchmark = true
    let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
    let prompt = "Explain what on-device AI means in simple terms."

    do {
      let config = try EngineConfig(
        modelPath: path, backend: .gpu, maxNumTokens: 2048, cacheDir: caches?.path)
      let engine = Engine(engineConfig: config)
      let initStart = Date()
      try await engine.initialize()
      log(String(format: "engine init %.1fs · %@", Date().timeIntervalSince(initStart),
        mb(LiteRTChat.memoryFootprintBytes())))

      // topK 40 / temperature 0 ≈ greedy, but a non-nil SamplerConfig is what
      // keeps benchmark mode from crashing (output_buffer_dup) on this build.
      let sampler = try SamplerConfig(topK: 40, topP: 0.95, temperature: 0.0)

      for turn in 1...4 {
        let conv = try await engine.createConversation(
          with: ConversationConfig(samplerConfig: sampler))
        for try await _ in conv.sendMessageStream(Message(prompt)) {}
        let b = try conv.getBenchmarkInfo()
        log(String(
          format: "turn %d (%@): decode %.1f tok/s · %d tok · prefill %.1f tok/s · ttft %.2fs · %@",
          turn, turn == 1 ? "cold" : "warm", b.lastDecodeTokensPerSecond,
          b.lastDecodeTokenCount, b.lastPrefillTokensPerSecond, b.timeToFirstTokenInSecond,
          mb(LiteRTChat.memoryFootprintBytes())))
      }
    } catch {
      log("FAILED: \(error.localizedDescription)")
    }

    // Product-API check: does LiteRTChat's prewarm make the *first* message warm?
    for pw in [false, true] {
      do {
        let chat = try await LiteRTChat(
          model, modalities: [] as Modality, enableBenchmark: true, prewarm: pw)
        _ = try await chat.respond(prompt)  // the user's first real message
        let b = try chat.lastBenchmark()
        log(String(format: "LiteRTChat prewarm=%@: first-message decode %.1f tok/s · ttft %.2fs",
          pw ? "ON " : "off", b.lastDecodeTokensPerSecond, b.timeToFirstTokenInSecond))
        try? await Task.sleep(nanoseconds: 800_000_000)
      } catch {
        log("LiteRTChat prewarm=\(pw) FAILED: \(error.localizedDescription)")
      }
    }

    log("DONE")
  }
}

// MARK: - Local converted-models smoke test (LITERT_LOCAL_TEST=1)
//
// Loads every `.litertlm` pushed into the app's Documents directory (via
// `devicectl device copy to … --destination <name>`), runs one short generation
// on each, and logs load time, peak footprint, the actual OUTPUT text (to eyeball
// coherence / numerical sanity), and decode/prefill tok/s + TTFT. On-device gate
// for the freshly converted hybrid/SSM models. Lines tagged "LOCAL:".
enum LocalModelsSelfTest {
  private static let logger = Logger(subsystem: "com.example.litertdemo", category: "LOCAL")

  static var isRequested: Bool {
    ProcessInfo.processInfo.environment["LITERT_LOCAL_TEST"] != nil
  }

  static func log(_ message: String) {
    logger.log("\(message, privacy: .public)")
    print("LOCAL: \(message)")
    fflush(stdout)
  }

  private static func mb(_ bytes: Int64) -> String {
    String(format: "%.0f MB", Double(bytes) / 1_048_576)
  }

  static func run() async {
    log("start — device=\(ProcessInfo.processInfo.operatingSystemVersionString)")
    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    // LITERT_LOCAL_ONLY=<substring> benches only matching models (case-insensitive),
    // e.g. LITERT_LOCAL_ONLY=llama to test one model instead of the whole Documents sweep.
    let only = ProcessInfo.processInfo.environment["LITERT_LOCAL_ONLY"]
    let files = ((try? FileManager.default.contentsOfDirectory(
      at: docs, includingPropertiesForKeys: nil)) ?? [])
      .filter { $0.pathExtension == "litertlm" }
      .filter { only == nil || $0.lastPathComponent.range(of: only!, options: .caseInsensitive) != nil }
      .sorted { $0.lastPathComponent < $1.lastPathComponent }

    log("found \(files.count) local .litertlm in Documents")
    if files.isEmpty {
      log("DONE — no models; push with: devicectl device copy to "
        + "--domain-type appDataContainer --domain-identifier com.rockyshikoku.litertdemo "
        + "--source X.litertlm --destination X.litertlm")
      return
    }

    // LITERT_LOCAL_DELETE=1 → remove the matching files (honours LITERT_LOCAL_ONLY) and stop.
    if ProcessInfo.processInfo.environment["LITERT_LOCAL_DELETE"] != nil {
      for url in files {
        do {
          try FileManager.default.removeItem(at: url)
          log("  DELETED \(url.lastPathComponent)")
        } catch {
          log("  DELETE FAILED \(url.lastPathComponent): \(error.localizedDescription)")
        }
      }
      log("DONE")
      return
    }

    // LITERT_LOCAL_LONG=1 → Metal-System-Trace mode: high max-tokens, a
    // long-output prompt, one warmup turn (discarded) then N measured turns on
    // the same warm engine so there is a wide steady-state window to attach
    // Instruments to. LITERT_LOCAL_MAXTOK / LITERT_LOCAL_TURNS tune it.
    let env = ProcessInfo.processInfo.environment
    let longMode = env["LITERT_LOCAL_LONG"] != nil
    let maxTok = Int(env["LITERT_LOCAL_MAXTOK"] ?? "") ?? (longMode ? 2000 : 512)
    let measuredTurns = Int(env["LITERT_LOCAL_TURNS"] ?? "") ?? (longMode ? 3 : 1)
    let shortPrompt = "Explain on-device AI in one short sentence."
    let longPrompt = "Write an extremely detailed, comprehensive essay of at "
      + "least 1500 words about the entire history of computing: mechanical "
      + "calculators, Babbage and Lovelace, Turing, vacuum tubes and ENIAC, the "
      + "transistor, integrated circuits, microprocessors, personal computers, "
      + "the internet, mobile systems-on-chip, and modern AI accelerators. "
      + "Cover each era thoroughly with dates, names, and technical detail, and "
      + "do not stop early."
    let prompt = longMode ? longPrompt : shortPrompt
    for url in files {
      let name = url.lastPathComponent
      let sz = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
      log("==== \(name) (\(mb(sz ?? 0))) max-tok \(maxTok) ====")
      do {
        let t0 = Date()
        let chat = try await LiteRTChat(
          modelFileURL: url,
          modalities: [] as Modality,
          maxTokens: maxTok,
          enableBenchmark: true,
          prewarm: false)
        log(String(format: "  loaded in %.1fs · footprint %@",
          Date().timeIntervalSince(t0), mb(LiteRTChat.memoryFootprintBytes())))

        if longMode {
          // Warmup turn (shader compile / weight conversion) — discarded.
          _ = try await chat.respond(shortPrompt)
          let bw = try chat.lastBenchmark()
          log(String(format: "  warmup decode %.1f tok/s · %d tok", bw.lastDecodeTokensPerSecond,
            bw.lastDecodeTokenCount))
        }

        for turn in 1...measuredTurns {
          let genStart = Date()
          let response = try await chat.respond(prompt)
          let b = try chat.lastBenchmark()
          log(String(format: "  turn %d RESULT decode %.1f tok/s · %d tok · prefill %.1f tok/s · ttft %.2fs · gen %.1fs · footprint %@",
            turn, b.lastDecodeTokensPerSecond, b.lastDecodeTokenCount,
            b.lastPrefillTokensPerSecond, b.timeToFirstTokenInSecond,
            Date().timeIntervalSince(genStart), mb(LiteRTChat.memoryFootprintBytes())))
          if turn == 1 {
            log("  OUTPUT: \(response.replacingOccurrences(of: "\n", with: " ").prefix(200))")
          }
        }
      } catch {
        log("  FAILED: \(error.localizedDescription)")
      }
    }
    log("DONE")
  }
}
