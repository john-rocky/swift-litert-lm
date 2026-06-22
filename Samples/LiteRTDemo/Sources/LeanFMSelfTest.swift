// Device self-test for the LEAN upstream-candidate adapter
// (`pr/LiteRTLMFoundationModels`) — not the app-layer FM. Verifies the minimal
// configuration (respond / guided / tools) runs through a real
// `LanguageModelSession` on device. Launch with LITERT_LEAN_FM=1.
//
// The model file is obtained via the app-layer downloader; the FM run below uses
// ONLY the lean adapter (`LiteRTLMFoundationModels.LiteRTLanguageModel`).

#if canImport(FoundationModels)

import Foundation
import FoundationModels
import LiteRTFoundation          // downloader + catalog (to fetch the model file)
import LiteRTLMFoundationModels  // the lean adapter under test

@available(iOS 27.0, macOS 27.0, *)
@Generable
struct LeanColors {
  @Guide(description: "Exactly three additive primary colors")
  var colors: [String]
}

@available(iOS 27.0, macOS 27.0, *)
struct LeanTemperatureTool: FoundationModels.Tool {
  let name = "get_temperature"
  let description = "Get the current temperature for a city."
  @Generable struct Arguments {
    @Guide(description: "The city name")
    var city: String
  }
  func call(arguments: Arguments) async throws -> String {
    "The temperature in \(arguments.city) is 21°C and clear."
  }
}

@available(iOS 27.0, macOS 27.0, *)
enum LeanFMSelfTest {
  static var isRequested: Bool { ProcessInfo.processInfo.environment["LITERT_LEAN_FM"] != nil }

  static func log(_ s: String) { print("LEAN: \(s)"); fflush(stdout) }

  static func run() async {
    log("start — lean LiteRTLMFoundationModels adapter, minimal config (text)")
    do {
      // Obtain the model file (already downloaded on this device → returns fast).
      let path = try await LiteRTChat.ensureModel(.gemma4_E2B)
      log("model path ok")

      // Build the backend with the LEAN adapter only (text-only = minimal config).
      let model = LiteRTLMFoundationModels.LiteRTLanguageModel(modelPath: path)

      log("respond…")
      let a = try await LanguageModelSession(model: model)
        .respond(to: "Explain on-device AI in one sentence.")
      log("RESPOND → \(a.content.replacingOccurrences(of: "\n", with: " "))")

      log("guided…")
      let g = try await LanguageModelSession(model: model)
        .respond(generating: LeanColors.self) { "List the three additive primary colors." }
      log("GUIDED → \(g.content.colors)")

      log("tool…")
      let t = try await LanguageModelSession(model: model, tools: [LeanTemperatureTool()])
        .respond(to: "What is the temperature in Tokyo right now?")
      log("TOOL → \(t.content.replacingOccurrences(of: "\n", with: " "))")

      log("DONE")
    } catch {
      log("FAILED: \(error.localizedDescription)")
      log("DONE")
    }
  }
}

#endif
