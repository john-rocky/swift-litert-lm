// LiteRTDemo — Foundation Models multimodal + guided-generation self-test.
//
// Exercises the full FM-mode consumer API over the LiteRT backend:
//   • G2 guided generation — `respond(generating:)` with a `@Generable` type
//   • images via Foundation Models' `Attachment` in a `@PromptBuilder`
//   • tool calling
//
// Audio and video have no Foundation Models path since Xcode 27 beta 5; LITERT_MMCHAT
// runs them in Easy mode.
//
// Launch with LITERT_FM=1. Lines are tagged "FM:" for devicectl polling.

#if canImport(FoundationModels)

import Foundation
import FoundationModels
import ImageIO
import LiteRTFoundation
import os

@available(iOS 27.0, macOS 27.0, *)
@Generable
struct PrimaryColors {
  @Guide(description: "Exactly three additive primary colors")
  var colors: [String]
}

/// A sample Foundation Models tool the LiteRT backend can call.
/// (Qualify `FoundationModels.Tool` — LiteRT-LM also exports a `Tool` type.)
@available(iOS 27.0, macOS 27.0, *)
struct TemperatureTool: FoundationModels.Tool {
  let name = "get_temperature"
  let description = "Get the current temperature for a city."

  @Generable
  struct Arguments {
    @Guide(description: "The city name")
    var city: String
  }

  func call(arguments: Arguments) async throws -> String {
    "The temperature in \(arguments.city) is 22°C and sunny."
  }
}

@available(iOS 27.0, macOS 27.0, *)
enum FMMultimodalSelfTest {
  private static let logger = Logger(subsystem: "com.example.litertdemo", category: "FM")

  static var isRequested: Bool { ProcessInfo.processInfo.environment["LITERT_FM"] != nil }

  static func log(_ message: String) {
    logger.log("\(message, privacy: .public)")
    print("FM: \(message)")
    fflush(stdout)
  }

  static func run() async {
    log("start — images + guided + tools over the Foundation Models API")

    let model: LiteRTLanguageModel
    let session: LanguageModelSession
    do {
      model = try await LiteRTLanguageModel(.gemma4_E2B)
      session = LanguageModelSession(model: model)
      log("session ready (LiteRT backend)")
    } catch {
      log("FATAL setup: \(error.localizedDescription)")
      log("DONE")
      return
    }

    // Each modality is independent so one failure doesn't mask the others.

    // Image probes in fresh sessions: separate "single vs multiple frames" from
    // "long-session history" effects.
    if let png = Bundle.main.url(forResource: "apple", withExtension: "png"),
      let source = CGImageSourceCreateWithURL(png as CFURL, nil),
      let img = CGImageSourceCreateImageAtIndex(source, 0, nil)
    {
      do {
        let s1 = LanguageModelSession(model: model)
        let a1 = try await s1.respond {
          Attachment(img)
          "What is shown in this image? Answer briefly."
        }
        log("IMG1(fresh) → \(a1.content.replacingOccurrences(of: "\n", with: " "))")
      } catch {
        log("IMG1(fresh) FAILED: \(error.localizedDescription)")
      }
      do {
        let s4 = LanguageModelSession(model: model)
        let a4 = try await s4.respond {
          Attachment(img)
          Attachment(img)
          Attachment(img)
          Attachment(img)
          "What is shown in these images? Answer briefly."
        }
        log("IMG4(fresh) → \(a4.content.replacingOccurrences(of: "\n", with: " "))")
      } catch {
        log("IMG4(fresh) FAILED: \(error.localizedDescription)")
      }
      // Multi-turn: the adapter rebuilds the conversation from the transcript
      // each turn, so images in *history* ride the initial-messages path.
      do {
        let s2 = LanguageModelSession(model: model)
        _ = try await s2.respond {
          Attachment(img)
          "What fruit is shown? One word."
        }
        let a2 = try await s2.respond(
          to: "What fruit was in the image I showed you earlier? One word.")
        log("IMG-HISTORY → \(a2.content.replacingOccurrences(of: "\n", with: " "))")
      } catch {
        log("IMG-HISTORY FAILED: \(error.localizedDescription)")
      }
    } else {
      log("IMG probes skipped — apple.png not bundled")
    }

    // G2 — guided generation into a @Generable type.
    do {
      let guided = try await session.respond(generating: PrimaryColors.self) {
        "List the three additive primary colors."
      }
      log("GUIDED → colors = \(guided.content.colors)")
    } catch {
      log("GUIDED FAILED: \(error.localizedDescription)")
    }

    // Tool calling through the FM API (the model calls the app's tool; FM runs
    // it and feeds the result back).
    do {
      let toolSession = LanguageModelSession(model: model, tools: [TemperatureTool()])
      let answer = try await toolSession.respond(to: "What is the temperature in Tokyo right now?")
      log("TOOL → \(answer.content.replacingOccurrences(of: "\n", with: " "))")
    } catch {
      log("TOOL FAILED: \(error.localizedDescription)")
    }

    log("DONE")
  }
}

#endif
