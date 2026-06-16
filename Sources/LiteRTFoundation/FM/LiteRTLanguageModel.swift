// swift-litert-lm — Foundation Models backend (FM mode)
//
// `LiteRTLanguageModel` makes LiteRT-LM a first-class Apple Foundation Models
// backend, alongside Apple's own `CoreAILanguageModel` and `MLXLanguageModel`:
//
//   let model   = try await LiteRTLanguageModel(.gemma4_E2B)
//   let session = LanguageModelSession(model: model)          // Apple's exact API
//   let answer  = try await session.respond(to: "Hi")          // streaming / tools / @Generable
//
// The FM API is transcript-based (each turn hands the executor the full
// conversation), while LiteRT-LM is stateful (a `Conversation` accumulates its
// own KV cache). We bridge by rebuilding a fresh LiteRT `Conversation` from the
// transcript on each turn — correct and simple; an incremental fast-path is a
// later optimization.

#if canImport(FoundationModels)

import Foundation
import FoundationModels
import LiteRTLM

/// A LiteRT-LM model exposed as an Apple Foundation Models backend.
@available(iOS 27.0, macOS 27.0, *)
public struct LiteRTLanguageModel: LanguageModel {
  public typealias Executor = LiteRTExecutor

  public let capabilities: LanguageModelCapabilities
  public let executorConfiguration: LiteRTExecutor.Configuration

  /// Create the backend, downloading the model on first use.
  ///
  /// - Parameters:
  ///   - model: Which catalog model to run.
  ///   - storageDirectory: Where to keep the downloaded model (defaults to
  ///     Application Support/LiteRTModels).
  ///   - onDownloadProgress: Called on first run while the model downloads.
  public init(
    _ model: LiteRTModel,
    storageDirectory: URL? = nil,
    onDownloadProgress: (@Sendable (ModelDownloader.Progress) -> Void)? = nil
  ) async throws {
    let path = try await LiteRTChat.ensureModel(
      model, storageDirectory: storageDirectory, onProgress: onDownloadProgress)
    self.executorConfiguration = LiteRTExecutor.Configuration(
      modelPath: path, maxNumTokens: model.defaultMaxTokens)
    // Text generation only for now; vision / guided generation are later phases.
    self.capabilities = LanguageModelCapabilities(capabilities: [])
  }
}

/// Drives generation for `LiteRTLanguageModel` over the FM executor protocol.
@available(iOS 27.0, macOS 27.0, *)
public final class LiteRTExecutor: LanguageModelExecutor {
  public typealias Model = LiteRTLanguageModel

  /// Lightweight, `Hashable` description of what engine to build. The actual
  /// (async-initialized) engine is created lazily by the executor.
  public struct Configuration: Hashable, Sendable {
    public let modelPath: String
    public let maxNumTokens: Int
    public init(modelPath: String, maxNumTokens: Int) {
      self.modelPath = modelPath
      self.maxNumTokens = maxNumTokens
    }
  }

  private let engine: LazyEngine

  public init(configuration: Configuration) throws {
    self.engine = LazyEngine(configuration: configuration)
  }

  public func prewarm(model: Model, transcript: Transcript) {
    // Kick off engine creation + a tiny warmup so the first real turn is fast.
    Task { try? await engine.prewarmed() }
  }

  public func respond(
    to request: LanguageModelExecutorGenerationRequest,
    model: Model,
    streamingInto channel: LanguageModelExecutorGenerationChannel
  ) async throws {
    let engine = try await self.engine.ready()
    let plan = try Self.plan(from: request.transcript)

    let conversation = try await engine.createConversation(
      with: ConversationConfig(
        systemMessage: plan.systemMessage,
        initialMessages: plan.history,
        samplerConfig: try? SamplerConfig(topK: 40, topP: 0.95, temperature: 0.8)))

    for try await chunk in conversation.sendMessageStream(plan.prompt) {
      let delta = chunk.toString
      if !delta.isEmpty {
        await channel.send(.response(action: .appendText(delta, tokenCount: 1)))
      }
    }
  }

  // MARK: Transcript → LiteRT messages

  private struct Plan {
    let systemMessage: Message?
    let history: [Message]
    let prompt: Message
  }

  /// Split the FM transcript into a system message, prior turns (history), and
  /// the final user prompt that this turn should answer.
  private static func plan(from transcript: Transcript) throws -> Plan {
    let entries = Array(transcript)
    guard
      let lastPromptIndex = entries.lastIndex(where: {
        if case .prompt = $0 { return true } else { return false }
      })
    else {
      throw LiteRTFMError.noPrompt
    }

    var systemText: [String] = []
    var history: [Message] = []
    var prompt: Message?

    for (i, entry) in entries.enumerated() {
      switch entry {
      case .instructions(let instructions):
        systemText.append(text(of: instructions.segments))
      case .prompt(let p):
        let message = Message(contents: contents(of: p.segments), role: .user)
        if i == lastPromptIndex { prompt = message } else { history.append(message) }
      case .response(let r):
        history.append(Message(contents: [.text(text(of: r.segments))], role: .model))
      case .toolCalls, .toolOutput, .reasoning:
        break  // not handled in this phase
      @unknown default:
        break
      }
    }

    let system = systemText.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    return Plan(
      systemMessage: system.isEmpty ? nil : Message(system, role: .system),
      history: history,
      prompt: prompt!  // guaranteed by lastPromptIndex
    )
  }

  /// Concatenate the text of a segment list (non-text segments ignored for now).
  private static func text(of segments: [Transcript.Segment]) -> String {
    segments.compactMap { segment in
      if case .text(let t) = segment { return t.content } else { return nil }
    }.joined(separator: " ")
  }

  /// Map FM segments to LiteRT content. Text today; image attachments are a
  /// later phase (the spine already supports `.imageData`).
  private static func contents(of segments: [Transcript.Segment]) -> [Content] {
    var out: [Content] = []
    for segment in segments {
      if case .text(let t) = segment, !t.content.isEmpty {
        out.append(.text(t.content))
      }
    }
    return out.isEmpty ? [.text("")] : out
  }
}

/// Errors specific to the Foundation Models bridge.
@available(iOS 27.0, macOS 27.0, *)
public enum LiteRTFMError: Error, LocalizedError {
  case noPrompt

  public var errorDescription: String? {
    switch self {
    case .noPrompt: return "The transcript contains no prompt to respond to."
    }
  }
}

/// Lazily creates and caches the LiteRT engine. The FM executor's `init` is
/// synchronous but engine initialization is async, so we defer it to the first
/// `respond` (which is async) and memoize the result.
@available(iOS 27.0, macOS 27.0, *)
private actor LazyEngine {
  private let configuration: LiteRTExecutor.Configuration
  private var engine: Engine?
  private var warmed = false

  init(configuration: LiteRTExecutor.Configuration) {
    self.configuration = configuration
  }

  func ready() async throws -> Engine {
    if let engine { return engine }
    let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
    let config = try EngineConfig(
      modelPath: configuration.modelPath, backend: .gpu,
      maxNumTokens: configuration.maxNumTokens, cacheDir: caches?.path)
    let created = Engine(engineConfig: config)
    try await created.initialize()
    engine = created
    return created
  }

  func prewarmed() async throws {
    let engine = try await ready()
    if warmed { return }
    warmed = true
    let warmup = try await engine.createConversation()
    for try await _ in warmup.sendMessageStream(Message("Hi")) {}
  }
}

#endif
