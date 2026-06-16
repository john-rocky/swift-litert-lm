// swift-litert-lm — Easy mode (`LiteRTChat`)
//
// The 10-minute on-ramp: pick a model, get a chat. `LiteRTChat` owns every part
// of running Gemma 4 on iPhone that is otherwise a multi-day yak-shave —
// downloading the right `.litertlm`, refusing to load on a device that would
// jetsam, bringing up the Metal GPU backend (and the vision/audio towers you
// ask for), and streaming tokens.
//
//   let chat = try await LiteRTChat(.gemma4_E2B)
//   for try await token in chat.stream("Describe this", image: jpeg) { print(token) }

import Foundation
@_exported import LiteRTLM  // re-export Message/Content/etc. for callers

/// Audio input for a multimodal turn.
public enum AudioInput: Sendable {
  /// Raw audio bytes (e.g. a WAV file's contents).
  case data(Data)
  /// An absolute file URL to an audio file.
  case file(URL)

  var content: Content {
    switch self {
    case .data(let d): return .audioData(d)
    case .file(let url): return .audioFile(url.path)
    }
  }
}

/// A ready-to-use LiteRT-LM chat session backed by the Metal GPU.
public final class LiteRTChat {

  /// The model this session is running.
  public let model: LiteRTModel
  /// The modalities actually brought up for this session.
  public let modalities: Modality
  /// Absolute path to the loaded model file.
  public let modelPath: String

  private let engine: Engine
  private let conversation: Conversation

  // MARK: Lifecycle

  /// Bring up a chat session, downloading the model on first use.
  ///
  /// - Parameters:
  ///   - model: Which catalog model to run.
  ///   - modalities: Towers to enable. Defaults to the model's `defaultModalities`
  ///     (text+image for Gemma 4 E2B). Requesting an unsupported tower is ignored.
  ///   - storageDirectory: Where to keep the downloaded model. Defaults to
  ///     Application Support/LiteRTModels (persistent, excluded from backup).
  ///   - allowUnsafeMemory: Bypass the device-RAM safety check. Off by default —
  ///     Easy mode refuses to load on a device that would likely jetsam mid-run.
  ///   - enableBenchmark: Turn on the engine's benchmark instrumentation so
  ///     `lastBenchmark()` reports real prefill/decode tokens-per-second.
  ///   - onDownloadProgress: Called on the first run as the model downloads.
  /// - Throws: `LiteRTChatError` for memory/availability problems, `LiteRTLMError`
  ///   for engine failures, or a download error.
  public convenience init(
    _ model: LiteRTModel,
    modalities: Modality? = nil,
    storageDirectory: URL? = nil,
    allowUnsafeMemory: Bool = false,
    enableBenchmark: Bool = false,
    onDownloadProgress: (@Sendable (ModelDownloader.Progress) -> Void)? = nil
  ) async throws {
    // Memory gate: refuse rather than let the OS kill us partway through a load.
    if !allowUnsafeMemory {
      let ram = Int64(ProcessInfo.processInfo.physicalMemory)
      if ram < model.minimumDeviceRAM {
        throw LiteRTChatError.insufficientMemory(
          haveBytes: ram, needBytes: model.minimumDeviceRAM)
      }
    }

    let path = try await LiteRTChat.ensureModel(
      model, storageDirectory: storageDirectory, onProgress: onDownloadProgress)

    var wanted = modalities ?? model.defaultModalities
    wanted.formIntersection(model.supportedModalities)

    // Gemma 4's variable-resolution vision: cap per-image visual tokens to keep
    // the GPU working set bounded. Takes effect immediately, so set before init.
    ExperimentalFlags.optIntoExperimentalAPIs()
    if wanted.contains(.vision), let budget = model.defaultVisualTokenBudget {
      ExperimentalFlags.visualTokenBudget = budget
    }
    if enableBenchmark { ExperimentalFlags.enableBenchmark = true }

    let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
    // Each tower's backend is dictated by the model's section constraints, not a
    // free choice: e.g. Gemma 4 E2B's audio encoder is CPU-only (see catalog).
    let config = try EngineConfig(
      modelPath: path,
      backend: .gpu,
      visionBackend: wanted.contains(.vision) ? model.visionBackend : nil,
      audioBackend: wanted.contains(.audio) ? model.audioBackend : nil,
      maxNumTokens: model.defaultMaxTokens,
      cacheDir: caches?.path
    )
    let engine = Engine(engineConfig: config)
    try await engine.initialize()  // runs on the engine actor, off the main thread
    let conversation = try await engine.createConversation()

    self.init(
      model: model, modalities: wanted, modelPath: path,
      engine: engine, conversation: conversation)
  }

  private init(
    model: LiteRTModel, modalities: Modality, modelPath: String,
    engine: Engine, conversation: Conversation
  ) {
    self.model = model
    self.modalities = modalities
    self.modelPath = modelPath
    self.engine = engine
    self.conversation = conversation
  }

  // MARK: Generation

  /// Stream a response token-by-token. Each yielded value is the next text delta.
  ///
  /// - Parameters:
  ///   - prompt: The user's text.
  ///   - image: Optional image bytes (PNG/JPEG). Requires the vision tower.
  ///   - audio: Optional audio input. Requires the audio tower.
  public func stream(
    _ prompt: String, image: Data? = nil, audio: AudioInput? = nil
  ) -> AsyncThrowingStream<String, Error> {
    var contents: [Content] = [.text(prompt)]
    if let image { contents.append(.imageData(image)) }
    if let audio { contents.append(audio.content) }
    let message = Message(contents: contents)

    return AsyncThrowingStream { continuation in
      let task = Task {
        do {
          for try await chunk in conversation.sendMessageStream(message) {
            let delta = chunk.toString
            if !delta.isEmpty { continuation.yield(delta) }
          }
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
      continuation.onTermination = { _ in task.cancel() }
    }
  }

  /// Generate a full response (non-streaming convenience).
  public func respond(
    _ prompt: String, image: Data? = nil, audio: AudioInput? = nil
  ) async throws -> String {
    var out = ""
    for try await delta in stream(prompt, image: image, audio: audio) { out += delta }
    return out
  }

  /// Cancel the in-flight generation, if any.
  public func cancel() throws { try conversation.cancel() }

  /// Engine-measured benchmark info for the most recent turn (prefill/decode
  /// tokens-per-second, time-to-first-token). Requires `enableBenchmark: true`
  /// at construction.
  public func lastBenchmark() throws -> BenchmarkInfo { try conversation.getBenchmarkInfo() }

  /// Current resident memory footprint of the app process, in bytes — the
  /// jetsam-relevant `phys_footprint`. Handy for verifying you're under the
  /// device's memory ceiling while a model is loaded.
  public static func memoryFootprintBytes() -> Int64 { processFootprintBytes() }

  // MARK: Model management

  /// Ensure the model file is present locally, downloading it if needed, and
  /// return its absolute path. Useful to pre-download before constructing a chat.
  @discardableResult
  public static func ensureModel(
    _ model: LiteRTModel,
    storageDirectory: URL? = nil,
    onProgress: (@Sendable (ModelDownloader.Progress) -> Void)? = nil
  ) async throws -> String {
    let dir = try storageDirectory ?? defaultStorageDirectory()
    let dest = dir.appendingPathComponent(model.fileName)
    try await ModelDownloader.shared.download(
      from: model.downloadURL, to: dest,
      expectedBytes: model.approximateBytes, onProgress: onProgress)
    return dest.path
  }

  /// The default on-device storage location for models.
  public static func defaultStorageDirectory() throws -> URL {
    let base = try FileManager.default.url(
      for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
    let dir = base.appendingPathComponent("LiteRTModels", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
  }
}

/// Errors specific to Easy mode.
public enum LiteRTChatError: Error, LocalizedError {
  /// The device does not have enough RAM to load this model safely.
  case insufficientMemory(haveBytes: Int64, needBytes: Int64)

  public var errorDescription: String? {
    switch self {
    case .insufficientMemory(let have, let need):
      let haveStr = ByteCountFormatter.string(fromByteCount: have, countStyle: .memory)
      let needStr = ByteCountFormatter.string(fromByteCount: need, countStyle: .memory)
      return
        "This device has \(haveStr) of RAM; this model needs at least \(needStr). "
        + "Pass allowUnsafeMemory: true to try anyway."
    }
  }
}
