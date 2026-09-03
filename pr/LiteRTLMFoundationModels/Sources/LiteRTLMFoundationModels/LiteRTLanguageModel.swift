// Copyright 2026 Daisuke Majima
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// LiteRT-LM as an Apple Foundation Models backend.
//
// `LiteRTLanguageModel` conforms to the iOS 27 `LanguageModel` protocol, so a
// LiteRT-LM model can drive a stock `LanguageModelSession` — alongside Apple's
// own conformers `SystemLanguageModel` (on-device) and
// `PrivateCloudComputeLanguageModel`:
//
//   let cfg     = try EngineConfig(modelPath: path, backend: .gpu)
//   let model   = LiteRTLanguageModel(engineConfig: cfg)
//   let session = LanguageModelSession(model: model)        // Apple's exact API
//   let answer  = try await session.respond(to: "Hi")        // streaming / tools / @Generable
//
// The FM API is transcript-based (each turn hands the executor the full
// conversation); LiteRT-LM is stateful (a `Conversation` accumulates its own KV
// cache). We bridge by rebuilding a fresh LiteRT `Conversation` from the
// transcript on each turn — correct and simple; an incremental fast-path is a
// later optimization.
//
// Depends on only the LiteRT-LM core Swift API (`LiteRTLM`) and `FoundationModels`.

#if canImport(FoundationModels) && compiler(>=6.4)

import Foundation
import FoundationModels
import LiteRTLM
import OSLog

private let logger = Logger(
  subsystem: "com.google.odml.litertlm.swift", category: "FoundationModels")

// MARK: - Model

/// A LiteRT-LM model exposed as an Apple Foundation Models backend.
@available(iOS 27.0, macOS 27.0, *)
public struct LiteRTLanguageModel: LanguageModel {
  public typealias Executor = LiteRTExecutor

  public let capabilities: LanguageModelCapabilities
  public let executorConfiguration: LiteRTExecutor.Configuration

  /// Build from an `EngineConfig` (the primary initializer). The whole config is
  /// carried through to the engine verbatim — including `cacheDir`, `loraRank`,
  /// `audioLoraRank` and `maxNumImages`.
  ///
  /// - Parameters:
  ///   - engineConfig: How to build the LiteRT engine.
  ///   - visualTokenBudget: Per-image visual-token cap, applied to each
  ///     conversation created over this model; nil = engine default.
  public init(engineConfig: EngineConfig, visualTokenBudget: Int32? = nil) {
    self.executorConfiguration = LiteRTExecutor.Configuration(
      engineConfig: engineConfig, visualTokenBudget: visualTokenBudget)
    var capabilities: [LanguageModelCapabilities.Capability] = [.guidedGeneration, .toolCalling]
    if engineConfig.visionBackend != nil { capabilities.append(.vision) }
    self.capabilities = LanguageModelCapabilities(capabilities)
  }

  /// Build from a model path and explicit settings (sugar over `init(engineConfig:)`).
  ///
  /// Unlike `init(engineConfig:)`, this initializer supplies a `cacheDir` default:
  /// the app's Caches directory, which is writable on every Apple platform.
  ///
  /// - Parameters:
  ///   - modelPath: Absolute path to an on-disk `.litertlm`.
  ///   - backend: Main compute backend (default `.gpu`).
  ///   - visionBackend / audioBackend: Backend per encoder tower, or nil to leave
  ///     that tower off (the safe default for a text-only model).
  ///   - visualTokenBudget: Per-image visual-token cap (nil = engine default).
  ///   - maxTokens: KV/context budget (nil = model/engine default).
  ///   - cacheDir: Where the engine writes its cache files (nil = the app's Caches
  ///     directory).
  /// - Throws: `LiteRTLMError` if `maxTokens` is less than or equal to 0.
  public init(
    modelPath: String,
    backend: Backend = .gpu,
    visionBackend: Backend? = nil,
    audioBackend: Backend? = nil,
    visualTokenBudget: Int32? = nil,
    maxTokens: Int? = 2048,
    cacheDir: String? = nil
  ) throws {
    let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
    self.init(
      engineConfig: try EngineConfig(
        modelPath: modelPath, backend: backend,
        visionBackend: visionBackend, audioBackend: audioBackend,
        maxNumTokens: maxTokens, cacheDir: cacheDir ?? caches?.path),
      visualTokenBudget: visualTokenBudget)
  }

  /// Release every cached LiteRT engine built for FM sessions, freeing their
  /// (multi-GB) weights. Any live `LanguageModelSession` over this backend
  /// rebuilds its engine on the next turn.
  public static func releaseCachedEngines() async {
    await EngineCache.shared.purgeAll()
  }
}

// MARK: - Executor

/// Drives generation for `LiteRTLanguageModel` over the FM executor protocol.
@available(iOS 27.0, macOS 27.0, *)
public final class LiteRTExecutor: LanguageModelExecutor {
  public typealias Model = LiteRTLanguageModel

  /// What engine to build. FM requires this to be a `Hashable` value, so it
  /// cannot hold a live `Engine`; the engine is built lazily and shared across
  /// every executor whose configuration compares equal.
  ///
  /// Wrapping `EngineConfig` whole (rather than mirroring a subset of its fields)
  /// means every engine setting — `cacheDir`, `loraRank`, `audioLoraRank`,
  /// `maxNumImages`, and anything the core adds later — flows through unchanged,
  /// and equality covers all of them.
  public struct Configuration: Hashable, Sendable {
    public let engineConfig: EngineConfig
    /// Per-image visual-token cap, applied to each conversation this executor
    /// creates. A conversation-level setting, so it is not part of `EngineConfig`.
    public let visualTokenBudget: Int32?

    public var modelPath: String { engineConfig.modelPath }

    public init(engineConfig: EngineConfig, visualTokenBudget: Int32? = nil) {
      self.engineConfig = engineConfig
      self.visualTokenBudget = visualTokenBudget
    }
  }

  private let engine: LazyEngine
  private let visualTokenBudget: Int32?

  public init(configuration: Configuration) throws {
    // Share one engine per engine configuration across executors. FM builds a
    // new executor per session (a plain session and a tool-enabled session over
    // the same model yield two executors), and each engine loads multi-GB
    // weights — without sharing, the second session OOMs the app.
    // `visualTokenBudget` is applied per conversation, so it does not force a
    // separate engine either.
    self.engine = EngineCache.shared.engine(for: configuration.engineConfig)
    self.visualTokenBudget = configuration.visualTokenBudget
  }

  public func prewarm(model: Model, transcript: Transcript) {
    Task { try? await engine.prewarmed() }
  }

  public func respond(
    to request: LanguageModelExecutorGenerationRequest,
    model: Model,
    streamingInto channel: LanguageModelExecutorGenerationChannel
  ) async throws {
    try Self.checkCapabilities(of: model, for: request)
    let engine = try await self.engine.ready()
    // Guided generation: if the request carries a schema, encode it to JSON and
    // steer the model via the prompt (skeleton-in-prompt). Tools: if enabled,
    // describe them in the prompt and detect a tool-call in the output. Both are
    // soft (prompt-driven); hard constrained decoding (llguidance) is a follow-up.
    let tools = request.enabledToolDefinitions
    let schemaJSON = request.schema.flatMap { try? Self.encodeSchema($0) }
    let plan = try Self.plan(from: request.transcript, schemaJSON: schemaJSON, tools: tools)

    let structured = schemaJSON != nil || !tools.isEmpty
    let conversation = try await engine.createConversation(
      with: ConversationConfig(
        systemMessage: plan.systemMessage,
        initialMessages: plan.history,
        samplerConfig: Self.sampler(for: request.generationOptions, structured: structured),
        visualTokenBudget: visualTokenBudget))

    if !tools.isEmpty {
      var full = ""
      for try await chunk in conversation.sendMessageStream(plan.prompt) { full += chunk.toString }
      if let call = Self.parseToolCall(from: full, tools: tools) {
        await channel.send(
          .toolCalls(
            action: .toolCall(
              id: UUID().uuidString, name: call.name,
              action: .appendArguments(call.arguments, tokenCount: call.arguments.count))))
      } else {
        await channel.send(.response(action: .appendText(full, tokenCount: full.count)))
      }
    } else if schemaJSON != nil {
      var full = ""
      for try await chunk in conversation.sendMessageStream(plan.prompt) { full += chunk.toString }
      let json = Self.unwrapSchemaEcho(
        Self.extractJSONObject(from: full) ?? full, schemaJSON: schemaJSON)
      if let field = Self.schemaEchoField(in: json, schemaJSON: schemaJSON) {
        throw LiteRTFMError.schemaEcho(field: field)
      }
      await channel.send(.response(action: .appendText(json, tokenCount: json.count)))
    } else {
      for try await chunk in conversation.sendMessageStream(plan.prompt) {
        let delta = chunk.toString
        if !delta.isEmpty {
          await channel.send(.response(action: .appendText(delta, tokenCount: 1)))
        }
      }
    }
  }

  /// Refuse, loudly, what the model did not declare: an image attachment on a
  /// model created without a vision backend is `unsupportedCapability(.vision)`,
  /// never a silently dropped segment.
  private static func checkCapabilities(
    of model: Model, for request: LanguageModelExecutorGenerationRequest
  ) throws {
    if !model.capabilities.contains(.vision), hasImageAttachment(request.transcript) {
      throw LanguageModelError.unsupportedCapability(
        .init(
          capability: .vision,
          debugDescription:
            "This LiteRT-LM model has no vision backend; create it with a "
            + "`visionBackend` to send image attachments."))
    }
  }

  private static func hasImageAttachment(_ transcript: Transcript) -> Bool {
    transcript.contains { entry in
      let segments: [Transcript.Segment]
      switch entry {
      case .instructions(let i): segments = i.segments
      case .prompt(let p): segments = p.segments
      case .response(let r): segments = r.segments
      case .toolOutput(let o): segments = o.segments
      default: return false
      }
      return segments.contains { segment in
        if case .attachment(let a) = segment, case .image = a.content { return true }
        return false
      }
    }
  }

  /// Undo a "schema echo": asked for an object matching a schema, a small model
  /// sometimes returns the schema itself with the values filled in under
  /// `properties` (`{"type": "object", "properties": {"colors": [...]}, ...}`).
  /// When the object carries none of the schema's top-level keys but its
  /// `properties` member does, hand FM that member instead.
  private static func unwrapSchemaEcho(_ json: String, schemaJSON: String?) -> String {
    guard let schemaJSON,
      let schemaData = schemaJSON.data(using: .utf8),
      let schema = (try? JSONSerialization.jsonObject(with: schemaData)) as? [String: Any],
      let expected = (schema["properties"] as? [String: Any])?.keys, !expected.isEmpty,
      let data = json.data(using: .utf8),
      let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
      !expected.contains(where: { obj[$0] != nil }),
      let inner = obj["properties"] as? [String: Any],
      expected.contains(where: { inner[$0] != nil }),
      let out = try? JSONSerialization.data(withJSONObject: inner),
      let string = String(data: out, encoding: .utf8)
    else { return json }
    logger.warning("Guided generation: unwrapped a schema-shaped reply to its `properties`.")
    return string
  }

  /// Guided-generation guidance for the prompt: a field guide (name, type,
  /// description) plus a skeleton instance with `<...>` placeholders, both built
  /// from the encoded `GenerationSchema`. The raw schema never enters the
  /// prompt: a small model imitates whatever shape it sees, and a schema dump
  /// comes back as the schema (`{"colors": {"type": "array", "items": ...}}`),
  /// which no post-processing can turn into the array that was asked for.
  static func guidedInstructions(fromSchemaJSON json: String) -> String {
    guard let data = json.data(using: .utf8),
      let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    else {
      return "Respond with ONLY a JSON object that conforms to this JSON schema. "
        + "Output valid JSON and nothing else:\n\(json)"
    }
    let hint = SchemaHint(definitions: root["$defs"] as? [String: Any] ?? [:])
    var guide: [String] = []
    hint.describe(root, path: "", into: &guide, depth: 0)
    var lines = [
      "Respond with ONLY a JSON object and nothing else: no prose, no code fence, "
        + "and do not repeat these instructions."
    ]
    if !guide.isEmpty {
      lines.append("Fields:")
      lines.append(contentsOf: guide)
    }
    lines.append("Use exactly this shape, replacing every <...> placeholder with a real value:")
    lines.append(hint.skeleton(root, depth: 0))
    return lines.joined(separator: "\n")
  }

  /// A nested schema echo that no unwrapping can repair: a top-level field whose
  /// value is a schema node (`{"type": "array", "items": ...}`) where the schema
  /// asks for a non-object. Returns the offending field so the caller can fail
  /// loudly instead of handing FM a value it cannot decode.
  static func schemaEchoField(in json: String, schemaJSON: String?) -> String? {
    guard let schemaJSON,
      let schemaData = schemaJSON.data(using: .utf8),
      let schema = (try? JSONSerialization.jsonObject(with: schemaData)) as? [String: Any],
      let properties = schema["properties"] as? [String: Any],
      let data = json.data(using: .utf8),
      let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    else { return nil }
    let hint = SchemaHint(definitions: schema["$defs"] as? [String: Any] ?? [:])
    for key in properties.keys.sorted() {
      guard let expected = (properties[key] as? [String: Any]).map(hint.resolve),
        let expectedType = expected["type"] as? String, expectedType != "object",
        let value = object[key] as? [String: Any],
        (value["type"] as? String) == expectedType
      else { continue }
      return key
    }
    return nil
  }

  /// Walks an encoded `GenerationSchema` (JSON Schema with `$defs` / `$ref`,
  /// `x-order`, `enum`, `anyOf`, `required`) to render the field guide and the
  /// placeholder instance used by `guidedInstructions`.
  struct SchemaHint {
    let definitions: [String: Any]
    private static let maxDepth = 6

    func resolve(_ node: [String: Any]) -> [String: Any] {
      if let ref = node["$ref"] as? String, let name = ref.split(separator: "/").last,
        let target = definitions[String(name)] as? [String: Any]
      {
        return target
      }
      if let anyOf = node["anyOf"] as? [[String: Any]], let first = anyOf.first {
        return resolve(first)
      }
      return node
    }

    /// Properties in `x-order` (declaration order), then any the order missed.
    func orderedProperties(_ node: [String: Any]) -> [(String, [String: Any])] {
      guard let properties = node["properties"] as? [String: Any] else { return [] }
      let declared = (node["x-order"] as? [String]) ?? []
      let keys = declared.filter { properties[$0] != nil }
        + properties.keys.sorted().filter { !declared.contains($0) }
      return keys.compactMap { key in (properties[key] as? [String: Any]).map { (key, $0) } }
    }

    func skeleton(_ raw: [String: Any], depth: Int) -> String {
      guard depth < Self.maxDepth else { return "<value>" }
      let node = resolve(raw)
      if let values = node["enum"] as? [Any] {
        return "\"<" + values.map { "\($0)" }.joined(separator: " | ") + ">\""
      }
      switch node["type"] as? String {
      case "object":
        let fields = orderedProperties(node).map { key, child in
          "\"\(key)\": \(skeleton(child, depth: depth + 1))"
        }
        return "{" + fields.joined(separator: ", ") + "}"
      case "array":
        let item = (node["items"] as? [String: Any]).map { skeleton($0, depth: depth + 1) }
        return "[\(item ?? "<value>")]"
      case "string": return "\"<string>\""
      case "integer": return "<integer>"
      case "number": return "<number>"
      case "boolean": return "<true or false>"
      default: return "<value>"
      }
    }

    func describe(_ raw: [String: Any], path: String, into lines: inout [String], depth: Int) {
      guard depth < Self.maxDepth else { return }
      let node = resolve(raw)
      let required = Set(node["required"] as? [String] ?? [])
      for (key, rawChild) in orderedProperties(node) {
        let child = resolve(rawChild)
        let name = path.isEmpty ? key : "\(path).\(key)"
        var parts = [typeName(child)]
        if !required.contains(key) { parts.append("optional") }
        var line = "- \(name) (\(parts.joined(separator: ", ")))"
        if let description = (rawChild["description"] ?? child["description"]) as? String,
          !description.isEmpty
        {
          line += ": \(description)"
        }
        lines.append(line)
        switch child["type"] as? String {
        case "object":
          describe(child, path: name, into: &lines, depth: depth + 1)
        case "array":
          if let items = child["items"] as? [String: Any],
            (resolve(items)["type"] as? String) == "object"
          {
            describe(items, path: name + "[]", into: &lines, depth: depth + 1)
          }
        default: break
        }
      }
    }

    func typeName(_ node: [String: Any]) -> String {
      if let values = node["enum"] as? [Any] {
        return "one of " + values.map { "\"\($0)\"" }.joined(separator: ", ")
      }
      switch node["type"] as? String {
      case "array":
        let item = (node["items"] as? [String: Any]).map { typeName(resolve($0)) }
        return "array of \(item ?? "value")"
      case let other?: return other
      default: return "value"
      }
    }
  }

  /// Extract the first balanced JSON object from model text (strips prose/fences).
  private static func extractJSONObject(from text: String) -> String? {
    guard let start = text.firstIndex(of: "{") else { return nil }
    var depth = 0
    var inString = false
    var escaped = false
    var idx = start
    while idx < text.endIndex {
      let ch = text[idx]
      if inString {
        if escaped { escaped = false } else if ch == "\\" { escaped = true }
        else if ch == "\"" { inString = false }
      } else if ch == "\"" {
        inString = true
      } else if ch == "{" {
        depth += 1
      } else if ch == "}" {
        depth -= 1
        if depth == 0 { return String(text[start...idx]) }
      }
      idx = text.index(after: idx)
    }
    return nil
  }

  // MARK: Transcript → LiteRT messages

  private struct Plan {
    let systemMessage: Message?
    let history: [Message]
    let prompt: Message
  }

  /// Split the FM transcript into a system message, prior turns (history), and
  /// the message to generate from. The generation trigger is the last `.prompt`
  /// OR (in a tool round-trip) the last `.toolOutput`.
  private static func plan(
    from transcript: Transcript, schemaJSON: String?, tools: [Transcript.ToolDefinition]
  ) throws -> Plan {
    let entries = Array(transcript)
    guard
      let triggerIndex = entries.lastIndex(where: {
        switch $0 {
        case .prompt, .toolOutput: return true
        default: return false
        }
      })
    else {
      throw LiteRTFMError.noPrompt
    }

    var systemText: [String] = []
    if !tools.isEmpty { systemText.append(toolInstructions(tools)) }
    var history: [Message] = []
    var trigger: Message?

    for (i, entry) in entries.enumerated() {
      let isTrigger = (i == triggerIndex)
      switch entry {
      case .instructions(let instructions):
        systemText.append(text(of: instructions.segments))
      case .prompt(let p):
        var c = contents(of: p.segments)
        if isTrigger, let schemaJSON, !schemaJSON.isEmpty {
          c.append(.text("\n\n" + guidedInstructions(fromSchemaJSON: schemaJSON)))
        }
        let message = Message(contents: c, role: .user)
        if isTrigger { trigger = message } else { history.append(message) }
      case .response(let r):
        history.append(Message(contents: [.text(text(of: r.segments))], role: .model))
      case .toolOutput(let output):
        let result = text(of: output.segments)
        // Keep the chain open: a bare "answer the user" here makes the model
        // stop after one call even when the request needs several tools.
        let message = Message(
          "Tool \"\(output.toolName)\" returned: \(result)\n"
            + "If more tool calls are needed to finish the user's request, call the "
            + "next tool; otherwise answer the user using the results.",
          role: .user)
        if isTrigger { trigger = message } else { history.append(message) }
      case .toolCalls(let toolCalls):
        // Render past calls the way the model is asked to make them. A literal
        // "[the assistant called a tool]" placeholder here gets parroted back
        // as the next answer at temperature 0, killing multi-step chains.
        let rendered = toolCalls
          .map { "{\"tool_call\": {\"name\": \"\($0.toolName)\"}}" }
          .joined(separator: "\n")
        history.append(Message(rendered, role: .model))
      case .reasoning:
        break
      @unknown default:
        break
      }
    }

    let system = systemText.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    return Plan(
      systemMessage: system.isEmpty ? nil : Message(system, role: .system),
      history: history,
      prompt: trigger!  // guaranteed by triggerIndex
    )
  }

  /// Arguments are shown as a minimal example object, NOT the raw
  /// GenerationSchema JSON — small models imitate whatever shape they see, and
  /// a schema dump gets echoed back as nested schema-shaped "arguments".
  private static func toolInstructions(_ tools: [Transcript.ToolDefinition]) -> String {
    var lines = ["You can call tools to help answer the user. Available tools:"]
    for tool in tools {
      let schemaJSON = (try? encodeSchema(tool.parameters)) ?? "{}"
      let hint = argumentsHint(fromSchemaJSON: schemaJSON) ?? schemaJSON
      lines.append("- \(tool.name): \(tool.description). Call it with arguments like: \(hint)")
    }
    lines.append(
      "To call a tool, reply with ONLY this JSON and nothing else: "
        + "{\"tool_call\": {\"name\": \"<tool name>\", \"arguments\": { ... }}}. "
        + "If no tool is needed, answer the user directly.")
    lines.append(
      "Call at most one tool per reply. Never ask the user a follow-up question — "
        + "if a detail is missing, choose a sensible value yourself.")
    return lines.joined(separator: "\n")
  }

  /// Parse a tool call from model output, if present and naming a known tool.
  /// Accepts the instructed JSON shape and, as a fallback, Gemma's native
  /// function-calling syntax (`<|tool_call>call:name{arg: "value"}<tool_call|>`),
  /// which fine-tuned checkpoints sometimes revert to despite the instructions.
  private static func parseToolCall(from text: String, tools: [Transcript.ToolDefinition])
    -> (name: String, arguments: String)?
  {
    // The model may batch several {"tool_call": …} objects (one per line)
    // despite being asked for one per reply — parse the FIRST parseable one;
    // FM feeds the result back and the model re-issues the rest next round.
    var candidates = [text]
    if text.contains("\n") {
      candidates = text.split(separator: "\n", omittingEmptySubsequences: true).map(String.init)
        + candidates
    }
    for candidate in candidates {
      if let hit = parseSingleToolCall(from: candidate, tools: tools) { return hit }
    }
    return parseNativeToolCall(from: text, tools: tools)
  }

  /// Rewrite object keys to proper quoted form, whether the model wrote them
  /// bare ({area: …}), half-quoted ({area": …} — an ODD quote count that also
  /// derails the string-aware brace scanners), or fully quoted. Run BEFORE
  /// any structural extraction.
  private static func quotedKeys(_ text: String) -> String {
    text.replacingOccurrences(
      of: #"([{,]\s*)"?([A-Za-z_][A-Za-z0-9_]*)"?(\s*:)"#,
      with: "$1\"$2\"$3",
      options: .regularExpression)
  }

  /// The instructed JSON shape, from one candidate chunk (with brace repair).
  private static func parseSingleToolCall(from rawText: String, tools: [Transcript.ToolDefinition])
    -> (name: String, arguments: String)?
  {
    let text = quotedKeys(rawText)
    guard let json = extractJSONObject(from: text) ?? repairedJSONObject(from: text),
      let data = json.data(using: .utf8),
      let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
      let call = obj["tool_call"] as? [String: Any],
      let name = call["name"] as? String,
      tools.contains(where: { $0.name == name })
    else { return nil }
    let args = call["arguments"] ?? [String: Any]()
    let argsData = (try? JSONSerialization.data(withJSONObject: args)) ?? Data("{}".utf8)
    let raw = String(data: argsData, encoding: .utf8) ?? "{}"
    return (name, normalizedArguments(from: raw))
  }

  /// Gemma-native fallback: `call:<name>` followed by a balanced `{…}` argument
  /// object. Bare identifier keys are quoted so `{text: "hi"}` parses as JSON.
  private static func parseNativeToolCall(from text: String, tools: [Transcript.ToolDefinition])
    -> (name: String, arguments: String)?
  {
    guard let marker = text.range(of: "call:") else { return nil }
    let after = quotedKeys(String(text[marker.upperBound...]))
    let name = String(after.prefix(while: { $0.isLetter || $0.isNumber || $0 == "_" }))
    guard tools.contains(where: { $0.name == name }) else { return nil }
    guard let argsRaw = extractJSONObject(from: after) ?? repairedJSONObject(from: after) else {
      return (name, "{}")
    }
    return (name, normalizedArguments(from: argsRaw))
  }

  /// Build a minimal example arguments object (`{"city": "<value>"}`) from an
  /// encoded GenerationSchema, for the tool instructions.
  private static func argumentsHint(fromSchemaJSON json: String) -> String? {
    guard let data = json.data(using: .utf8),
      let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
      let properties = obj["properties"] as? [String: Any]
    else { return nil }
    if properties.isEmpty { return "{}" }
    let fields = properties.keys.sorted().map { "\"\($0)\": \"<value>\"" }
    return "{" + fields.joined(separator: ", ") + "}"
  }

  /// Salvage an argument object whose closing brace(s) were cut off: take the
  /// first `{` through the LAST `}` present and append the closers the
  /// string-aware depth scan says are missing.
  private static func repairedJSONObject(from text: String) -> String? {
    guard let start = text.firstIndex(of: "{"),
      let lastBrace = text.lastIndex(of: "}"),
      lastBrace > start
    else { return nil }
    let end = text.index(after: lastBrace)
    var depth = 0
    var inString = false
    var escaped = false
    var idx = start
    while idx < end {
      let ch = text[idx]
      if inString {
        if escaped { escaped = false } else if ch == "\\" { escaped = true }
        else if ch == "\"" { inString = false }
      } else if ch == "\"" {
        inString = true
      } else if ch == "{" {
        depth += 1
      } else if ch == "}" {
        depth -= 1
      }
      idx = text.index(after: idx)
    }
    guard depth >= 0 else { return nil }
    return String(text[start..<end]) + String(repeating: "}", count: depth)
  }

  /// Best-effort cleanup of model-written arguments: quote bare identifier
  /// keys (`{text: "hi"}`), then unwrap a "schema echo" — a field whose value
  /// is an object nesting the same field (`{"message": {"message": "hi",
  /// "x-order": …}}`), which small models produce by imitating the schema.
  private static func normalizedArguments(from raw: String) -> String {
    for candidate in [raw, quotedKeys(raw)] {
      guard let data = candidate.data(using: .utf8),
        var obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
      else { continue }
      for (key, value) in obj {
        if let inner = value as? [String: Any], let unwrapped = inner[key] {
          obj[key] = unwrapped
        }
      }
      if let out = try? JSONSerialization.data(withJSONObject: obj),
        let string = String(data: out, encoding: .utf8) {
        return string
      }
    }
    return "{}"
  }

  private static func text(of segments: [Transcript.Segment]) -> String {
    segments.compactMap { segment in
      if case .text(let t) = segment { return t.content } else { return nil }
    }.joined(separator: " ")
  }

  private static func encodeSchema(_ schema: GenerationSchema) throws -> String {
    let data = try JSONEncoder().encode(schema)
    return String(data: data, encoding: .utf8) ?? ""
  }

  /// Translate the caller's FM `GenerationOptions` into a LiteRT `SamplerConfig`,
  /// so `temperature` / `.greedy` / `.random(top:)` / `.random(probabilityThreshold:)`
  /// are honored instead of overridden. Structured output (guided / tools) is
  /// parsed as JSON, so it's forced near-deterministic regardless.
  private static func sampler(for options: GenerationOptions, structured: Bool) -> SamplerConfig? {
    if structured { return try? SamplerConfig(topK: 1, topP: 1.0, temperature: 0.0) }

    var topK = 40
    var topP: Float = 0.95
    var temperature = Float(options.temperature ?? 0.8)

    if let kind = options.samplingMode?.kind {
      switch kind {
      case .greedy:
        topK = 1
        temperature = 0.0
      case .randomTopK(let k, _):
        topK = k
      case .randomProbabilityThreshold(let threshold, _):
        topP = Float(threshold)
      @unknown default:
        break
      }
    }
    return try? SamplerConfig(topK: topK, topP: topP, temperature: temperature)
  }

  /// Map FM segments to LiteRT content: text and image attachments. (Audio and
  /// video have no FM transcript segment since Xcode 27 beta 5 dropped
  /// `Transcript.CustomSegment`.)
  private static func contents(of segments: [Transcript.Segment]) -> [Content] {
    var out: [Content] = []
    for segment in segments {
      switch segment {
      case .text(let t):
        if !t.content.isEmpty { out.append(.text(t.content)) }
      case .attachment(let attachment):
        if case .image(let image) = attachment.content {
          if let png = pngData(from: image.cgImage) {
            out.append(.imageData(png))
          } else {
            // Don't fail the turn, but leave a trace: a silently missing image
            // makes the model's answer look wrong for no visible reason.
            logger.warning("Dropping an image attachment: PNG encoding failed.")
          }
        }
      case .structure:
        break
      @unknown default:
        break
      }
    }
    return out.isEmpty ? [.text("")] : out
  }
}

/// Errors specific to the Foundation Models bridge.
@available(iOS 27.0, macOS 27.0, *)
public enum LiteRTFMError: Error, LocalizedError {
  case noPrompt
  /// Guided generation: the model returned the schema of `field` instead of a
  /// value for it, and the reply cannot be decoded into the requested type.
  case schemaEcho(field: String)

  public var errorDescription: String? {
    switch self {
    case .noPrompt: return "The transcript contains no prompt to respond to."
    case .schemaEcho(let field):
      return "Guided generation failed: the model returned the schema for \"\(field)\" "
        + "instead of a value."
    }
  }
}

// MARK: - Engine cache + lazy engine

/// Process-wide cache of one `LazyEngine` per engine configuration, so multiple
/// FM executors / sessions sharing an engine configuration share a single loaded
/// engine — even when their conversation-level settings (`visualTokenBudget`)
/// differ.
@available(iOS 27.0, macOS 27.0, *)
final class EngineCache: @unchecked Sendable {
  static let shared = EngineCache()
  private let lock = NSLock()
  private var engines: [EngineConfig: LazyEngine] = [:]

  /// How many distinct engines are currently held. Not part of the public API.
  var count: Int {
    lock.lock()
    defer { lock.unlock() }
    return engines.count
  }

  func engine(for engineConfig: EngineConfig) -> LazyEngine {
    lock.lock()
    defer { lock.unlock() }
    if let engine = engines[engineConfig] { return engine }
    let engine = LazyEngine(engineConfig: engineConfig)
    engines[engineConfig] = engine
    return engine
  }

  func purgeAll() async {
    for engine in drain() { await engine.release() }
  }

  private func drain() -> [LazyEngine] {
    lock.lock()
    defer { lock.unlock() }
    let all = Array(engines.values)
    engines.removeAll()
    return all
  }
}

/// Lazily creates and caches the LiteRT engine. The FM executor's `init` is
/// synchronous but engine initialization is async, so we defer it to the first
/// `respond` (which is async) and memoize the result.
@available(iOS 27.0, macOS 27.0, *)
actor LazyEngine {
  private let engineConfig: EngineConfig
  private var engineTask: Task<Engine, Error>?
  private var warmed = false

  init(engineConfig: EngineConfig) {
    self.engineConfig = engineConfig
  }

  func ready() async throws -> Engine {
    // Memoize the in-flight creation task, not the finished engine: awaiting
    // `initialize()` suspends the actor, so two concurrent first calls (e.g.
    // `prewarm` plus an immediate `respond`) would otherwise both see no engine
    // and load the multi-GB weights twice.
    if let engineTask { return try await engineTask.value }
    let engineConfig = self.engineConfig
    let task = Task {
      let created = Engine(engineConfig: engineConfig)
      try await created.initialize()
      return created
    }
    engineTask = task
    do {
      return try await task.value
    } catch {
      // A failed initialization stays retryable on the next call.
      if engineTask == task { engineTask = nil }
      throw error
    }
  }

  func prewarmed() async throws {
    let engine = try await ready()
    if warmed { return }
    warmed = true
    let warmup = try await engine.createConversation()
    for try await _ in warmup.sendMessageStream(Message("Hi")) {}
  }

  func release() {
    engineTask = nil
    warmed = false
  }
}

#endif
