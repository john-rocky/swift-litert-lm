// LiteRTDemo — what an app sees when things go wrong, plus a two-turn chat.
//
// Launch with LITERT_FAILURES=1 for every failure path, or a comma list of ids
// (a,b1,b2,b3,c,d,e) for some of them. LITERT_CHAT_CHECK=1 adds the chat leg:
// two turns on `LiteRTChat(.gemma4_E2B)`, the second referring to the first.
// LITERT_MODEL_PATH points at a model file already on disk instead of the app's
// storage directory (a macOS test tool has no app container).
//
// Each path logs one line on the screen and on stdout ("FP:"), and the run ends
// with `### DEVICE: <machine>` and `### FAILURE-PATHS: <n> PASS`. A path passes
// when it throws, with a non-empty `localizedDescription`, and a normal turn on a
// new chat works afterwards. Paths d and e also record a turn on the chat that
// failed; that turn does not count. Path e (cancel) is recorded but not counted.

import Foundation
import LiteRTFoundation
import SwiftUI
import os

enum FailurePathsSelfTest {
  private static let logger = Logger(subsystem: "com.example.litertdemo", category: "FP")
  private static let environment = ProcessInfo.processInfo.environment
  /// A path that takes longer than this has hung: say so and stop the run.
  private static let hangLimitSeconds = 900

  static var isRequested: Bool {
    environment["LITERT_FAILURES"] != nil || environment["LITERT_CHAT_CHECK"] != nil
  }

  static func run() async {
    log("start — device=\(deviceName()) OS=\(ProcessInfo.processInfo.operatingSystemVersionString)")
    let model: URL
    do {
      model = try await realModelURL()
      log("model \(model.path)")
    } catch {
      log("FATAL could not obtain model: \(error.localizedDescription)")
      return
    }

    if environment["LITERT_CHAT_CHECK"] != nil {
      await guarded("chat") { await chatCheck(model: model) }
    }

    if let selection = environment["LITERT_FAILURES"] {
      let ids = selection == "1" ? nil : Set(selection.split(separator: ",").map(String.init))
      let scratch = FileManager.default.temporaryDirectory
        .appendingPathComponent("litert-failure-paths-\(UUID().uuidString)", isDirectory: true)
      defer { try? FileManager.default.removeItem(at: scratch) }
      var outcomes: [String: Bool] = [:]
      do {
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        for path in paths(scratch: scratch, model: model) where ids?.contains(path.id) ?? true {
          outcomes[path.id] = await guarded(path.id) { await runPath(path, model: model) }
        }
      } catch {
        log("FATAL could not prepare the test files: \(error.localizedDescription)")
      }
      // Path b is one path over three files; it counts when all three ran and passed.
      let groups: [(String, [String])] = [
        ("a", ["a"]), ("b", ["b1", "b2", "b3"]), ("c", ["c"]), ("d", ["d"]),
      ]
      let passed = groups.filter { _, members in members.allSatisfy { outcomes[$0] == true } }
      emit("### DEVICE: \(deviceName())")
      emit("### FAILURE-PATHS: \(passed.count) PASS")
    }
    log("DONE")
  }

  // MARK: Failure paths

  private struct FailurePath {
    let id: String
    let name: String
    /// False for a path that is recorded but not part of the pass count.
    var counted = true
    let body: () async throws -> String
  }

  private static func paths(scratch: URL, model: URL) -> [FailurePath] {
    [
      FailurePath(id: "a", name: "model file missing") {
        let missing = scratch.appendingPathComponent("does-not-exist.litertlm")
        _ = try await LiteRTChat(modelFileURL: missing)
        return "returned a chat"
      },
      FailurePath(id: "b1", name: "random bytes") {
        let file = scratch.appendingPathComponent("random.litertlm")
        var bytes = [UInt8](repeating: 0, count: 1 << 20)
        for i in bytes.indices { bytes[i] = UInt8.random(in: 0...255) }
        try Data(bytes).write(to: file)
        _ = try await LiteRTChat(modelFileURL: file)
        return "returned a chat"
      },
      FailurePath(id: "b2", name: "first 64 MB of the real model") {
        let file = scratch.appendingPathComponent("truncated.litertlm")
        let source = try FileHandle(forReadingFrom: model)
        defer { try? source.close() }
        let head = try source.read(upToCount: 64 << 20) ?? Data()
        try head.write(to: file)
        _ = try await LiteRTChat(modelFileURL: file)
        return "returned a chat"
      },
      FailurePath(id: "b3", name: "a directory") {
        let directory = scratch.appendingPathComponent("directory.litertlm", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        _ = try await LiteRTChat(modelFileURL: directory)
        return "returned a chat"
      },
      FailurePath(id: "c", name: "not enough RAM") {
        let ram = Int64(ProcessInfo.processInfo.physicalMemory)
        _ = try await LiteRTChat(modelFileURL: model, minimumDeviceRAM: ram + (1 << 30))
        return "returned a chat"
      },
      FailurePath(id: "d", name: "input longer than maxTokens") {
        try await longInput(model: model)
      },
      FailurePath(id: "e", name: "cancel during a stream", counted: false) {
        try await cancelDuringStream(model: model)
      },
    ]
  }

  /// Run one path and log its line. Returns whether it passed.
  private static func runPath(_ path: FailurePath, model: URL) async -> Bool {
    log("(\(path.id)) \(path.name): running")
    var description: String?
    var note = ""
    var nextTurnWorked: Bool?
    do {
      note = try await path.body()
    } catch let failure as RecordedFailure {
      description = failure.description
      note = failure.notes
      nextTurnWorked = failure.newChatWorked
    } catch {
      description = error.localizedDescription
    }
    var nextTurn = ""
    if nextTurnWorked == nil {
      do {
        nextTurn = "; next turn on a new chat: \(oneLine(try await normalTurn(model: model)))"
        nextTurnWorked = true
      } catch {
        nextTurn = "; next turn on a new chat FAILED: \(oneLine(error.localizedDescription))"
        nextTurnWorked = false
      }
    }
    let passed = !(description ?? "").isEmpty && nextTurnWorked == true
    let verdict = path.counted ? (passed ? "PASS" : "FAIL") : "RECORDED"
    let message = description.map { "threw \"\(oneLine($0))\"" } ?? "did not throw"
    log("(\(path.id)) \(path.name): \(verdict) · \(message)\(note.isEmpty ? "" : "; \(note)")\(nextTurn)")
    return path.counted && passed
  }

  /// What a path that tries the turns after the failure itself has to report.
  private struct RecordedFailure: Error {
    let description: String
    let notes: String
    let newChatWorked: Bool
  }

  /// An input past the KV budget, then a short turn on the same chat and on a new one.
  private static func longInput(model: URL) async throws -> String {
    let sentence = "The quick brown fox jumps over the lazy dog near the quiet river bank. "
    let input = String(repeating: sentence, count: 400)
    var failure = ""
    var notes: [String] = []
    do {
      let chat = try await LiteRTChat(
        modelFileURL: model, modalities: [] as Modality, maxTokens: 2048, prewarm: false)
      do {
        let reply = try await chat.respond(input)
        notes.append("the long turn returned \(reply.count) characters")
      } catch {
        failure = error.localizedDescription
      }
      notes.append(await describeTurn("same chat") { try await chat.respond(shortPrompt) }.line)
    }
    await settle()
    let newChat = await describeTurn("new chat") { try await normalTurn(model: model) }
    notes.append(newChat.line)
    throw RecordedFailure(
      description: failure, notes: notes.joined(separator: "; "), newChatWorked: newChat.worked)
  }

  /// `cancel()` once tokens are flowing, then a short turn on the same chat and on a new one.
  private static func cancelDuringStream(model: URL) async throws -> String {
    var failure = ""
    var notes: [String] = []
    do {
      let chat = try await LiteRTChat(
        modelFileURL: model, modalities: [] as Modality, maxTokens: 2048, prewarm: false)
      var deltas = 0
      do {
        for try await _ in chat.stream("Count from 1 to 500, one number per line.") {
          deltas += 1
          if deltas == 5 { try chat.cancel() }
        }
        notes.append("the stream ended normally after \(deltas) deltas")
      } catch {
        failure = error.localizedDescription
        notes.append("the stream threw after \(deltas) deltas")
      }
      notes.append(await describeTurn("same chat") { try await chat.respond(shortPrompt) }.line)
    }
    await settle()
    let newChat = await describeTurn("new chat") { try await normalTurn(model: model) }
    notes.append(newChat.line)
    throw RecordedFailure(
      description: failure, notes: notes.joined(separator: "; "), newChatWorked: newChat.worked)
  }

  private static let shortPrompt = "Reply with one short sentence about the sea."

  /// A normal text turn on a new chat: what an app tries after a failure. The chat
  /// is released before this returns.
  private static func normalTurn(model: URL) async throws -> String {
    let chat = try await LiteRTChat(
      modelFileURL: model, modalities: [] as Modality, maxTokens: 512, prewarm: false)
    let reply = try await chat.respond(shortPrompt)
    guard !reply.isEmpty else { throw EmptyReply() }
    return reply
  }

  private struct EmptyReply: LocalizedError {
    var errorDescription: String? { "the turn ended with an empty reply" }
  }

  private static func describeTurn(
    _ label: String, _ turn: () async throws -> String
  ) async -> (line: String, worked: Bool) {
    do {
      let reply = try await turn()
      return reply.isEmpty
        ? ("\(label): empty reply", false)
        : ("\(label): \(oneLine(reply))", true)
    } catch {
      return ("\(label): threw \"\(oneLine(error.localizedDescription))\"", false)
    }
  }

  // MARK: Chat

  private static func chatCheck(model: URL) async {
    do {
      let storage: URL? = environment["LITERT_MODEL_PATH"] == nil ? nil : model.deletingLastPathComponent()
      let chat = try await LiteRTChat(.gemma4_E2B, storageDirectory: storage, enableBenchmark: true)
      var worked = true
      let turns = [
        "Name one primary color. Answer with the color only.",
        "Which color did you just name? Answer in one short sentence.",
      ]
      for (index, prompt) in turns.enumerated() {
        let reply = try await chat.respond(prompt)
        let info = try chat.lastBenchmark()
        worked = worked && !reply.isEmpty
          && info.lastDecodeTokenCount < LiteRTModel.gemma4_E2B.defaultMaxTokens
        log(String(
          format: "chat turn %d: %@ · decode %d tok · %.1f tok/s · footprint %.0f MB",
          index + 1, oneLine(reply), info.lastDecodeTokenCount, info.lastDecodeTokensPerSecond,
          Double(LiteRTChat.memoryFootprintBytes()) / 1_048_576))
      }
      emit("### CHAT-CHECK: \(worked ? "PASS" : "FAIL")")
    } catch {
      log("chat FAILED: \(error.localizedDescription)")
      emit("### CHAT-CHECK: FAIL")
    }
    await settle()
  }

  // MARK: Helpers

  private static func realModelURL() async throws -> URL {
    if let path = environment["LITERT_MODEL_PATH"] { return URL(fileURLWithPath: path) }
    return URL(fileURLWithPath: try await LiteRTChat.ensureModel(.gemma4_E2B))
  }

  /// Give the native engine a moment to free GPU memory before the next one loads.
  private static func settle() async {
    try? await Task.sleep(nanoseconds: 800_000_000)
  }

  private static func guarded<T>(_ id: String, _ body: () async -> T) async -> T {
    let watchdog = DispatchWorkItem {
      emit("### HANG: (\(id)) after \(hangLimitSeconds) s")
      exit(3)
    }
    DispatchQueue.global().asyncAfter(
      deadline: .now() + .seconds(hangLimitSeconds), execute: watchdog)
    let result = await body()
    watchdog.cancel()
    await settle()
    return result
  }

  private static func oneLine(_ text: String) -> String {
    String(
      text.replacingOccurrences(of: "\n", with: " ")
        .trimmingCharacters(in: .whitespacesAndNewlines).prefix(240))
  }

  private static func log(_ message: String) { emit("FP: \(message)") }

  /// One line on stdout, in the unified log, and on the screen.
  private static func emit(_ line: String) {
    logger.log("\(line, privacy: .public)")
    print(line)
    fflush(stdout)
    Task { @MainActor in FailurePathsOutput.shared.lines.append(line) }
  }

  private static func deviceName() -> String {
    #if os(macOS)
    var size = 0
    sysctlbyname("hw.model", nil, &size, nil, 0)
    var name = [CChar](repeating: 0, count: size)
    sysctlbyname("hw.model", &name, &size, nil, 0)
    return String(cString: name)
    #else
    var info = utsname()
    uname(&info)
    return withUnsafeBytes(of: &info.machine) { raw in
      String(cString: raw.baseAddress!.assumingMemoryBound(to: CChar.self))
    }
    #endif
  }
}

@MainActor
final class FailurePathsOutput: ObservableObject {
  static let shared = FailurePathsOutput()
  @Published var lines: [String] = []
}

/// Runs the self-test and shows its lines as they arrive.
struct FailurePathsRunnerView: View {
  @ObservedObject private var output = FailurePathsOutput.shared
  @State private var started = false

  var body: some View {
    ScrollView {
      Text(output.lines.joined(separator: "\n"))
        .font(.system(.caption, design: .monospaced))
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
    }
    .task {
      guard !started else { return }
      started = true
      await FailurePathsSelfTest.run()
    }
  }
}
