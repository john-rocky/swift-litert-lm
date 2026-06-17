// LiteRTDemo — "FM API" mode screen.
//
// Proves, on-screen, that LiteRT-LM is driven through Apple's *real* Foundation
// Models API — the same `LanguageModelSession` you'd use for Apple Intelligence,
// but the model is Google's Gemma 4 via LiteRT, 100% on-device.
//
// Every card shows its **input** (an editable prompt — type your own and re-Run
// to prove a real LLM is processing it, not a canned script) and what the FM API
// did with it:
//   • respond(to:)        — plain generation, same call as Apple's own model.
//   • respond(generating:)— you declare a Swift type; FM returns a filled, typed
//                           value (not text you have to parse).
//   • Tool calling        — the model decides to call your Swift function; FM
//                           runs it and feeds the result back into the answer.

#if canImport(FoundationModels)

import SwiftUI
import FoundationModels
import LiteRTFoundation

// MARK: - Generable type FM fills for guided generation

/// A general-purpose structured answer, so the guided demo works for *any*
/// question the user types — FM decodes the model's output into these typed
/// fields (no JSON parsing in app code).
@available(iOS 27.0, macOS 27.0, *)
@Generable
struct StructuredAnswer {
  @Guide(description: "A short title for the answer")
  var title: String
  @Guide(description: "The key points of the answer, each a short phrase")
  var points: [String]
}

// MARK: - A tool that records when Foundation Models invokes it

/// The model must read the user's prompt, extract a city, and call this — so
/// changing the prompt (e.g. Tokyo → Paris) and seeing the city flow through is
/// proof the LLM actually processed the input. The "22°C and sunny" string is
/// data only the tool knows; its appearance in the answer proves FM ran the tool.
@available(iOS 27.0, macOS 27.0, *)
struct TracedTemperatureTool: FoundationModels.Tool {
  let name = "get_temperature"
  let description = "Get the current temperature for a given city."
  let onCall: @Sendable (String) -> Void

  @Generable
  struct Arguments {
    @Guide(description: "The city name")
    var city: String
  }

  func call(arguments: Arguments) async throws -> String {
    onCall(arguments.city)
    return "The temperature in \(arguments.city) is 22°C and sunny."
  }
}

// MARK: - Screen

@available(iOS 27.0, macOS 27.0, *)
struct FMModeView: View {
  @StateObject private var vm = FMViewModel()
  @Environment(\.dismiss) private var dismiss
  @FocusState private var focused: Bool

  var body: some View {
    NavigationStack {
      ScrollView {
        VStack(spacing: 16) {
          banner
          textCard
          guidedCard
          toolCard
          Text(
            "Every API on this screen is Apple's FoundationModels. The only "
              + "LiteRT-specific code is the `model:` argument.")
            .font(.footnote).foregroundStyle(.secondary)
            .multilineTextAlignment(.center).padding(.top, 4)
        }
        .padding()
      }
      .scrollDismissesKeyboard(.interactively)
      .navigationTitle("Foundation Models API")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .topBarTrailing) { Button("Done") { dismiss() } }
        ToolbarItem(placement: .keyboard) {
          HStack { Spacer(); Button("Done") { focused = false } }
        }
      }
      .overlay { if !vm.isReady { loadingOverlay } }
    }
    .task { await vm.load() }
  }

  // MARK: Cards

  private var banner: some View {
    VStack(alignment: .leading, spacing: 8) {
      Label("Apple's Foundation Models API — running Gemma 4", systemImage: "cpu")
        .font(.subheadline.bold())
      Text("The same `LanguageModelSession` API as Apple Intelligence — but the "
        + "model is Google's Gemma 4 via LiteRT, 100% on-device.")
        .font(.caption).foregroundStyle(.secondary)
      Text("let model   = try await LiteRTLanguageModel(.gemma4_E2B)\n"
        + "let session = LanguageModelSession(model: model)")
        .font(.caption.monospaced())
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.tertiarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
    .cardStyle()
  }

  private var textCard: some View {
    DemoCard(
      title: "respond(to:)", subtitle: "Plain generation — same call as Apple's own model.",
      systemImage: "text.bubble", prompt: $vm.textPrompt,
      running: vm.running == .text, disabled: vm.running != nil, focused: $focused
    ) {
      Task { await vm.runText() }
    } content: {
      if !vm.textOut.isEmpty {
        Text(vm.textOut).font(.callout).textSelection(.enabled)
      }
    }
  }

  private var guidedCard: some View {
    DemoCard(
      title: "respond(generating:)",
      subtitle: "You declare a Swift type; FM returns a filled, typed value — not text to parse.",
      systemImage: "checklist", prompt: $vm.guidedPrompt,
      running: vm.running == .guided, disabled: vm.running != nil, focused: $focused
    ) {
      Task { await vm.runGuided() }
    } content: {
      if let answer = vm.guidedAnswer {
        VStack(alignment: .leading, spacing: 8) {
          Text(answer.title).font(.callout.bold())
          ForEach(answer.points, id: \.self) { p in
            Text(p).font(.caption.bold())
              .padding(.horizontal, 10).padding(.vertical, 6)
              .background(swatch(for: p)).foregroundStyle(.white)
              .clipShape(Capsule())
          }
          Label("Typed StructuredAnswer { title: String; points: [String] } — decoded by FM",
            systemImage: "checkmark.seal.fill")
            .font(.caption2).foregroundStyle(.green)
        }
      } else if let err = vm.guidedError {
        Label(err, systemImage: "exclamationmark.triangle.fill")
          .font(.caption).foregroundStyle(.red)
      }
    }
  }

  private var toolCard: some View {
    DemoCard(
      title: "Tool calling",
      subtitle: "The model reads your prompt, calls your Swift function; FM runs it and feeds it back.",
      systemImage: "wrench.and.screwdriver", prompt: $vm.toolPrompt,
      running: vm.running == .tool, disabled: vm.running != nil, focused: $focused
    ) {
      Task { await vm.runTool() }
    } content: {
      VStack(alignment: .leading, spacing: 8) {
        ForEach(vm.toolTrace, id: \.self) { line in
          Text(line).font(.caption.monospaced()).foregroundStyle(.orange)
        }
        if !vm.toolAnswer.isEmpty {
          Text(vm.toolAnswer).font(.callout).textSelection(.enabled)
          if vm.toolAnswer.contains("22") {
            Label("Answer carries the tool's data (22°C) — FM fed it back",
              systemImage: "arrow.uturn.backward.circle.fill")
              .font(.caption2).foregroundStyle(.green)
          }
        }
      }
    }
  }

  private var loadingOverlay: some View {
    ZStack {
      Color(.systemBackground).opacity(0.85).ignoresSafeArea()
      VStack(spacing: 10) {
        ProgressView()
        Text(vm.loadError ?? "Bringing up the LiteRT backend…")
          .font(.callout).foregroundStyle(vm.loadError == nil ? Color.secondary : Color.red)
          .multilineTextAlignment(.center)
      }
      .padding()
    }
  }

  private func swatch(for value: String) -> Color {
    switch value.lowercased() {
    case let c where c.contains("red"): return .red
    case let c where c.contains("green"): return .green
    case let c where c.contains("blue"): return .blue
    default: return .gray
    }
  }
}

// MARK: - Reusable demo card (editable input + result)

@available(iOS 27.0, macOS 27.0, *)
private struct DemoCard<Content: View>: View {
  let title: String
  let subtitle: String
  let systemImage: String
  @Binding var prompt: String
  let running: Bool
  let disabled: Bool
  var focused: FocusState<Bool>.Binding
  let action: () -> Void
  @ViewBuilder let content: () -> Content

  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack {
        Label(title, systemImage: systemImage).font(.subheadline.bold())
        Spacer()
        Button(action: action) {
          if running { ProgressView().controlSize(.small) }
          else { Text("Run").bold() }
        }
        .buttonStyle(.borderedProminent).controlSize(.small)
        .disabled(disabled)
      }
      Text(subtitle).font(.caption).foregroundStyle(.secondary)

      // Editable input: shows exactly what's sent, and lets you change it so you
      // can confirm a real LLM is processing your words (not a fixed script).
      VStack(alignment: .leading, spacing: 4) {
        Text("INPUT").font(.caption2.bold()).foregroundStyle(.secondary)
        TextField("prompt", text: $prompt, axis: .vertical)
          .font(.callout).lineLimit(1...4)
          .textFieldStyle(.plain)
          .focused(focused)
          .padding(8)
          .frame(maxWidth: .infinity, alignment: .leading)
          .background(Color(.tertiarySystemBackground))
          .clipShape(RoundedRectangle(cornerRadius: 8))
          .disabled(disabled)
      }

      content()
    }
    .cardStyle()
  }
}

private extension View {
  func cardStyle() -> some View {
    self
      .padding()
      .frame(maxWidth: .infinity, alignment: .leading)
      .background(Color(.secondarySystemBackground))
      .clipShape(RoundedRectangle(cornerRadius: 14))
  }
}

// MARK: - View model

@available(iOS 27.0, macOS 27.0, *)
@MainActor
final class FMViewModel: ObservableObject {
  enum Running { case text, guided, tool }

  @Published var isReady = false
  @Published var loadError: String?
  @Published var running: Running?

  // Editable inputs (each card sends its own).
  @Published var textPrompt = "Explain on-device AI in one sentence."
  @Published var guidedPrompt = "List the three additive primary colors."
  @Published var toolPrompt = "What is the temperature in Tokyo right now?"

  @Published var textOut = ""
  @Published var guidedAnswer: StructuredAnswer?
  @Published var guidedError: String?
  @Published var toolTrace: [String] = []
  @Published var toolAnswer = ""

  private var model: LiteRTLanguageModel?

  func load() async {
    guard model == nil else { return }
    // FM mode doesn't display tok/s; disabling the benchmark counters also avoids
    // the engine's no-sampler prewarm tripping `output_buffer_dup` if Easy mode
    // left the global benchmark flag on.
    ExperimentalFlags.enableBenchmark = false
    do {
      self.model = try await LiteRTLanguageModel(.gemma4_E2B)
      isReady = true
    } catch {
      loadError = error.localizedDescription
    }
  }

  func runText() async {
    guard let model, running == nil else { return }
    let prompt = textPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !prompt.isEmpty else { return }
    running = .text; defer { running = nil }
    textOut = ""
    do {
      // A fresh session per run so each Run is self-contained — no prior turn in
      // the transcript steering the result.
      let session = LanguageModelSession(model: model)
      let r = try await session.respond(to: prompt)
      textOut = r.content
    } catch {
      textOut = "[error] \(error.localizedDescription)"
    }
  }

  func runGuided() async {
    guard let model, running == nil else { return }
    let prompt = guidedPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !prompt.isEmpty else { return }
    running = .guided; defer { running = nil }
    guidedAnswer = nil; guidedError = nil
    // Fresh session (clean transcript) + one retry: schema-in-prompt JSON can
    // occasionally come back unparseable on a 2B model.
    for attempt in 1...2 {
      do {
        let session = LanguageModelSession(model: model)
        let r = try await session.respond(generating: StructuredAnswer.self) { prompt }
        guidedAnswer = r.content
        return
      } catch {
        if attempt == 2 { guidedError = error.localizedDescription }
      }
    }
  }

  func runTool() async {
    guard let model, running == nil else { return }
    let prompt = toolPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !prompt.isEmpty else { return }
    running = .tool; defer { running = nil }
    toolTrace = []; toolAnswer = ""
    // The trace closure hops to the main actor when FM invokes the tool, so the
    // round-trip (and the city the model extracted) is visible on screen.
    let tool = TracedTemperatureTool { [weak self] city in
      Task { @MainActor in
        self?.toolTrace.append("🔧 get_temperature(city: \"\(city)\")  →  22°C and sunny")
      }
    }
    let session = LanguageModelSession(model: model, tools: [tool])
    do {
      let r = try await session.respond(to: prompt)
      toolAnswer = r.content
    } catch {
      toolAnswer = "[error] \(error.localizedDescription)"
    }
  }
}

#endif
