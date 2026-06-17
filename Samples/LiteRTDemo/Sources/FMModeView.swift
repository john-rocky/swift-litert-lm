// LiteRTDemo — "FM API" mode screen.
//
// Proves, on-screen, that LiteRT-LM is driven through Apple's *real* Foundation
// Models API — not a lookalike wrapper. Every type here is from Apple's
// `FoundationModels` framework (`LanguageModelSession`, `@Generable`, `Tool`,
// `GenerationSchema`); the only LiteRT-specific token is the `model:` argument:
//
//     let session = LanguageModelSession(model: LiteRTLanguageModel(.gemma4_E2B))
//
// It demonstrates the two behaviors a wrapper *can't* fake, because Apple's
// runtime orchestrates them:
//   • Guided generation — `respond(generating:)` returns a validated @Generable.
//   • Tool calling       — the model emits a tool call, FM runs the app's Tool
//                          and feeds the result back into the answer.

#if canImport(FoundationModels)

import SwiftUI
import FoundationModels
import LiteRTFoundation

// MARK: - A tool that records when Foundation Models invokes it

/// Same shape as the self-test's tool, but it reports each invocation so the UI
/// can show the round-trip. The canned "22°C and sunny" string is data only the
/// tool knows — if it surfaces in the final answer, FM truly called the tool.
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

  var body: some View {
    NavigationStack {
      ScrollView {
        VStack(spacing: 16) {
          codeCard
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
      .navigationTitle("Foundation Models API")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .topBarTrailing) {
          Button("Done") { dismiss() }
        }
      }
      .overlay { if !vm.isReady { loadingOverlay } }
    }
    .task { await vm.load() }
  }

  // MARK: Cards

  private var codeCard: some View {
    VStack(alignment: .leading, spacing: 8) {
      Label("Driven by Apple's LanguageModelSession", systemImage: "cpu")
        .font(.subheadline.bold())
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
      title: "respond(to:)", subtitle: "Plain generation through the FM API",
      systemImage: "text.bubble", running: vm.running == .text,
      disabled: vm.running != nil
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
      title: "respond(generating:)", subtitle: "Guided generation → typed @Generable",
      systemImage: "checklist", running: vm.running == .guided,
      disabled: vm.running != nil
    ) {
      Task { await vm.runGuided() }
    } content: {
      if let colors = vm.guidedColors {
        VStack(alignment: .leading, spacing: 8) {
          HStack {
            ForEach(colors, id: \.self) { c in
              Text(c).font(.caption.bold())
                .padding(.horizontal, 10).padding(.vertical, 6)
                .background(swatch(for: c)).foregroundStyle(.white)
                .clipShape(Capsule())
            }
          }
          Label("Validated PrimaryColors.colors: [String] — decoded by FM",
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
      title: "Tool calling", subtitle: "Model → FM runs your Tool → answer",
      systemImage: "wrench.and.screwdriver", running: vm.running == .tool,
      disabled: vm.running != nil
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

  private func swatch(for color: String) -> Color {
    switch color.lowercased() {
    case let c where c.contains("red"): return .red
    case let c where c.contains("green"): return .green
    case let c where c.contains("blue"): return .blue
    default: return .gray
    }
  }
}

// MARK: - Reusable demo card

@available(iOS 27.0, macOS 27.0, *)
private struct DemoCard<Content: View>: View {
  let title: String
  let subtitle: String
  let systemImage: String
  let running: Bool
  let disabled: Bool
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
  @Published var textOut = ""
  @Published var guidedColors: [String]?
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
    running = .text; defer { running = nil }
    textOut = ""
    do {
      // A fresh session per demo so each Run is self-contained — no prior turn in
      // the transcript steering the result.
      let session = LanguageModelSession(model: model)
      let r = try await session.respond(to: "Explain on-device AI in one sentence.")
      textOut = r.content
    } catch {
      textOut = "[error] \(error.localizedDescription)"
    }
  }

  func runGuided() async {
    guard let model, running == nil else { return }
    running = .guided; defer { running = nil }
    guidedColors = nil; guidedError = nil
    // A fresh session (clean transcript — prior demo turns otherwise pollute the
    // structured output) plus one retry: schema-in-prompt JSON can occasionally
    // come back unparseable on a 2B model.
    for attempt in 1...2 {
      do {
        let session = LanguageModelSession(model: model)
        let r = try await session.respond(generating: PrimaryColors.self) {
          "List the three additive primary colors."
        }
        guidedColors = r.content.colors
        return
      } catch {
        if attempt == 2 { guidedError = error.localizedDescription }
      }
    }
  }

  func runTool() async {
    guard let model, running == nil else { return }
    running = .tool; defer { running = nil }
    toolTrace = []; toolAnswer = ""
    // A fresh session carrying the tool. The trace closure hops to the main
    // actor when FM invokes the tool, so the round-trip is visible on screen.
    let tool = TracedTemperatureTool { [weak self] city in
      Task { @MainActor in
        self?.toolTrace.append("🔧 get_temperature(city: \"\(city)\")  →  22°C and sunny")
      }
    }
    let toolSession = LanguageModelSession(model: model, tools: [tool])
    do {
      let r = try await toolSession.respond(to: "What is the temperature in Tokyo right now?")
      toolAnswer = r.content
    } catch {
      toolAnswer = "[error] \(error.localizedDescription)"
    }
  }
}

#endif
