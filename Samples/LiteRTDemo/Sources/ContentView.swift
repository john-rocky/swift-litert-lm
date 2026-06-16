// LiteRTDemo — interactive chat UI.
//
// Shows the whole Easy-mode flow in ~one screen: first-run model download with
// progress, a prompt field, an optional photo and an optional sample audio clip,
// streaming output, and a live tokens/sec + memory readout.

import SwiftUI
import PhotosUI
import LiteRTFoundation

struct ContentView: View {
  @StateObject private var vm = ChatViewModel()
  @State private var photoItem: PhotosPickerItem?

  var body: some View {
    let hasImage = vm.imageData != nil
    return VStack(alignment: .leading, spacing: 12) {
      header

      switch vm.phase {
      case .loading(let fraction):
        VStack(alignment: .leading, spacing: 6) {
          ProgressView(value: fraction)
          Text("Downloading & loading Gemma 4 E2B… \(Int(fraction * 100))%")
            .font(.caption).foregroundStyle(.secondary)
        }
      case .error(let message):
        Text(message).font(.callout).foregroundStyle(.red)
      case .idle, .ready:
        EmptyView()
      }

      // Attachments
      HStack(spacing: 12) {
        PhotosPicker(selection: $photoItem, matching: .images) {
          Label(hasImage ? "Photo ✓" : "Photo", systemImage: "photo")
        }
        Toggle(isOn: $vm.attachSampleAudio) {
          Label("Sample audio", systemImage: "waveform")
        }
        .toggleStyle(.button)
        if hasImage {
          Button(role: .destructive) { vm.imageData = nil; photoItem = nil } label: {
            Image(systemName: "xmark.circle")
          }
        }
      }
      .font(.callout)
      .disabled(!vm.isReady)

      TextField("Ask something…", text: $vm.prompt, axis: .vertical)
        .textFieldStyle(.roundedBorder)
        .lineLimit(1...3)

      Button(action: { Task { await vm.generate() } }) {
        Label(vm.isGenerating ? "Generating…" : "Generate", systemImage: "sparkles")
          .frame(maxWidth: .infinity)
      }
      .buttonStyle(.borderedProminent)
      .disabled(!vm.isReady || vm.isGenerating || vm.prompt.isEmpty)

      ScrollView {
        Text(vm.output.isEmpty ? " " : vm.output)
          .frame(maxWidth: .infinity, alignment: .leading)
          .textSelection(.enabled)
      }
      .frame(maxHeight: .infinity)

      if !vm.stats.isEmpty {
        Text(vm.stats).font(.caption.monospaced()).foregroundStyle(.secondary)
      }
    }
    .padding()
    .task { await vm.loadIfNeeded() }
    .onChange(of: photoItem) { item in  // single-parameter form for iOS 16 compatibility
      Task { await vm.loadPhoto(item) }
    }
  }

  private var header: some View {
    HStack {
      Text("LiteRT · Gemma 4 E2B").font(.headline)
      Spacer()
      Circle()
        .fill(vm.isReady ? .green : .orange)
        .frame(width: 10, height: 10)
    }
  }
}

@MainActor
final class ChatViewModel: ObservableObject {
  enum Phase: Equatable {
    case idle, loading(Double), ready, error(String)
  }

  @Published var phase: Phase = .idle
  @Published var prompt: String = "Explain quantum computing in one sentence."
  @Published var output: String = ""
  @Published var stats: String = ""
  @Published var imageData: Data?
  @Published var attachSampleAudio = false
  @Published var isGenerating = false

  private var chat: LiteRTChat?

  var isReady: Bool { if case .ready = phase { return true } else { return false } }

  func loadIfNeeded() async {
    guard chat == nil, case .idle = phase else { return }
    phase = .loading(0)
    do {
      let chat = try await LiteRTChat(.gemma4_E2B, modalities: .all, enableBenchmark: true) {
        [weak self] progress in
        Task { @MainActor in
          if let self, case .loading = self.phase { self.phase = .loading(progress.fraction) }
        }
      }
      self.chat = chat
      phase = .ready
    } catch {
      phase = .error(error.localizedDescription)
    }
  }

  func loadPhoto(_ item: PhotosPickerItem?) async {
    guard let item else { return }
    if let data = try? await item.loadTransferable(type: Data.self) {
      imageData = data
    }
  }

  func generate() async {
    guard let chat, !isGenerating else { return }
    isGenerating = true
    output = ""
    stats = ""
    defer { isGenerating = false }

    let audio: AudioInput? = {
      guard attachSampleAudio,
        let url = Bundle.main.url(forResource: "have_a_wonderful_day", withExtension: "wav")
      else { return nil }
      return .file(url)
    }()

    let start = Date()
    do {
      for try await delta in chat.stream(prompt, image: imageData, audio: audio) {
        output += delta
      }
      let wall = Date().timeIntervalSince(start)
      if let b = try? chat.lastBenchmark() {
        stats = String(
          format: "%.1f tok/s decode · %.1f prefill · ttft %.2fs · %.0f MB",
          b.lastDecodeTokensPerSecond, b.lastPrefillTokensPerSecond,
          b.timeToFirstTokenInSecond, Double(LiteRTChat.memoryFootprintBytes()) / 1_048_576)
      } else {
        stats = String(format: "%.1fs · %.0f MB", wall,
          Double(LiteRTChat.memoryFootprintBytes()) / 1_048_576)
      }
    } catch {
      output += "\n\n[error] \(error.localizedDescription)"
    }
  }
}
