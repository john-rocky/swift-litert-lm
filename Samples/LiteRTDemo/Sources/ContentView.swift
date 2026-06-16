// LiteRTDemo — conversational chat UI (ChatGPT / Claude style).
//
// A scrolling message list with user/assistant bubbles, inline image
// attachments, a bottom input bar, and live token streaming into the assistant
// bubble. Multi-turn over a single LiteRTChat conversation (Gemma 4 E2B,
// text + image + audio, on the Metal GPU).

import SwiftUI
import PhotosUI
import LiteRTFoundation

// MARK: - Model

struct ChatMessage: Identifiable {
  enum Role { case user, assistant }
  let id = UUID()
  let role: Role
  var text: String
  var image: Data? = nil
  var stats: String? = nil
}

// MARK: - Root

struct ContentView: View {
  @StateObject private var vm = ChatViewModel()
  @State private var photoItem: PhotosPickerItem?
  @State private var input = "What can you do?"

  var body: some View {
    VStack(spacing: 0) {
      header
      Divider()
      messageList
      inputBar
    }
    .task { await vm.loadIfNeeded() }
    .onChange(of: photoItem) { item in Task { await vm.attachPhoto(item) } }
  }

  private var header: some View {
    HStack(spacing: 8) {
      Image(systemName: "sparkles").foregroundStyle(.tint)
      Text("Gemma 4 E2B").font(.headline)
      Spacer()
      switch vm.phase {
      case .loading(let f):
        HStack(spacing: 6) {
          ProgressView().controlSize(.small)
          Text("\(Int(f * 100))%").font(.caption).foregroundStyle(.secondary)
        }
      case .ready:
        Circle().fill(.green).frame(width: 9, height: 9)
      case .error:
        Circle().fill(.red).frame(width: 9, height: 9)
      case .idle:
        EmptyView()
      }
    }
    .padding(.horizontal).padding(.vertical, 10)
  }

  private var messageList: some View {
    ScrollViewReader { proxy in
      ScrollView {
        LazyVStack(spacing: 12) {
          if case .error(let message) = vm.phase {
            Text(message).font(.callout).foregroundStyle(.red).padding()
          }
          ForEach(vm.messages) { MessageBubble(message: $0) }
          if vm.isGenerating, vm.messages.last?.role != .assistant {
            HStack { ProgressView().controlSize(.small); Spacer() }.padding(.horizontal)
          }
          Color.clear.frame(height: 1).id(bottomID)
        }
        .padding(.vertical, 12)
      }
      .scrollDismissesKeyboard(.interactively)
      .onChange(of: vm.scrollTick) { _ in
        withAnimation(.easeOut(duration: 0.15)) { proxy.scrollTo(bottomID, anchor: .bottom) }
      }
    }
  }

  private let bottomID = "bottom"

  private var inputBar: some View {
    VStack(spacing: 6) {
      if let image = vm.attachedImage, let ui = uiImage(image) {
        HStack {
          Image(uiImage: ui).resizable().scaledToFill()
            .frame(width: 44, height: 44).clipShape(RoundedRectangle(cornerRadius: 8))
          Text("Image attached").font(.caption).foregroundStyle(.secondary)
          Spacer()
          Button { vm.attachedImage = nil; photoItem = nil } label: {
            Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
          }
        }
        .padding(.horizontal)
      }
      HStack(spacing: 10) {
        PhotosPicker(selection: $photoItem, matching: .images) {
          Image(systemName: "photo.on.rectangle").font(.title3)
        }
        .disabled(!vm.isReady)

        TextField("Message", text: $input, axis: .vertical)
          .textFieldStyle(.plain)
          .lineLimit(1...5)
          .padding(.horizontal, 12).padding(.vertical, 8)
          .background(Color(.secondarySystemBackground))
          .clipShape(RoundedRectangle(cornerRadius: 18))

        Button {
          let text = input
          input = ""
          Task { await vm.send(text) }
        } label: {
          Image(systemName: "arrow.up.circle.fill").font(.title)
        }
        .disabled(!vm.isReady || vm.isGenerating || input.trimmingCharacters(in: .whitespaces).isEmpty)
      }
      .padding(.horizontal).padding(.bottom, 8).padding(.top, 4)
    }
  }

  private func uiImage(_ data: Data) -> UIImage? { UIImage(data: data) }
}

// MARK: - Bubble

private struct MessageBubble: View {
  let message: ChatMessage

  var body: some View {
    HStack {
      if message.role == .user { Spacer(minLength: 40) }
      VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 6) {
        if let image = message.image, let ui = UIImage(data: image) {
          Image(uiImage: ui).resizable().scaledToFill()
            .frame(maxWidth: 220, maxHeight: 220).clipShape(RoundedRectangle(cornerRadius: 12))
        }
        if !message.text.isEmpty {
          Text(message.text)
            .padding(.horizontal, 12).padding(.vertical, 8)
            .background(message.role == .user ? Color.accentColor : Color(.secondarySystemBackground))
            .foregroundStyle(message.role == .user ? Color.white : Color.primary)
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .textSelection(.enabled)
        }
        if let stats = message.stats {
          Text(stats).font(.caption2.monospaced()).foregroundStyle(.secondary)
        }
      }
      if message.role == .assistant { Spacer(minLength: 40) }
    }
    .padding(.horizontal)
  }
}

// MARK: - View model

@MainActor
final class ChatViewModel: ObservableObject {
  enum Phase: Equatable { case idle, loading(Double), ready, error(String) }

  @Published var phase: Phase = .idle
  @Published var messages: [ChatMessage] = []
  @Published var attachedImage: Data?
  @Published var isGenerating = false
  @Published var scrollTick = 0  // bumped to trigger auto-scroll

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
      if ProcessInfo.processInfo.environment["LITERT_DEMO"] != nil { await runDemo() }
    } catch {
      phase = .error(error.localizedDescription)
    }
  }

  func attachPhoto(_ item: PhotosPickerItem?) async {
    guard let item else { return }
    if let data = try? await item.loadTransferable(type: Data.self) { attachedImage = data }
  }

  func send(_ text: String) async {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard let chat, !isGenerating, !trimmed.isEmpty else { return }
    isGenerating = true
    defer { isGenerating = false }

    let image = attachedImage
    attachedImage = nil
    messages.append(ChatMessage(role: .user, text: trimmed, image: image))
    scrollTick += 1

    let assistantIndex = messages.count
    messages.append(ChatMessage(role: .assistant, text: ""))

    let start = Date()
    do {
      for try await delta in chat.stream(trimmed, image: image) {
        messages[assistantIndex].text += delta
        scrollTick += 1
      }
      if let b = try? chat.lastBenchmark() {
        messages[assistantIndex].stats = String(
          format: "%.0f tok/s", b.lastDecodeTokensPerSecond)
      } else {
        messages[assistantIndex].stats = String(format: "%.1fs", Date().timeIntervalSince(start))
      }
    } catch {
      messages[assistantIndex].text += "\n[error] \(error.localizedDescription)"
    }
    scrollTick += 1
  }

  /// Headless demo (LITERT_DEMO=1): one image turn so a screenshot shows a real chat.
  private func runDemo() async {
    if let url = Bundle.main.url(forResource: "apple", withExtension: "png"),
      let data = try? Data(contentsOf: url) {
      attachedImage = data
    }
    await send("What is in this photo? Answer in one short sentence.")
  }
}
