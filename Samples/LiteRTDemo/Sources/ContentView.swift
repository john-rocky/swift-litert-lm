// LiteRTDemo — conversational chat UI (ChatGPT / Claude style).
//
// A scrolling message list with user/assistant bubbles, attachments (photo,
// microphone audio, and video — sampled to frames), a bottom input bar, and
// live token streaming. Multi-turn over one LiteRTChat conversation (Gemma 4
// E2B, text + image + audio, Metal GPU).

import SwiftUI
import PhotosUI
import CoreTransferable
import UniformTypeIdentifiers
import LiteRTFoundation

// MARK: - Models

struct ChatMessage: Identifiable {
  enum Role { case user, assistant }
  let id = UUID()
  let role: Role
  var text: String
  var image: Data? = nil
  var videoThumb: Data? = nil
  var hasAudio: Bool = false
  var stats: String? = nil
}

/// A video pulled from the photo library as a temp-file URL (so we can sample
/// frames without loading the whole clip into memory).
struct Movie: Transferable {
  let url: URL
  static var transferRepresentation: some TransferRepresentation {
    FileRepresentation(contentType: .movie) { movie in SentTransferredFile(movie.url) } importing: {
      received in
      let copy = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString + ".mov")
      try? FileManager.default.removeItem(at: copy)
      try FileManager.default.copyItem(at: received.file, to: copy)
      return Movie(url: copy)
    }
  }
}

// MARK: - Root

struct ContentView: View {
  @StateObject private var vm = ChatViewModel()
  @State private var photoItem: PhotosPickerItem?
  @State private var videoItem: PhotosPickerItem?
  @State private var input = "What can you do?"
  @FocusState private var inputFocused: Bool

  var body: some View {
    VStack(spacing: 0) {
      header
      Divider()
      messageList
      inputBar
    }
    .task { await vm.loadIfNeeded() }
    .onChange(of: photoItem) { item in Task { await vm.attachPhoto(item) } }
    .onChange(of: videoItem) { item in Task { await vm.attachVideo(item) } }
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
      case .ready: Circle().fill(.green).frame(width: 9, height: 9)
      case .error: Circle().fill(.red).frame(width: 9, height: 9)
      case .idle: EmptyView()
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
      attachmentChips
      HStack(spacing: 12) {
        PhotosPicker(selection: $photoItem, matching: .images) {
          Image(systemName: "photo.on.rectangle")
        }
        PhotosPicker(selection: $videoItem, matching: .videos) {
          Image(systemName: "video")
        }
        Button { Task { await vm.toggleRecording() } } label: {
          Image(systemName: vm.isRecording ? "stop.circle.fill" : "mic")
            .foregroundStyle(vm.isRecording ? Color.red : Color.accentColor)
        }
        TextField("Message", text: $input, axis: .vertical)
          .textFieldStyle(.plain).lineLimit(1...5)
          .focused($inputFocused)
          .padding(.horizontal, 12).padding(.vertical, 8)
          .background(Color(.secondarySystemBackground))
          .clipShape(RoundedRectangle(cornerRadius: 18))
        Button {
          let text = input
          input = ""
          inputFocused = false  // dismiss the keyboard when generation starts
          Task { await vm.send(text) }
        } label: {
          Image(systemName: "arrow.up.circle.fill").font(.title)
        }
        .disabled(!vm.canSend(text: input))
      }
      .font(.title3)
      .disabled(!vm.isReady)
      .padding(.horizontal).padding(.bottom, 8).padding(.top, 4)
    }
  }

  @ViewBuilder private var attachmentChips: some View {
    if vm.attachedImage != nil || vm.attachedVideoThumb != nil || vm.attachedAudioURL != nil {
      HStack(spacing: 8) {
        if let data = vm.attachedImage, let ui = UIImage(data: data) {
          chip(thumb: ui, label: "Photo") { vm.attachedImage = nil; photoItem = nil }
        }
        if let data = vm.attachedVideoThumb, let ui = UIImage(data: data) {
          chip(thumb: ui, label: "Video", system: "video.fill") {
            vm.clearVideo(); videoItem = nil
          }
        }
        if vm.attachedAudioURL != nil {
          chip(thumb: nil, label: "Audio", system: "waveform") { vm.attachedAudioURL = nil }
        }
        Spacer()
      }
      .padding(.horizontal)
    }
  }

  private func chip(thumb: UIImage?, label: String, system: String = "photo", remove: @escaping () -> Void)
    -> some View
  {
    HStack(spacing: 6) {
      if let thumb {
        Image(uiImage: thumb).resizable().scaledToFill()
          .frame(width: 28, height: 28).clipShape(RoundedRectangle(cornerRadius: 6))
      } else {
        Image(systemName: system)
      }
      Text(label).font(.caption)
      Button(action: remove) { Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary) }
    }
    .padding(.horizontal, 8).padding(.vertical, 4)
    .background(Color(.secondarySystemBackground)).clipShape(Capsule())
  }
}

// MARK: - Bubble

private struct MessageBubble: View {
  let message: ChatMessage

  var body: some View {
    HStack {
      if message.role == .user { Spacer(minLength: 40) }
      VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 6) {
        if let data = message.image ?? message.videoThumb, let ui = UIImage(data: data) {
          ZStack(alignment: .bottomLeading) {
            Image(uiImage: ui).resizable().scaledToFill()
              .frame(maxWidth: 220, maxHeight: 220).clipShape(RoundedRectangle(cornerRadius: 12))
            if message.videoThumb != nil {
              Image(systemName: "play.circle.fill").foregroundStyle(.white).padding(6)
            }
          }
        }
        if message.hasAudio {
          Label("Audio", systemImage: "waveform").font(.caption).foregroundStyle(.secondary)
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
  @Published var attachedVideoFrames: [Data]?
  @Published var attachedVideoThumb: Data?
  @Published var attachedAudioURL: URL?
  @Published var isGenerating = false
  @Published var isRecording = false
  @Published var scrollTick = 0

  private var chat: LiteRTChat?
  private let recorder = AudioRecorder()

  var isReady: Bool { if case .ready = phase { return true } else { return false } }

  func canSend(text: String) -> Bool {
    guard isReady, !isGenerating, !isRecording else { return false }
    let hasAttachment = attachedImage != nil || attachedVideoFrames != nil || attachedAudioURL != nil
    return !text.trimmingCharacters(in: .whitespaces).isEmpty || hasAttachment
  }

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

  // MARK: Attachments

  func attachPhoto(_ item: PhotosPickerItem?) async {
    guard let item else { return }
    if let data = try? await item.loadTransferable(type: Data.self) { attachedImage = data }
  }

  func attachVideo(_ item: PhotosPickerItem?) async {
    guard let item else { return }
    guard let movie = try? await item.loadTransferable(type: Movie.self) else { return }
    if let frames = try? await VideoFrameSampler.sampleFrames(from: movie.url, count: 4),
      !frames.isEmpty {
      attachedVideoFrames = frames
      attachedVideoThumb = frames.first
    }
    try? FileManager.default.removeItem(at: movie.url)
  }

  func clearVideo() { attachedVideoFrames = nil; attachedVideoThumb = nil }

  func toggleRecording() async {
    if isRecording {
      attachedAudioURL = recorder.stop()
      isRecording = false
    } else {
      guard await recorder.requestPermission() else { return }
      if recorder.start() { isRecording = true }
    }
  }

  // MARK: Generation

  func send(_ text: String) async {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard let chat, !isGenerating else { return }

    let image = attachedImage
    let frames = attachedVideoFrames
    let videoThumb = attachedVideoThumb
    let audioURL = attachedAudioURL
    if trimmed.isEmpty && image == nil && frames == nil && audioURL == nil { return }

    isGenerating = true
    defer { isGenerating = false }

    attachedImage = nil
    clearVideo()
    attachedAudioURL = nil

    let prompt = trimmed.isEmpty ? defaultPrompt(image: image, frames: frames, audio: audioURL) : trimmed
    messages.append(
      ChatMessage(
        role: .user, text: trimmed, image: image, videoThumb: videoThumb, hasAudio: audioURL != nil))
    scrollTick += 1

    let assistantIndex = messages.count
    messages.append(ChatMessage(role: .assistant, text: ""))

    let start = Date()
    let audio: AudioInput? = audioURL.map { .file($0) }
    do {
      for try await delta in chat.stream(prompt, image: image, images: frames ?? [], audio: audio) {
        messages[assistantIndex].text += delta
        scrollTick += 1
      }
      if let b = try? chat.lastBenchmark() {
        messages[assistantIndex].stats = String(format: "%.0f tok/s", b.lastDecodeTokensPerSecond)
      } else {
        messages[assistantIndex].stats = String(format: "%.1fs", Date().timeIntervalSince(start))
      }
    } catch {
      messages[assistantIndex].text += "\n[error] \(error.localizedDescription)"
    }
    scrollTick += 1
  }

  private func defaultPrompt(image: Data?, frames: [Data]?, audio: URL?) -> String {
    if audio != nil {
      return "Listen to this audio and respond to it. If it asks a question, answer it."
    }
    if frames != nil {
      return "These images are frames sampled from a video in chronological order. "
        + "Describe what is happening in the video."
    }
    if image != nil { return "What is in this photo?" }
    return ""
  }

  /// Auto-demo (LITERT_DEMO=1): a couple of turns so a screen recording / GIF
  /// shows a real multimodal chat without manual input.
  private func runDemo() async {
    await send("In one short sentence, what can you do?")
    try? await Task.sleep(nanoseconds: 600_000_000)
    if let url = Bundle.main.url(forResource: "apple", withExtension: "png"),
      let data = try? Data(contentsOf: url) {
      attachedImage = data
    }
    await send("What is in this photo? Answer in one short sentence.")
  }
}
