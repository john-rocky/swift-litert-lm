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
  @State private var showFM = false
  @State private var showModelPicker = false
  @FocusState private var inputFocused: Bool

  var body: some View {
    chatStack
  }

  // The Easy-mode chat, plus an optional "FM API" cover. Entering FM mode frees
  // the Easy engine (`releaseEngine`) and FM builds its own on demand, so only
  // one multi-GB model is ever resident; leaving it reloads the Easy chat.
  @ViewBuilder private var chatStack: some View {
    let base = VStack(spacing: 0) {
      header
      Divider()
      messageList
      inputBar
    }
    .task { await vm.loadIfNeeded() }
    .onChange(of: photoItem) { item in Task { await vm.attachPhoto(item) } }
    .onChange(of: videoItem) { item in Task { await vm.attachVideo(item) } }
    .sheet(isPresented: $showModelPicker) { ModelPickerView(vm: vm) }

    #if canImport(FoundationModels)
    if #available(iOS 27.0, macOS 27.0, *) {
      base.fullScreenCover(isPresented: $showFM, onDismiss: {
        Task {
          await LiteRTLanguageModel.releaseCachedEngines()  // free FM engine first
          await vm.loadIfNeeded()                           // then reload Easy chat
        }
      }) { FMModeView() }
    } else {
      base
    }
    #else
    base
    #endif
  }

  private var header: some View {
    HStack(spacing: 8) {
      Image(systemName: "sparkles").foregroundStyle(.tint)
      Button {
        showModelPicker = true
      } label: {
        HStack(spacing: 4) {
          Text(vm.currentModelName).font(.headline).lineLimit(1)
          Image(systemName: "chevron.down").font(.caption2.bold())
        }
        .foregroundStyle(.primary)
      }
      .disabled(vm.isGenerating)
      Spacer()
      fmButton
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

  // Opens the FM-API demo. Releases the Easy engine before presenting so FM
  // mode's own engine doesn't load alongside it.
  @ViewBuilder private var fmButton: some View {
    #if canImport(FoundationModels)
    if #available(iOS 27.0, macOS 27.0, *) {
      Button {
        vm.releaseEngine()
        showFM = true
      } label: {
        Label("FM API", systemImage: "cpu").font(.caption.bold())
      }
      .buttonStyle(.bordered).controlSize(.small)
      .disabled(!vm.isReady || vm.isGenerating)
    }
    #endif
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

// MARK: - Model picker

/// Choose the model: the bundled, device-verified Gemma 4 E2B, any Hugging Face
/// `.litertlm` repo (downloaded on first use), or a local `.litertlm` file.
private struct ModelPickerView: View {
  @ObservedObject var vm: ChatViewModel
  @Environment(\.dismiss) private var dismiss

  @State private var repo = "litert-community/gemma-4-E4B-it-litert-lm"
  @State private var file = "gemma-4-E4B-it.litertlm"
  @State private var multimodal = false
  @State private var showImporter = false
  @State private var docsModels: [URL] = []

  /// `.litertlm` files sitting in the app's Documents (pushed via `devicectl
  /// device copy to … --destination Documents/X.litertlm`). Listed so they load
  /// with one tap — no Files navigation, no re-download.
  private func scanDocsModels() {
    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
    docsModels = ((try? FileManager.default.contentsOfDirectory(
      at: docs ?? URL(fileURLWithPath: "/"), includingPropertiesForKeys: nil)) ?? [])
      .filter { $0.pathExtension == "litertlm" }
      .sorted { $0.lastPathComponent < $1.lastPathComponent }
  }

  var body: some View {
    NavigationStack {
      List {
        Section("Bundled (device-verified)") {
          Button { pick(.bundledE2B) } label: {
            row("Gemma 4 E2B", "text · image · audio · ~2.6 GB", selected: vm.source == .bundledE2B)
          }
        }

        if !docsModels.isEmpty {
          Section("On this device (Documents)") {
            ForEach(docsModels, id: \.self) { url in
              Button { pick(.localFile(url, multimodal: multimodal)) } label: {
                row(url.deletingPathExtension().lastPathComponent,
                    "local · pushed to Documents", selected: false)
              }
            }
          }
        }

        Section("Options for downloaded / local models") {
          Toggle("Bring up image / audio towers", isOn: $multimodal)
          Text("Off = text-only (safe default for an unknown model). Turn on only "
            + "if the model ships vision/audio encoders, or loading may fail.")
            .font(.caption).foregroundStyle(.secondary)
        }

        Section("Hugging Face — any .litertlm repo") {
          Button {
            pick(.huggingFace(
              repo: "litert-community/gemma-4-E4B-it-litert-lm",
              file: "gemma-4-E4B-it.litertlm", multimodal: multimodal))
          } label: {
            row("Gemma 4 E4B", "text · ~3.7 GB · downloads on first use", selected: false)
          }
          Button {
            pick(.huggingFace(
              repo: "mlboydaisuke/Falcon3-3B-Instruct-LiteRT",
              file: "model.litertlm", multimodal: false))
          } label: {
            row("Falcon3-3B Instruct", "text · int4 · ~1.9 GB · downloads on first use", selected: false)
          }
          Button {
            pick(.huggingFace(
              repo: "mlboydaisuke/Llama-3.2-3B-Instruct-LiteRT",
              file: "model.litertlm", multimodal: false))
          } label: {
            row("Llama-3.2-3B Instruct", "text · int4 · ~2.1 GB · downloads on first use", selected: false)
          }
          Button {
            pick(.huggingFace(
              repo: "mlboydaisuke/Ministral-3-3B-Instruct-2512-LiteRT",
              file: "model.litertlm", multimodal: false))
          } label: {
            row("Ministral-3-3B Instruct", "text · int4 · ~2.3 GB · downloads on first use", selected: false)
          }
          Button {
            pick(.huggingFace(
              repo: "mlboydaisuke/SmolLM3-3B-LiteRT",
              file: "model.litertlm", multimodal: false))
          } label: {
            row("SmolLM3-3B", "text · int4 · ~1.9 GB · downloads on first use", selected: false)
          }
          Button {
            pick(.huggingFace(
              repo: "mlboydaisuke/OLMo-2-1B-Instruct-LiteRT",
              file: "model.litertlm", multimodal: false))
          } label: {
            row("OLMo-2-1B Instruct", "text · int4 · ~0.9 GB · downloads on first use", selected: false)
          }
          Button {
            pick(.huggingFace(
              repo: "mlboydaisuke/DeepSeek-R1-Distill-Qwen-1.5B-LiteRT",
              file: "model.litertlm", multimodal: false))
          } label: {
            row("DeepSeek-R1-Distill-Qwen-1.5B", "reasoning · int4 · ~1.0 GB · downloads on first use", selected: false)
          }
          TextField("owner/repo", text: $repo)
            .textInputAutocapitalization(.never).autocorrectionDisabled().font(.callout)
          TextField("file.litertlm", text: $file)
            .textInputAutocapitalization(.never).autocorrectionDisabled().font(.callout)
          Button("Download & load") {
            pick(.huggingFace(repo: repo, file: file, multimodal: multimodal))
          }
          .disabled(repo.isEmpty || !file.hasSuffix(".litertlm"))
        }

        Section("Local file") {
          Button("Choose a .litertlm…") { showImporter = true }
        }

        Section {
          Text("Gemma 4 E2B and the OLMo-2-1B / SmolLM3-3B / Ministral-3-3B / Llama-3.2-3B presets "
            + "are verified on device (iPhone 17 Pro). Other models run through the same engine — "
            + "bring any LiteRT-LM `.litertlm`.")
            .font(.caption).foregroundStyle(.secondary)
        }
      }
      .navigationTitle("Model")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar { ToolbarItem(placement: .topBarTrailing) { Button("Cancel") { dismiss() } } }
      .fileImporter(isPresented: $showImporter, allowedContentTypes: [.data]) { result in
        if case .success(let url) = result { pick(.localFile(url, multimodal: multimodal)) }
      }
      .onAppear { scanDocsModels() }
    }
  }

  private func pick(_ source: ChatViewModel.ModelSource) {
    dismiss()
    Task { await vm.switchModel(to: source) }
  }

  private func row(_ title: String, _ subtitle: String, selected: Bool) -> some View {
    HStack {
      VStack(alignment: .leading, spacing: 2) {
        Text(title).font(.body.bold()).foregroundStyle(.primary)
        Text(subtitle).font(.caption).foregroundStyle(.secondary)
      }
      Spacer()
      if selected { Image(systemName: "checkmark").foregroundStyle(.tint) }
    }
  }
}

// MARK: - View model

@MainActor
final class ChatViewModel: ObservableObject {
  enum Phase: Equatable { case idle, loading(Double), ready, error(String) }

  /// Where the model comes from. The bundled E2B is the verified default; the
  /// other two let you run *any* LiteRT-LM model — downloaded from a Hugging Face
  /// repo, or loaded from a local `.litertlm`.
  enum ModelSource: Equatable {
    case bundledE2B
    case huggingFace(repo: String, file: String, multimodal: Bool)
    case localFile(URL, multimodal: Bool)

    var displayName: String {
      switch self {
      case .bundledE2B: return "Gemma 4 E2B"
      case .huggingFace(_, let file, _): return file.replacingOccurrences(of: ".litertlm", with: "")
      case .localFile(let url, _): return url.deletingPathExtension().lastPathComponent
      }
    }
  }

  @Published var phase: Phase = .idle
  @Published var source: ModelSource = .bundledE2B
  @Published var messages: [ChatMessage] = []
  @Published var attachedImage: Data?
  @Published var attachedVideoFrames: [Data]?
  @Published var attachedVideoThumb: Data?
  @Published var attachedAudioURL: URL?
  @Published var isGenerating = false
  @Published var isRecording = false
  @Published var scrollTick = 0

  var currentModelName: String { source.displayName }

  private var chat: LiteRTChat?
  private var securityScopedURL: URL?
  private let recorder = AudioRecorder()

  var isReady: Bool { if case .ready = phase { return true } else { return false } }

  func canSend(text: String) -> Bool {
    guard isReady, !isGenerating, !isRecording else { return false }
    let hasAttachment = attachedImage != nil || attachedVideoFrames != nil || attachedAudioURL != nil
    return !text.trimmingCharacters(in: .whitespaces).isEmpty || hasAttachment
  }

  func loadIfNeeded() async {
    guard chat == nil, case .idle = phase else { return }
    // LITERT_OPEN_FILE=<name substring> opens straight into a local Documents model
    // (e.g. LITERT_OPEN_FILE=llama → launch directly into Llama chat, no download).
    if let want = ProcessInfo.processInfo.environment["LITERT_OPEN_FILE"],
       let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first,
       let match = ((try? FileManager.default.contentsOfDirectory(
         at: docs, includingPropertiesForKeys: nil)) ?? [])
         .first(where: { $0.pathExtension == "litertlm"
           && $0.lastPathComponent.range(of: want, options: .caseInsensitive) != nil }) {
      source = .localFile(match, multimodal: false)
    }
    await load(source)
  }

  /// Tear down the current model and load a different source (bundled / Hugging
  /// Face / local file). Chat history is kept.
  func switchModel(to newSource: ModelSource) async {
    guard !isGenerating else { return }
    releaseEngine()
    source = newSource
    await load(newSource)
  }

  private func load(_ src: ModelSource) async {
    guard chat == nil else { return }
    phase = .loading(0)
    let onProgress: @Sendable (ModelDownloader.Progress) -> Void = { [weak self] p in
      Task { @MainActor in
        if let self, case .loading = self.phase { self.phase = .loading(p.fraction) }
      }
    }
    do {
      let loaded: LiteRTChat
      switch src {
      case .bundledE2B:
        loaded = try await LiteRTChat(
          .gemma4_E2B, modalities: .all, enableBenchmark: true, prewarm: true,
          onDownloadProgress: onProgress)
      case .huggingFace(let repo, let file, let multimodal):
        // Match the conservative config the on-device self-test uses for converted
        // models: text-only, a moderate token budget, and no prewarm — a second
        // warmup conversation trips the runtime's KV-cache buffer copy in
        // CreateNewContext on some converted .litertlm (tensor_buffer error).
        loaded = try await LiteRTChat(
          huggingFaceRepo: repo, fileName: file, modalities: multimodal ? .all : [],
          maxTokens: 512, enableBenchmark: true, prewarm: false, onDownloadProgress: onProgress)
      case .localFile(let url, let multimodal):
        // Hold the security scope open while the engine has the file mapped.
        _ = url.startAccessingSecurityScopedResource()
        securityScopedURL = url
        loaded = try await LiteRTChat(
          modelFileURL: url, modalities: multimodal ? .all : [],
          maxTokens: 512, enableBenchmark: true, prewarm: false)
      }
      self.chat = loaded
      phase = .ready
      if ProcessInfo.processInfo.environment["LITERT_DEMO"] != nil { await runDemo() }
    } catch {
      phase = .error(error.localizedDescription)
    }
  }

  /// Drop the Easy-mode engine (freeing its weights) and reset to `.idle` so a
  /// later `loadIfNeeded()` rebuilds it. Used when handing memory to FM mode or
  /// before switching models. Chat history (`messages`) is kept.
  func releaseEngine() {
    chat = nil
    phase = .idle
    if let url = securityScopedURL {
      url.stopAccessingSecurityScopedResource()
      securityScopedURL = nil
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
