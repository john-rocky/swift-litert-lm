// LiteRTDemo — "Showcase": a one-take, auto-playing demo designed to be
// screen-recorded vertically for a social post. Two selectable takes:
//
//   • .offline — "Gemma runs your iPhone". Every scene works with the radios
//     off, so it is recorded in Airplane Mode and the status-bar icon is the
//     "no cloud" proof (the status bar is deliberately left visible).
//   • .story — "An evening run with Gemma 4". The same capabilities wrapped
//     in one runner's story, plus a route-scouting scene that calls
//     MKLocalSearch for real Apple Maps results. That scene needs the
//     network, and says so on screen — the honest claim everywhere is that
//     the MODEL is 100% on-device, like any real app pairing local inference
//     with online data.
//
// Every turn goes through Apple's *real* Foundation Models API
// (`LanguageModelSession`) with Google's Gemma 4 as the model via LiteRT-LM:
// live token streaming (with a tok/s badge), vision, tool calling into real
// iOS frameworks (UIDevice battery, a real UserNotifications banner that
// drops mid-recording, an ActivityKit Live Activity, AVSpeechSynthesizer —
// and MKLocalSearch in the story take), and guided generation rendered
// straight into an animated Swift Charts bar chart.
//
// Recording recipe:
//   1. Run once online so the Gemma 4 E2B model is downloaded; allow the
//      notification permission prompt (it appears while loading, off-camera).
//   2. Offline take: turn ON Airplane Mode. Story take: stay online.
//   3. Start an iOS screen recording, open the showcase, tap Start (a 3-2-1
//      countdown gives the recording a clean lead-in).
//   4. Both takes end by swiping Home: Gemma's timer ticks in the Dynamic
//      Island (an app's own Live Activity hides while it is foreground).
//   Optional: LITERT_SHOWCASE=1 boots straight into the offline take,
//   LITERT_SHOWCASE=story into the story take.
//   Optional: bundle a photo named `showcase.jpg` (Resources/) to replace the
//   default apple.png in the vision scene — for the story take, a shot of
//   running shoes or an evening sky reads best.
//
// Brand note: the Apple logo glyph (SF Symbol `apple.logo`) is deliberately
// NOT used anywhere on screen — Apple's SF Symbols license restricts it, and
// a partner-channel post shouldn't carry it. Apple is referenced by name.

#if canImport(FoundationModels)

import SwiftUI
import FoundationModels
import LiteRTFoundation
import AVFoundation
import UserNotifications
import ActivityKit
import Charts
import CoreLocation
import MapKit
import UIKit

// MARK: - Variants

enum ShowcaseVariant {
  case offline, story
}

/// Everything that differs between the two takes — scene copy, prompts, and
/// which scenes run. The view and view model stay identical.
@available(iOS 27.0, macOS 27.0, *)
struct ShowcaseScript {
  let heroKicker: String        // small line above "Google Gemma 4"
  let heroSubtitle: String
  let overlayTip: String
  let introPrompt: String
  let introBubble: String       // short on-screen version of the intro prompt
  let visionPrompt: String
  /// Bundle base names tried (jpg/jpeg/png) for the vision photo, in order.
  let visionImageNames: [String]
  let mapPrompt: String?        // nil → no map scene (offline take)
  /// Agent mode: ONE prompt, and the model chains every tool itself. When
  /// nil, the toolbox falls back to the scripted one-tool-per-turn prompts —
  /// the reliable cut if the chain proves flaky on a given model.
  let agenticToolboxPrompt: String?
  let toolboxPrompts: [String]
  let chartPrompt: String
  let chartWeekly: Bool         // true → weekly km plan, false → food ranking
  let chartUnit: String
  let outroTitle: String
  let outroBadges: [(icon: String, text: String)]

  static let offline = ShowcaseScript(
    heroKicker: "running",
    heroSubtitle: "on-device · no cloud · powered by LiteRT-LM",
    overlayTip: "Tip: turn on Airplane Mode first — the status bar becomes the proof.",
    introPrompt:
      "You are Gemma 4, Google's open model, running fully on-device on this "
      + "iPhone through Apple's Foundation Models API, powered by LiteRT-LM. "
      + "Introduce yourself to iOS developers in two short, punchy sentences.",
    introBubble: "Introduce yourself to iOS developers in two short sentences.",
    visionPrompt: "What do you see in this photo? One short, vivid sentence.",
    visionImageNames: ["apple"],
    mapPrompt: nil,
    agenticToolboxPrompt:
      "Get me ready for my evening run: check my battery, schedule a reminder "
      + "to head out, start my 20-minute run timer, and use the speak tool to "
      + "cheer me on out loud. Handle it all yourself — don't ask me anything.",
    toolboxPrompts: [
      "Check my battery — do I have enough charge left?",
      "Schedule a notification right now that says: Time for your evening run!",
      "Start my 20-minute run timer — label it Evening Run.",
      "Now cheer me on out loud!",
    ],
    chartPrompt:
      "Rate four iconic Japanese foods by how much a first-time visitor "
      + "to Tokyo must try them, score 0-100. Reply with ONLY JSON shaped exactly like "
      + "{\"foods\": [\"a\", \"b\", \"c\", \"d\"], \"scores\": [90, 80, 70, 60]}.",
    chartWeekly: false,
    chartUnit: "",
    outroTitle: "That was Gemma 4.",
    outroBadges: [
      ("airplane", "Runs in Airplane Mode"),
      ("lock.fill", "Nothing left this iPhone"),
      ("curlybraces", "Apple's Foundation Models API"),
      ("bolt.fill", "Google's Gemma 4, via LiteRT-LM"),
    ])

  static let story = ShowcaseScript(
    heroKicker: "an evening run with",
    heroSubtitle: "your on-device running coach · powered by LiteRT-LM",
    overlayTip: "Tip: turn on Airplane Mode first — the status bar becomes the proof.",
    introPrompt:
      "You are Gemma 4, Google's open model, running fully on-device on this "
      + "iPhone through Apple's Foundation Models API, powered by LiteRT-LM. "
      + "You are my running coach tonight. In two short, punchy sentences, "
      + "pump me up for my evening run.",
    introBubble: "Coach me tonight — pump me up for my evening run.",
    visionPrompt:
      "Look at this photo and tell me in one playful sentence whether it's "
      + "a good omen for tonight's run.",
    visionImageNames: ["showcase", "apple"],
    // Map scene disabled: MKLocalSearch throttles repeated queries, which is
    // too flaky for a one-take recording — and skipping it makes this take
    // fully offline too. Re-enable by restoring a prompt here.
    mapPrompt: nil,
    agenticToolboxPrompt:
      "Coach, get me out the door: check my battery, set a heads-up "
      + "notification, start my 20-minute run timer, and use the speak tool to "
      + "cheer me on out loud. Handle it all yourself — don't ask me anything.",
    toolboxPrompts: [
      "Do I have enough battery left for my run playlist?",
      "Schedule a notification right now that says: Time to head out for your run!",
      "Start my 20-minute run timer — label it Evening Run.",
      "Send me off with a cheer — out loud!",
    ],
    chartPrompt:
      "Plan my running week: kilometers for each day, Monday to Sunday. "
      + "I can handle about 25 km in total. Reply with ONLY JSON shaped exactly like "
      + "{\"days\": [\"Mon\", \"Tue\", \"Wed\", \"Thu\", \"Fri\", \"Sat\", \"Sun\"], "
      + "\"km\": [5, 0, 8, 0, 5, 10, 0]}.",
    chartWeekly: true,
    chartUnit: " km",
    outroTitle: "Have a great run.",
    outroBadges: [
      ("airplane", "Runs in Airplane Mode"),
      ("lock.fill", "Nothing left this iPhone"),
      ("curlybraces", "Apple's Foundation Models API"),
      ("bolt.fill", "Google's Gemma 4, via LiteRT-LM"),
    ])
}

// MARK: - Gemma's iOS toolbox (each tool wraps a real framework)

/// Reads the device's actual battery level — data the model cannot know,
/// proving a real Swift function ran.
@available(iOS 27.0, macOS 27.0, *)
struct BatteryTool: FoundationModels.Tool {
  let name = "check_battery"
  let description = "Read this iPhone's current battery level and charging state."
  let onCall: @Sendable (String) -> Void

  @Generable
  struct Arguments {}

  func call(arguments: Arguments) async throws -> String {
    let summary = await MainActor.run { Self.read() }
    onCall(summary)
    return summary
  }

  @MainActor static func read() -> String {
    let device = UIDevice.current
    device.isBatteryMonitoringEnabled = true
    let level = device.batteryLevel
    guard level >= 0 else { return "The battery level is unavailable on this device." }
    let state: String
    switch device.batteryState {
    case .charging: state = "charging"
    case .full: state = "fully charged"
    case .unplugged: state = "on battery"
    default: state = "in an unknown state"
    }
    return "The battery is at \(Int(level * 100))% and \(state)."
  }
}

/// Schedules a real iOS notification a few seconds out, so the system banner
/// drops onto the screen while the recording is still on this scene.
@available(iOS 27.0, macOS 27.0, *)
struct NotifyTool: FoundationModels.Tool {
  let name = "schedule_notification"
  let description = "Schedule a real iOS notification banner that pops up on this iPhone after a short delay."
  let onCall: @Sendable (String) -> Void

  @Generable
  struct Arguments {
    @Guide(description: "The notification message text, one short sentence")
    var message: String
  }

  func call(arguments: Arguments) async throws -> String {
    let content = UNMutableNotificationContent()
    content.title = "Gemma 4"
    content.body = arguments.message
    content.sound = .default
    try await UNUserNotificationCenter.current().add(
      UNNotificationRequest(
        identifier: UUID().uuidString, content: content,
        trigger: UNTimeIntervalNotificationTrigger(timeInterval: 4, repeats: false)))
    onCall(arguments.message)
    return "Notification scheduled — it will appear at the top of the screen in a few seconds."
  }
}

/// Speaks through the iPhone speaker via AVSpeechSynthesizer.
@available(iOS 27.0, macOS 27.0, *)
struct SpeakTool: FoundationModels.Tool {
  let name = "speak"
  let description = "Speak a short text out loud through the iPhone speaker. "
    + "This is the only way to actually say something out loud."
  let onCall: @Sendable (String) -> Void

  @Generable
  struct Arguments {
    @Guide(description: "The short, upbeat text to speak — one sentence")
    var text: String
  }

  func call(arguments: Arguments) async throws -> String {
    onCall(arguments.text)
    return "Spoken out loud."
  }
}

/// Starts a Live Activity countdown — the payoff appears in the Dynamic
/// Island once the user swipes Home (the island hides an app's own activity
/// while that app is in the foreground). Degrades gracefully when Live
/// Activities are disabled, so the rest of the showcase is unaffected.
@available(iOS 27.0, macOS 27.0, *)
struct TimerTool: FoundationModels.Tool {
  let name = "start_run_timer"
  let description = "Start a visible 20-minute run countdown the user can see from anywhere on the iPhone."
  let onCall: @Sendable (String) -> Void

  @Generable
  struct Arguments {
    @Guide(description: "A short motivational label for the timer, a few words")
    var label: String
  }

  func call(arguments: Arguments) async throws -> String {
    let result = await MainActor.run { Self.startActivity(label: arguments.label) }
    onCall(arguments.label)
    return result
  }

  @MainActor static func startActivity(label: String) -> String {
    guard ActivityAuthorizationInfo().areActivitiesEnabled else {
      return "Live Activities are disabled on this device, so no timer was started."
    }
    do {
      let state = GemmaRunAttributes.ContentState(
        message: label, endDate: Date().addingTimeInterval(20 * 60))
      _ = try Activity.request(
        attributes: GemmaRunAttributes(title: "Gemma 4"),
        content: .init(state: state, staleDate: nil))
      return "A 20-minute run timer is now live — it shows in the Dynamic Island."
    } catch {
      return "Could not start the Live Activity: \(error.localizedDescription)"
    }
  }
}

/// A real place found on Apple Maps, ready to pin.
struct MapSpot: Identifiable, Sendable {
  let id = UUID()
  let name: String
  let latitude: Double
  let longitude: Double
  var coordinate: CLLocationCoordinate2D {
    CLLocationCoordinate2D(latitude: latitude, longitude: longitude)
  }
}

/// Story take only: searches Apple Maps (MKLocalSearch) for real running
/// spots. This is the one tool that needs the network — the scene labels it.
@available(iOS 27.0, macOS 27.0, *)
struct MapSearchTool: FoundationModels.Tool {
  let name = "find_running_spots"
  let description = "Search Apple Maps for real parks or running spots near an area."
  let onCall: @Sendable ([MapSpot]) -> Void

  @Generable
  struct Arguments {
    @Guide(description: "The neighborhood or city to search near")
    var area: String
  }

  func call(arguments: Arguments) async throws -> String {
    // MKLocalSearch throttles repeated automated queries, so retry once
    // before giving up — and keep the failure answer SHORT: without the
    // explicit instruction the model falls back to a long list of
    // from-memory suggestions, which kills the scene's pacing.
    var lastError: Error?
    for attempt in 1...2 {
      let request = MKLocalSearch.Request()
      request.naturalLanguageQuery = "park for running near \(arguments.area)"
      request.resultTypes = .pointOfInterest
      do {
        let response = try await MKLocalSearch(request: request).start()
        let spots = response.mapItems.prefix(4).map {
          MapSpot(
            name: $0.name ?? "Unknown",
            latitude: $0.placemark.coordinate.latitude,
            longitude: $0.placemark.coordinate.longitude)
        }
        onCall(Array(spots))
        guard !spots.isEmpty else { return "No parks found near \(arguments.area)." }
        return "Real parks found on Apple Maps: "
          + spots.map(\.name).joined(separator: ", ")
          + ". Pick one for the user and say why in one sentence."
      } catch {
        lastError = error
        if attempt == 1 { try? await Task.sleep(nanoseconds: 1_500_000_000) }
      }
    }
    onCall([])
    return "Map search is unavailable right now "
      + "(\(lastError?.localizedDescription ?? "unknown error")). "
      + "Apologize in ONE short sentence and move on — do not list any suggestions."
  }
}

// MARK: - Generable types for the chart scene

/// Guided-generation payoff, offline take: a food ranking.
@available(iOS 27.0, macOS 27.0, *)
@Generable
struct FoodChart {
  @Guide(description: "Exactly four iconic Japanese foods, one or two words each")
  var foods: [String]
  @Guide(description: "A 0-100 must-try score for each food, same order as foods")
  var scores: [Int]
}

/// Guided-generation payoff, story take: a weekly running plan.
@available(iOS 27.0, macOS 27.0, *)
@Generable
struct WeekPlan {
  @Guide(description: "Exactly seven weekday labels, Monday to Sunday, three letters each")
  var days: [String]
  @Guide(description: "Kilometers to run each day, 0 to 15, same order as days")
  var km: [Int]
}

// MARK: - View

@available(iOS 27.0, macOS 27.0, *)
struct FMShowcaseView: View {
  /// LITERT_SHOWCASE=1 boots the app straight into the offline take,
  /// LITERT_SHOWCASE=story into the story take — handy when setting up a
  /// recording.
  static var isStandaloneLaunch: Bool {
    ProcessInfo.processInfo.environment["LITERT_SHOWCASE"] != nil
  }

  static var standaloneVariant: ShowcaseVariant {
    ProcessInfo.processInfo.environment["LITERT_SHOWCASE"] == "story" ? .story : .offline
  }

  let standalone: Bool
  @StateObject private var vm: FMShowcaseVM
  @Environment(\.dismiss) private var dismiss

  init(standalone: Bool = false, variant: ShowcaseVariant = .offline) {
    self.standalone = standalone
    // Standalone (LITERT_SHOWCASE) launches auto-start once the model is
    // warm — hands-free recording, and headless runs via devicectl.
    _vm = StateObject(wrappedValue: FMShowcaseVM(variant: variant, autoStart: standalone))
  }

  var body: some View {
    ZStack {
      background
      content
      if !vm.playing { startOverlay }
      if !standalone { closeButton }
    }
    .preferredColorScheme(.dark)
    .toolbar(.hidden, for: .navigationBar)
    .task { await vm.load() }
    .onDisappear { vm.teardown() }
  }

  private var background: some View {
    LinearGradient(
      colors: [Color(red: 0.06, green: 0.06, blue: 0.14), .black],
      startPoint: .top, endPoint: .bottom)
    .ignoresSafeArea()
  }

  private var titleGradient: LinearGradient {
    LinearGradient(colors: [.blue, .purple], startPoint: .leading, endPoint: .trailing)
  }

  // MARK: Layout

  private var content: some View {
    VStack(spacing: 0) {
      topBar
      sceneBody
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .padding(.horizontal, 20)
      footer
    }
  }

  private var topBar: some View {
    VStack(spacing: 10) {
      HStack(spacing: 8) {
        Text("Apple Foundation Models API").font(.footnote.weight(.semibold))
        Text("×").font(.footnote).foregroundStyle(.secondary)
        Text("Gemma 4").font(.footnote.weight(.bold)).foregroundStyle(titleGradient)
      }
      .foregroundStyle(.primary)
      // Progress dots for the live scenes.
      HStack(spacing: 6) {
        ForEach(0..<vm.liveScenes.count, id: \.self) { i in
          Capsule()
            .fill(i == vm.currentDot ? AnyShapeStyle(titleGradient) : AnyShapeStyle(Color(.systemGray5)))
            .frame(width: i == vm.currentDot ? 22 : 7, height: 7)
        }
      }
      .opacity(vm.currentDot == nil ? 0.25 : 1)
      .animation(.spring(duration: 0.4), value: vm.scene)
    }
    .padding(.top, 8)
    .padding(.bottom, 14)
  }

  private var footer: some View {
    Text("Gemma 4 E2B · powered by LiteRT-LM · 100% on-device")
      .font(.caption2.weight(.medium)).foregroundStyle(.secondary)
      .padding(.bottom, 10)
  }

  @ViewBuilder private var sceneBody: some View {
    Group {
      switch vm.scene {
      case .hero: heroScene
      case .stream: streamScene
      case .vision: visionScene
      case .map: mapScene
      case .toolbox: toolboxScene
      case .chart: chartScene
      case .outro: outroScene
      }
    }
    .id(vm.scene)
    .transition(.asymmetric(
      insertion: .move(edge: .trailing).combined(with: .opacity),
      removal: .opacity))
  }

  // MARK: Scenes

  private var heroScene: some View {
    VStack(spacing: 22) {
      Spacer()
      VStack(spacing: 10) {
        Text("Apple Foundation Models")
          .font(.system(size: 28, weight: .bold, design: .rounded))
          .lineLimit(1).minimumScaleFactor(0.7)
        Text(vm.script.heroKicker)
          .font(.headline).foregroundStyle(.secondary)
        Text("Google Gemma 4")
          .font(.system(size: 42, weight: .heavy, design: .rounded))
          .foregroundStyle(titleGradient)
        Text(vm.script.heroSubtitle)
          .font(.subheadline.weight(.medium)).foregroundStyle(.secondary)
      }
      codeCard(
        "// Apple's built-in model\n"
        + "LanguageModelSession(model: SystemLanguageModel.default)\n\n"
        + "// Google's Gemma 4 — swap one line\n"
        + "LanguageModelSession(model: gemma)")
      Spacer()
      Spacer()
    }
  }

  private var streamScene: some View {
    VStack(alignment: .leading, spacing: 16) {
      sceneHeader("SCENE 1 · LIVE STREAMING", "Say hi, Gemma.",
        "session.streamResponse — Apple's exact API")
      promptBubble(vm.script.introBubble)
      answerCard {
        streamingText(vm.streamText)
      }
      if let tps = vm.streamTokensPerSecond {
        tokBadge(tps)
      }
      Spacer()
    }
  }

  private var visionScene: some View {
    VStack(alignment: .leading, spacing: 16) {
      sceneHeader("SCENE 2 · VISION", "It can see.",
        "an image, right in the prompt")
      if let data = vm.visionImageData, let ui = UIImage(data: data) {
        HStack {
          Spacer(minLength: 40)
          Image(uiImage: ui).resizable().scaledToFill()
            .frame(maxWidth: 200, maxHeight: 160).clipShape(RoundedRectangle(cornerRadius: 14))
        }
      }
      promptBubble(vm.script.visionPrompt)
      answerCard {
        streamingText(vm.visionText)
      }
      if let tps = vm.visionTokensPerSecond {
        tokBadge(tps)
      }
      Spacer()
    }
  }

  private var mapScene: some View {
    VStack(alignment: .leading, spacing: 14) {
      sceneHeader("SCENE 3 · MAPS + TOOL CALLING", "It scouts your route.",
        "MKLocalSearch — real Apple Maps results")
      // The honest annotation: this is the one scene that goes online.
      Label("Live Apple Maps data — this scene uses the network",
        systemImage: "network")
        .font(.caption.weight(.semibold)).foregroundStyle(.orange)
      if let prompt = vm.script.mapPrompt {
        promptBubble(prompt)
      }
      if !vm.mapSpots.isEmpty {
        Map(initialPosition: .region(vm.mapRegion)) {
          ForEach(vm.mapSpots) { spot in
            Marker(spot.name, coordinate: spot.coordinate)
          }
        }
        .frame(height: 200)
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .id(vm.mapSpots.map(\.name).joined())
        .overlay(alignment: .bottomTrailing) {
          Text("Apple Maps").font(.caption2.bold())
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(.ultraThinMaterial).clipShape(Capsule()).padding(8)
        }
        .transition(.move(edge: .bottom).combined(with: .opacity))
      }
      if vm.mapAnswer.isEmpty {
        thinkingRow("Gemma is searching Apple Maps…")
      } else {
        answerCard { Text(vm.mapAnswer).font(.callout).lineSpacing(2) }
      }
      Spacer()
    }
  }

  private var toolboxScene: some View {
    ScrollViewReader { proxy in
      ScrollView(showsIndicators: false) {
        VStack(alignment: .leading, spacing: 14) {
          if vm.script.agenticToolboxPrompt != nil {
            sceneHeader("SCENE · AGENT MODE", "One ask. It does the rest.",
              "a single prompt — Gemma chains four real iOS tools itself")
          } else {
            sceneHeader("SCENE · TOOL CALLING ×4", "It runs your iPhone.",
              "real Swift tools — the model picks and calls them")
          }
          // Task board: the four jobs are visible from the start; each row
          // checks off with the actual call and its real result, so
          // task → result stays legible in a recording.
          Text("GEMMA'S TASKS")
            .font(.caption2.weight(.heavy)).foregroundStyle(.secondary).kerning(1.2)
          VStack(spacing: 8) {
            ForEach(FMShowcaseVM.toolTasks, id: \.name) { task in
              taskRow(task)
            }
          }
          ForEach(vm.toolboxTurns) { turn in
            VStack(alignment: .leading, spacing: 8) {
              promptBubble(turn.prompt)
              if turn.answer.isEmpty {
                thinkingRow(vm.script.agenticToolboxPrompt != nil
                  ? "Gemma is working — watch the tools light up…"
                  : "Gemma is choosing a tool…")
              } else {
                Text(turn.answer).font(.callout).lineSpacing(2)
              }
            }
          }
          if vm.speaking, let spoken = vm.spokenText {
            HStack(spacing: 10) {
              speechWave
              Text("“\(spoken)”").font(.callout.italic()).foregroundStyle(.secondary)
            }
            .padding(.top, 2)
          }
          Color.clear.frame(height: 1).id("toolboxBottom")
        }
      }
      .onChange(of: vm.toolboxTick) { _ in
        withAnimation(.easeOut(duration: 0.2)) {
          proxy.scrollTo("toolboxBottom", anchor: .bottom)
        }
      }
    }
  }

  private var chartScene: some View {
    VStack(alignment: .leading, spacing: 16) {
      sceneHeader("SCENE · GUIDED GENERATION", "Typed Swift → Charts.",
        "@Generable output straight into Swift Charts — no JSON")
      promptBubble(vm.script.chartPrompt)
      codeCard(
        "let a = try await session.respond(generating: "
        + (vm.script.chartWeekly ? "WeekPlan" : "FoodChart") + ".self)\n"
        + "Chart(a.content) { BarMark(…) }   // typed, no parsing")
      if !vm.chartData.isEmpty {
        answerCard {
          VStack(alignment: .leading, spacing: 12) {
            Chart(vm.chartData) { item in
              BarMark(
                x: .value("Value", Double(item.value) * vm.chartProgress),
                y: .value("Label", item.label))
              .foregroundStyle(titleGradient)
              .cornerRadius(5)
              .annotation(position: .trailing) {
                Text("\(item.value)\(vm.script.chartUnit)")
                  .font(.caption2.monospaced().bold())
                  .opacity(vm.chartProgress > 0.85 ? 1 : 0)
              }
            }
            .chartXScale(domain: 0...(vm.chartDomainMax))
            .chartXAxis(.hidden)
            .chartYAxis {
              AxisMarks(position: .leading) {
                AxisValueLabel().font(.footnote.weight(.semibold))
              }
            }
            .frame(height: vm.script.chartWeekly ? 200 : 170)
            Label("A typed Swift value, rendered natively. No JSON, no regex.",
              systemImage: "checkmark.seal.fill")
              .font(.caption.weight(.semibold)).foregroundStyle(.green)
          }
        }
      } else if let err = vm.chartError {
        answerCard { Text(err).font(.callout).foregroundStyle(.red) }
      } else {
        thinkingRow("Gemma is filling your struct…")
      }
      Spacer()
    }
  }

  private var outroScene: some View {
    VStack(spacing: 24) {
      Spacer()
      Text(vm.script.outroTitle)
        .font(.system(size: 36, weight: .heavy, design: .rounded))
        .foregroundStyle(titleGradient)
      VStack(alignment: .leading, spacing: 14) {
        ForEach(vm.script.outroBadges, id: \.text) { badge in
          outroBadge(badge.icon, badge.text)
        }
      }
      Text("github.com/google-ai-edge/LiteRT-LM")
        .font(.callout.monospaced().weight(.semibold))
        .padding(.horizontal, 16).padding(.vertical, 10)
        .background(Color.white.opacity(0.08)).clipShape(Capsule())
      if vm.timerStarted {
        Label("One more thing — swipe Home: Gemma's timer is in the Dynamic Island",
          systemImage: "timer")
          .font(.callout.weight(.semibold)).foregroundStyle(.orange)
          .multilineTextAlignment(.center)
      }
      Spacer()
      Button { vm.replay() } label: {
        Label("Replay", systemImage: "arrow.counterclockwise")
          .font(.footnote.weight(.semibold))
      }
      .buttonStyle(.bordered).controlSize(.small).opacity(0.6)
      Spacer().frame(height: 8)
    }
    .frame(maxWidth: .infinity)
  }

  // MARK: Overlay

  private var startOverlay: some View {
    ZStack {
      background
      VStack(spacing: 24) {
        Spacer()
        VStack(spacing: 10) {
          Text("Apple Foundation Models")
            .font(.system(size: 24, weight: .bold, design: .rounded))
            .lineLimit(1).minimumScaleFactor(0.7)
          Text(vm.script.heroKicker).font(.title3).foregroundStyle(.secondary)
          Text("Gemma 4")
            .font(.system(size: 44, weight: .heavy, design: .rounded))
            .foregroundStyle(titleGradient)
        }
        if let n = vm.countdown {
          Text("\(n)")
            .font(.system(size: 90, weight: .heavy, design: .rounded))
            .foregroundStyle(titleGradient)
            .contentTransition(.numericText(countsDown: true))
            .animation(.spring(duration: 0.3), value: vm.countdown)
        } else if vm.isReady {
          Button { vm.start() } label: {
            Label("Start the one-take demo", systemImage: "play.fill")
              .font(.headline).padding(.horizontal, 10).padding(.vertical, 4)
          }
          .buttonStyle(.borderedProminent).controlSize(.large).tint(.indigo)
          Text(vm.script.overlayTip)
            .font(.caption).foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
        } else {
          VStack(spacing: 10) {
            ProgressView()
            Text(vm.status).font(.callout).foregroundStyle(.secondary)
          }
        }
        Spacer()
        Text("Gemma 4 E2B · powered by LiteRT-LM · 100% on-device")
          .font(.caption2.weight(.medium)).foregroundStyle(.secondary)
          .padding(.bottom, 10)
      }
      .padding(.horizontal, 24)
    }
  }

  private var closeButton: some View {
    VStack {
      HStack {
        Spacer()
        Button { dismiss() } label: {
          Image(systemName: "xmark.circle.fill")
            .font(.title3).foregroundStyle(.secondary).opacity(0.5)
        }
        .padding(.trailing, 16)
      }
      Spacer()
    }
    .padding(.top, 8)
  }

  // MARK: Components

  private func sceneHeader(_ kicker: String, _ title: String, _ sub: String) -> some View {
    VStack(alignment: .leading, spacing: 4) {
      Text(kicker).font(.caption.weight(.heavy)).foregroundStyle(titleGradient).kerning(1.2)
      Text(title).font(.system(size: 30, weight: .heavy, design: .rounded))
      Text(sub).font(.subheadline).foregroundStyle(.secondary)
    }
  }

  private func promptBubble(_ text: String) -> some View {
    HStack {
      Spacer(minLength: 40)
      Text(text)
        .font(.callout.weight(.medium))
        .padding(.horizontal, 14).padding(.vertical, 9)
        .background(Color.accentColor).foregroundStyle(.white)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }
  }

  private func answerCard<C: View>(@ViewBuilder _ content: () -> C) -> some View {
    content()
      .padding(16)
      .frame(maxWidth: .infinity, alignment: .leading)
      .background(Color.white.opacity(0.06))
      .clipShape(RoundedRectangle(cornerRadius: 14))
  }

  @ViewBuilder private func streamingText(_ text: String) -> some View {
    if text.isEmpty {
      thinkingRow("Gemma is thinking…")
    } else {
      Text(text).font(.title3.weight(.medium)).lineSpacing(3)
    }
  }

  private func thinkingRow(_ label: String) -> some View {
    HStack(spacing: 10) {
      ProgressView().controlSize(.small)
      Text(label).font(.callout).foregroundStyle(.secondary)
    }
  }

  private func tokBadge(_ tps: Double) -> some View {
    Label(String(format: "%.0f tok/s · on this iPhone", tps), systemImage: "bolt.fill")
      .font(.footnote.monospaced().weight(.bold))
      .foregroundStyle(.yellow)
      .padding(.horizontal, 12).padding(.vertical, 6)
      .background(Color.yellow.opacity(0.12)).clipShape(Capsule())
  }

  private func codeCard(_ code: String) -> some View {
    Text(code)
      .font(.caption.monospaced())
      .padding(14)
      .frame(maxWidth: .infinity, alignment: .leading)
      .background(Color.white.opacity(0.06))
      .clipShape(RoundedRectangle(cornerRadius: 12))
  }

  /// One task-board row: pending (dim ○) → done (✓ + the actual call + what
  /// really happened).
  private func taskRow(_ task: FMShowcaseVM.ToolTask) -> some View {
    let call = vm.toolCallsShown[task.name]
    let result = vm.toolResults[task.name]
    return HStack(alignment: .top, spacing: 10) {
      Image(systemName: result != nil ? "checkmark.circle.fill" : "circle")
        .font(.body.weight(.semibold))
        .foregroundStyle(result != nil ? AnyShapeStyle(titleGradient) : AnyShapeStyle(Color(.systemGray3)))
      VStack(alignment: .leading, spacing: 3) {
        HStack(spacing: 6) {
          Image(systemName: task.icon).font(.caption2)
          Text(task.title).font(.footnote.weight(.bold))
          Text("· \(task.framework)").font(.caption2).foregroundStyle(.secondary)
        }
        if let call {
          Text(call).font(.caption2.monospaced().weight(.semibold))
            .foregroundStyle(titleGradient).lineLimit(1)
        }
        // The result is the scene's payoff — give it a highlight pill that
        // springs in the moment the task checks off.
        if let result {
          Text(result)
            .font(.footnote.weight(.semibold))
            .foregroundStyle(.orange)
            .padding(.horizontal, 10).padding(.vertical, 5)
            .background(Color.orange.opacity(0.14))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .transition(.scale(scale: 0.85).combined(with: .opacity))
        }
      }
      Spacer(minLength: 0)
    }
    .padding(10)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(Color.white.opacity(result != nil ? 0.08 : 0.04))
    .clipShape(RoundedRectangle(cornerRadius: 10))
    .opacity(result != nil ? 1 : 0.55)
  }

  /// A small animated waveform shown while AVSpeechSynthesizer is talking.
  private var speechWave: some View {
    TimelineView(.animation) { context in
      let t = context.date.timeIntervalSinceReferenceDate
      HStack(spacing: 3) {
        ForEach(0..<7, id: \.self) { i in
          Capsule()
            .fill(titleGradient)
            .frame(width: 3, height: 8 + 14 * abs(sin(t * 6 + Double(i) * 0.9)))
        }
      }
      .frame(height: 24)
    }
  }

  private func outroBadge(_ icon: String, _ text: String) -> some View {
    HStack(spacing: 12) {
      Image(systemName: icon).font(.body.weight(.semibold))
        .foregroundStyle(titleGradient).frame(width: 26)
      Text(text).font(.body.weight(.semibold))
    }
  }
}

// MARK: - View model

@available(iOS 27.0, macOS 27.0, *)
@MainActor
final class FMShowcaseVM: ObservableObject {
  enum Scene: Int, Equatable {
    case hero, stream, vision, map, toolbox, chart, outro
  }

  struct ToolboxTurn: Identifiable {
    let id = UUID()
    let prompt: String
    var answer = ""
  }

  struct BarItem: Identifiable {
    let id = UUID()
    let label: String
    let value: Int
  }

  let script: ShowcaseScript
  /// The scenes between hero and outro, in play order — drives the dots.
  let liveScenes: [Scene]

  @Published var scene: Scene = .hero
  @Published var isReady = false
  @Published var status = "Loading Gemma 4 E2B…"
  @Published var playing = false
  @Published var countdown: Int?

  @Published var streamText = ""
  @Published var streamTokensPerSecond: Double?
  @Published var visionImageData: Data?
  @Published var visionText = ""
  @Published var visionTokensPerSecond: Double?

  @Published var mapSpots: [MapSpot] = []
  @Published var mapAnswer = ""

  struct ToolTask {
    let name: String
    let icon: String
    let title: String
    let framework: String
  }

  /// The task board shown in the toolbox scene, in the order the agentic
  /// prompt asks for them.
  static let toolTasks: [ToolTask] = [
    ToolTask(name: "check_battery", icon: "battery.100",
      title: "Check battery", framework: "UIDevice"),
    ToolTask(name: "schedule_notification", icon: "bell.badge.fill",
      title: "Schedule reminder", framework: "UserNotifications"),
    ToolTask(name: "start_run_timer", icon: "timer",
      title: "Start 20-min timer", framework: "ActivityKit"),
    ToolTask(name: "speak", icon: "waveform",
      title: "Cheer out loud", framework: "AVSpeechSynthesizer"),
  ]

  @Published var toolboxTurns: [ToolboxTurn] = []
  @Published var litTools: Set<String> = []
  @Published var toolCallsShown: [String: String] = [:]
  @Published var toolResults: [String: String] = [:]
  @Published var notificationText: String?
  @Published var timerStarted = false
  @Published var speaking = false
  @Published var spokenText: String?
  @Published var toolboxTick = 0

  @Published var chartData: [BarItem] = []
  @Published var chartProgress: Double = 0
  @Published var chartError: String?

  private var model: LiteRTLanguageModel?
  private var runTask: Task<Void, Never>?
  private let synthesizer = AVSpeechSynthesizer()
  private let bannerDelegate = BannerDelegate()
  private var speechDelegate: SpeechDelegate?

  private let autoStart: Bool

  init(variant: ShowcaseVariant, autoStart: Bool = false) {
    let script: ShowcaseScript = variant == .story ? .story : .offline
    self.script = script
    self.autoStart = autoStart
    self.liveScenes = script.mapPrompt == nil
      ? [.stream, .vision, .toolbox, .chart]
      : [.stream, .vision, .map, .toolbox, .chart]
  }

  var currentDot: Int? { liveScenes.firstIndex(of: scene) }

  var mapRegion: MKCoordinateRegion {
    guard let minLat = mapSpots.map(\.latitude).min(),
      let maxLat = mapSpots.map(\.latitude).max(),
      let minLon = mapSpots.map(\.longitude).min(),
      let maxLon = mapSpots.map(\.longitude).max()
    else {
      return MKCoordinateRegion(
        center: CLLocationCoordinate2D(latitude: 35.66, longitude: 139.7),
        span: MKCoordinateSpan(latitudeDelta: 0.1, longitudeDelta: 0.1))
    }
    return MKCoordinateRegion(
      center: CLLocationCoordinate2D(
        latitude: (minLat + maxLat) / 2, longitude: (minLon + maxLon) / 2),
      span: MKCoordinateSpan(
        latitudeDelta: max(0.02, (maxLat - minLat) * 1.6),
        longitudeDelta: max(0.02, (maxLon - minLon) * 1.6)))
  }

  var chartDomainMax: Double {
    Double(chartData.map(\.value).max() ?? 100) * 1.15
  }

  func load() async {
    guard model == nil else { return }
    // No tok/s from the engine here (we time the FM stream ourselves), and a
    // disabled benchmark avoids the no-sampler prewarm tripping
    // `output_buffer_dup` if Easy mode left the global flag on.
    ExperimentalFlags.enableBenchmark = false
    // Ask for notification permission while loading, so the system prompt is
    // long gone before the recording starts.
    let center = UNUserNotificationCenter.current()
    center.delegate = bannerDelegate
    _ = try? await center.requestAuthorization(options: [.alert, .sound])
    do {
      let m = try await LiteRTLanguageModel(.gemma4_E2B)
      model = m
      // A hidden warmup turn so the first on-camera scene starts instantly.
      status = "Warming up the engine…"
      _ = try await LanguageModelSession(model: m).respond(to: "Say: ready")
      isReady = true
      status = "Ready"
      if autoStart { start() }
    } catch {
      status = "Load failed: \(error.localizedDescription)"
    }
  }

  func start() {
    guard isReady, runTask == nil else { return }
    runTask = Task { await run() }
  }

  func replay() {
    runTask?.cancel()
    runTask = nil
    streamText = ""; streamTokensPerSecond = nil
    visionText = ""; visionTokensPerSecond = nil; visionImageData = nil
    mapSpots = []; mapAnswer = ""
    toolboxTurns = []; litTools = []; toolCallsShown = [:]; toolResults = [:]
    notificationText = nil
    timerStarted = false
    speaking = false; spokenText = nil
    chartData = []; chartProgress = 0; chartError = nil
    UNUserNotificationCenter.current().removeAllPendingNotificationRequests()
    for activity in Activity<GemmaRunAttributes>.activities {
      Task { await activity.end(nil, dismissalPolicy: .immediate) }
    }
    playing = false
    scene = .hero
    start()
  }

  func cancel() {
    runTask?.cancel()
    runTask = nil
  }

  /// Leaving the showcase: cancel the run AND drop the cached engine. A
  /// generation cancelled mid-stream keeps the engine busy for a long time,
  /// and the next entry (either take) then hangs on its first scene.
  func teardown() {
    cancel()
    Task { await LiteRTLanguageModel.releaseCachedEngines() }
  }

  // MARK: Orchestration

  private func run() async {
    // 3-2-1 lead-in so a screen recording starts clean.
    for n in [3, 2, 1] {
      countdown = n
      await hold(1.0)
      if Task.isCancelled { return }
    }
    countdown = nil
    playing = true
    withAnimation { scene = .hero }
    await hold(3.4)

    for next in liveScenes {
      if Task.isCancelled { return }
      switch next {
      case .stream:
        await runStream()
        await hold(2.4)
      case .vision:
        await runVision()
        await hold(2.4)
      case .map:
        await runMap()
        await hold(3.0)
      case .toolbox:
        await runToolbox()
        await hold(2.4)
      case .chart:
        await runChart()
        await hold(3.4)
      default:
        break
      }
    }
    if Task.isCancelled { return }

    withAnimation { scene = .outro }
    runTask = nil
  }

  private func runStream() async {
    guard let model else { return }
    withAnimation { scene = .stream }
    // Retry once with a fresh engine, and put a watchdog on each attempt: a
    // generation left half-cancelled by an earlier exit keeps the engine busy
    // and the first scene then hangs on "thinking" with no error to catch.
    for attempt in 1...2 {
      let start = Date()
      do {
        try await withWatchdog(45) { [self] in
          let session = LanguageModelSession(model: model)
          // Persona goes in the prompt (not session instructions) so the model
          // introduces itself truthfully — it can't otherwise know where it runs.
          var chunks = 0
          var firstToken: Date?
          for try await snapshot in session.streamResponse(to: script.introPrompt) {
            if firstToken == nil { firstToken = Date() }
            chunks += 1
            streamText = snapshot.content
            if chunks >= 8, let firstToken {
              streamTokensPerSecond =
                Double(chunks - 1) / max(0.001, Date().timeIntervalSince(firstToken))
            }
          }
        }
        return
      } catch {
        streamText = ""
        await recoverEngine()
        if attempt == 2 { streamText = "[error] \(error.localizedDescription)" }
      }
    }
  }

  /// Cancel `work` if it doesn't finish within `seconds` — a wedged engine
  /// hangs silently, so a plain `catch` never fires without this.
  private func withWatchdog(_ seconds: Double, _ work: @escaping () async throws -> Void) async throws {
    let task = Task { try await work() }
    let watchdog = Task {
      try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
      task.cancel()
    }
    defer { watchdog.cancel() }
    try await task.value
  }

  private func runVision() async {
    guard let model else { return }
    withAnimation { scene = .vision }
    guard let image = Self.bundledVisionImage(names: script.visionImageNames) else {
      visionText = "Bundle a showcase.jpg to run the vision scene."
      return
    }
    withAnimation { visionImageData = image }
    // One retry with a fresh engine: a failed vision turn can leave a zombie
    // task in the engine's callback pool that wedges every later conversation
    // (observed as DEADLINE_EXCEEDED spam, then a dead run).
    for attempt in 1...2 {
      let start = Date()
      do {
        try await withWatchdog(60) { [self] in
          let session = LanguageModelSession(model: model)
          var chunks = 0
          var firstToken: Date?
          let stream = session.streamResponse {
            LiteRTVideoSegment(frames: [image])
            script.visionPrompt
          }
          for try await snapshot in stream {
            if firstToken == nil { firstToken = Date() }
            chunks += 1
            visionText = snapshot.content
            if chunks >= 8, let firstToken {
              visionTokensPerSecond =
                Double(chunks - 1) / max(0.001, Date().timeIntervalSince(firstToken))
            }
          }
        }
        return
      } catch {
        visionText = ""
        await recoverEngine()
        if attempt == 2 { visionText = "[error] \(error.localizedDescription)" }
      }
    }
  }

  /// The line to voice when the model narrates the cheer instead of calling
  /// speak: the first quoted span of the answer if present, else the answer.
  private static func cheerLine(from answer: String) -> String {
    if let open = answer.firstIndex(of: "\""),
      let close = answer[answer.index(after: open)...].firstIndex(of: "\"") {
      let inner = String(answer[answer.index(after: open)..<close])
      if !inner.isEmpty { return inner }
    }
    return String(answer.prefix(140))
  }

  /// Drop the (possibly wedged) engine so the next conversation rebuilds it
  /// from scratch. Costs a reload, saves the take.
  private func recoverEngine() async {
    await LiteRTLanguageModel.releaseCachedEngines()
  }

  /// Story take: Gemma searches Apple Maps for real running spots and picks
  /// one. The one scene that uses the network — labeled on screen.
  private func runMap() async {
    guard let model, let prompt = script.mapPrompt else { return }
    withAnimation { scene = .map }
    let tool = MapSearchTool { [weak self] spots in
      Task { @MainActor in
        withAnimation(.spring(duration: 0.5)) { self?.mapSpots = spots }
      }
    }
    do {
      let session = LanguageModelSession(model: model, tools: [tool])
      let answer = try await session.respond(to: prompt).content
      withAnimation { mapAnswer = answer }
    } catch {
      withAnimation { mapAnswer = "[error] \(error.localizedDescription)" }
    }
  }

  /// Four quick turns over ONE session with four real iOS tools attached —
  /// the model picks the right framework each time, and the payoffs are
  /// system UI (a real notification banner, a Live Activity) and the speaker.
  private func runToolbox() async {
    guard let model else { return }
    withAnimation { scene = .toolbox }
    let battery = BatteryTool { [weak self] summary in
      Task { @MainActor in
        self?.toolFired("check_battery", icon: "battery.100",
          call: "check_battery()", detail: summary)
      }
    }
    let notify = NotifyTool { [weak self] message in
      Task { @MainActor in
        self?.toolFired("schedule_notification", icon: "bell.badge.fill",
          call: "schedule_notification(\"\(message.prefix(26))…\")",
          detail: "Real iOS banner in ~4 s — watch the top of the screen")
        self?.notificationText = message
      }
    }
    let speakTool = SpeakTool { [weak self] text in
      Task { @MainActor in
        self?.toolFired("speak", icon: "waveform",
          call: "speak(\"\(text.prefix(26))…\")",
          detail: "Speaking through the iPhone speaker")
        self?.speak(text)
      }
    }
    let timer = TimerTool { [weak self] label in
      Task { @MainActor in
        self?.toolFired("start_run_timer", icon: "timer",
          call: "start_run_timer(\"\(label.prefix(26))…\")",
          detail: "Live Activity started — it's in the Dynamic Island")
        self?.timerStarted = true
      }
    }
    let session = LanguageModelSession(model: model, tools: [battery, notify, timer, speakTool])
    if let agentic = script.agenticToolboxPrompt {
      // Agent mode: one ask, and FM keeps re-invoking the executor as the
      // model chains tool after tool — the chips light up one by one while a
      // single respond() runs.
      let index = toolboxTurns.count
      withAnimation { toolboxTurns.append(ToolboxTurn(prompt: agentic)) }
      toolboxTick += 1
      do {
        let answer = try await session.respond(to: agentic).content
        withAnimation { toolboxTurns[index].answer = answer }
        // Grace note: if the model role-played the cheer ("Speaking: …")
        // instead of calling the speak tool, voice its final line anyway. The
        // speak chip stays unlit — the app is speaking, not the tool.
        if !answer.isEmpty, !answer.hasPrefix("["), !answer.hasPrefix("{"),
          !answer.contains("tool_call"), !litTools.contains("speak") {
          speak(Self.cheerLine(from: answer))
        }
      } catch {
        withAnimation { toolboxTurns[index].answer = "[error] \(error.localizedDescription)" }
        await recoverEngine()
      }
      toolboxTick += 1
    } else {
      for prompt in script.toolboxPrompts {
        if Task.isCancelled { return }
        let index = toolboxTurns.count
        withAnimation { toolboxTurns.append(ToolboxTurn(prompt: prompt)) }
        toolboxTick += 1
        do {
          let answer = try await session.respond(to: prompt).content
          withAnimation { toolboxTurns[index].answer = answer }
        } catch {
          withAnimation { toolboxTurns[index].answer = "[error] \(error.localizedDescription)" }
        }
        toolboxTick += 1
        await hold(1.6)
      }
    }
    // Leave room for the scheduled banner to drop while still on this scene.
    await hold(2.0)
  }

  private func toolFired(_ name: String, icon: String, call: String, detail: String) {
    withAnimation(.spring(duration: 0.35)) {
      litTools.insert(name)
      toolCallsShown[name] = call
      toolResults[name] = detail
    }
    toolboxTick += 1
  }

  private func speak(_ text: String) {
    spokenText = text
    speaking = true
    try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio)
    try? AVAudioSession.sharedInstance().setActive(true)
    let delegate = SpeechDelegate { [weak self] in
      Task { @MainActor in self?.speaking = false }
    }
    speechDelegate = delegate
    synthesizer.delegate = delegate
    let utterance = AVSpeechUtterance(string: text)
    utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
    synthesizer.speak(utterance)
  }

  private func runChart() async {
    guard let model else { return }
    withAnimation { scene = .chart }
    // One retry — but with a VARIED prompt: guided generation runs at
    // temperature 0 in the adapter, so retrying the identical prompt would
    // deterministically reproduce the same unparseable output.
    for attempt in 1...2 {
      let prompt = attempt == 1
        ? script.chartPrompt
        : script.chartPrompt + " Output only the JSON object — no code fences, no explanations."
      do {
        let session = LanguageModelSession(model: model)
        let items: [BarItem]
        if script.chartWeekly {
          let answer = try await session.respond(generating: WeekPlan.self) {
            prompt
          }.content
          let count = min(answer.days.count, answer.km.count)
          // Keep day order — a training week is a sequence, not a ranking.
          items = (0..<count).map {
            BarItem(label: answer.days[$0], value: max(0, min(15, answer.km[$0])))
          }
        } else {
          let answer = try await session.respond(generating: FoodChart.self) {
            prompt
          }.content
          let count = min(answer.foods.count, answer.scores.count)
          items = (0..<count)
            .map { BarItem(label: answer.foods[$0], value: max(0, min(100, answer.scores[$0]))) }
            .sorted { $0.value > $1.value }
        }
        if items.isEmpty { continue }
        chartData = items
        await hold(0.3)
        withAnimation(.spring(duration: 0.9)) { chartProgress = 1 }
        return
      } catch {
        await recoverEngine()
        if attempt == 2 { chartError = error.localizedDescription }
      }
    }
  }

  // MARK: Helpers

  private func hold(_ seconds: Double) async {
    try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
  }

  /// The vision-scene photo, per the script's preference order (the offline
  /// take uses the stock apple.png; the story take prefers showcase.jpg).
  private static func bundledVisionImage(names: [String]) -> Data? {
    for name in names {
      for ext in ["jpg", "jpeg", "png"] {
        if let url = Bundle.main.url(forResource: name, withExtension: ext),
          let data = try? Data(contentsOf: url) {
          return normalizedPNG(from: data) ?? data
        }
      }
    }
    return nil
  }

  /// Decode → downscale → re-encode as PNG before handing the image to the
  /// engine: strips EXIF/JPEG variables its decoder may reject (the stock
  /// apple.png worked; a camera JPEG errored) and keeps vision prefill fast.
  private static func normalizedPNG(from data: Data, maxDimension: CGFloat = 768) -> Data? {
    guard let image = UIImage(data: data) else { return nil }
    let scale = min(1, maxDimension / max(image.size.width, image.size.height))
    let size = CGSize(width: image.size.width * scale, height: image.size.height * scale)
    let format = UIGraphicsImageRendererFormat.default()
    format.scale = 1
    let rendered = UIGraphicsImageRenderer(size: size, format: format).image { _ in
      image.draw(in: CGRect(origin: .zero, size: size))
    }
    return rendered.pngData()
  }
}

/// Presents Gemma's scheduled notification as a banner even though the app is
/// in the foreground — the on-screen payoff of the notify tool.
private final class BannerDelegate: NSObject, UNUserNotificationCenterDelegate {
  func userNotificationCenter(
    _ center: UNUserNotificationCenter, willPresent notification: UNNotification
  ) async -> UNNotificationPresentationOptions {
    [.banner, .sound]
  }
}

/// Flips the waveform off when AVSpeechSynthesizer finishes talking.
private final class SpeechDelegate: NSObject, AVSpeechSynthesizerDelegate {
  private let onFinish: () -> Void
  init(onFinish: @escaping () -> Void) { self.onFinish = onFinish }
  func speechSynthesizer(
    _ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance
  ) {
    onFinish()
  }
}

#endif
