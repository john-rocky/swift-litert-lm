// LiteRTDemo — app entry point.
//
// Two modes:
//   • Interactive (default): the ContentView chat UI.
//   • G0 self-test (LITERT_G0_TEST=1): headless text→image→audio benchmark for
//     device runs driven by `devicectl`.

import SwiftUI

@main
struct LiteRTDemoApp: App {
  var body: some Scene {
    WindowGroup {
      if G0SelfTest.isRequested {
        G0RunnerView()
      } else {
        ContentView()
      }
    }
  }
}

/// Minimal view shown while the headless G0 self-test runs; results go to the log.
private struct G0RunnerView: View {
  @State private var started = false
  var body: some View {
    VStack(spacing: 12) {
      ProgressView()
      Text("Running G0 self-test…")
        .font(.headline)
      Text("Results are logged with the “G0:” prefix.")
        .font(.caption)
        .foregroundStyle(.secondary)
    }
    .task {
      guard !started else { return }
      started = true
      await G0SelfTest.run()
    }
  }
}
