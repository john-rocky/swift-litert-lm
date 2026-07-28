// LiteRTDemo Widgets — renders the Live Activity Gemma starts from the
// showcase's `start_run_timer` tool.
//
// The system draws these views in the Dynamic Island and on the Lock Screen.
// The countdown ticks by itself via `Text(timerInterval:)` — no timeline
// updates or push tokens needed, so the whole thing works offline and without
// any special entitlement.

import WidgetKit
import SwiftUI
import ActivityKit

@main
struct LiteRTDemoWidgets: WidgetBundle {
  var body: some Widget {
    GemmaRunActivityWidget()
  }
}

struct GemmaRunActivityWidget: Widget {
  private var gradient: LinearGradient {
    LinearGradient(colors: [.blue, .purple], startPoint: .leading, endPoint: .trailing)
  }

  var body: some WidgetConfiguration {
    ActivityConfiguration(for: GemmaRunAttributes.self) { context in
      // Lock Screen / banner presentation.
      HStack(spacing: 12) {
        Image(systemName: "sparkles").font(.title2).foregroundStyle(gradient)
        VStack(alignment: .leading, spacing: 2) {
          Text(context.attributes.title).font(.caption.bold()).foregroundStyle(.secondary)
          Text(context.state.message).font(.callout.weight(.semibold)).lineLimit(1)
        }
        Spacer()
        countdown(context, font: .title3.monospacedDigit().weight(.bold), width: 70)
      }
      .padding(14)
    } dynamicIsland: { context in
      DynamicIsland {
        DynamicIslandExpandedRegion(.leading) {
          Image(systemName: "sparkles").font(.title2).foregroundStyle(gradient)
        }
        DynamicIslandExpandedRegion(.trailing) {
          countdown(context, font: .title3.monospacedDigit().weight(.bold), width: 64)
        }
        DynamicIslandExpandedRegion(.bottom) {
          Text(context.state.message).font(.callout.weight(.semibold)).lineLimit(2)
        }
      } compactLeading: {
        Image(systemName: "sparkles").foregroundStyle(gradient)
      } compactTrailing: {
        countdown(context, font: .caption2.monospacedDigit().weight(.bold), width: 44)
      } minimal: {
        Image(systemName: "sparkles").foregroundStyle(gradient)
      }
    }
  }

  private func countdown(
    _ context: ActivityViewContext<GemmaRunAttributes>, font: Font, width: CGFloat
  ) -> some View {
    Text(timerInterval: Date.now...max(Date.now, context.state.endDate), countsDown: true)
      .font(font)
      .multilineTextAlignment(.trailing)
      .frame(maxWidth: width)
  }
}
