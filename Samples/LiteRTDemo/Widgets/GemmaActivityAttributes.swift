// LiteRTDemo — Live Activity attributes for the timer Gemma starts in the
// showcase. Compiled into BOTH the app (which requests the activity) and the
// widget extension (which renders it in the Dynamic Island / Lock Screen).

import Foundation
#if canImport(ActivityKit)
import ActivityKit

@available(iOS 16.2, *)
struct GemmaRunAttributes: ActivityAttributes {
  struct ContentState: Codable, Hashable {
    var message: String
    var endDate: Date
  }

  var title: String
}
#endif
