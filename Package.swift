// swift-tools-version: 5.9
//
// swift-litert-lm — LiteRT-LM on iPhone, the easy way, and as an Apple
// Foundation Models backend.
//
// This package layers two faces on top of Google's official LiteRT-LM runtime:
//   • Easy mode  — `LiteRTChat(.gemma4_E2B)`: auto-download, memory-safe, GPU.
//   • FM mode    — `LiteRTLanguageModel`: an Apple Foundation Models backend
//                  (added in a later phase; needs the iOS 27 SDK).
//
// The native runtime ships as prebuilt xcframeworks attached to the official
// LiteRT-LM GitHub releases. We depend on those binary artifacts directly
// (they download cleanly over HTTPS) and vendor the official thin Swift
// wrapper under `Sources/LiteRTLM` (Apache-2.0; see NOTICE). We deliberately do
// NOT add `google-ai-edge/litert-lm` as a SwiftPM git dependency: that repo
// LFS-tracks Android/Linux/Windows prebuilt libs (`prebuilt/*/*.so` …) which
// are irrelevant to Apple platforms but make `swift package resolve` fragile
// (see upstream issue #2407). Binary targets keep the on-ramp bulletproof.

import PackageDescription

let liteRTLMVersion = "v0.19.0"

let package = Package(
  name: "swift-litert-lm",
  platforms: [
    .iOS(.v16),
    .macOS(.v13),
  ],
  products: [
    // The developer-facing library: Easy mode now, FM mode later.
    .library(name: "LiteRTFoundation", targets: ["LiteRTFoundation"]),
    // The vendored official wrapper, exposed for power users who want the raw
    // LiteRT-LM `Engine` / `Conversation` API.
    .library(name: "LiteRTLM", targets: ["LiteRTLM"]),
  ],
  targets: [
    // ── Native runtime (prebuilt, from the official LiteRT-LM releases) ──────
    .binaryTarget(
      name: "CLiteRTLM",
      url:
        "https://github.com/google-ai-edge/LiteRT-LM/releases/download/\(liteRTLMVersion)/CLiteRTLM.xcframework.zip",
      checksum: "b7b2450fc084058adc5185498f8b7810a9297e054a10cf55818b62e1d5f8d65d"
    ),
    .binaryTarget(
      name: "CLiteRTLM_mac",
      url:
        "https://github.com/google-ai-edge/LiteRT-LM/releases/download/\(liteRTLMVersion)/CLiteRTLM_mac.xcframework.zip",
      checksum: "1a9e283a99a119aefe21afda541a5308a7d88aa9cb12f58aa7974038e3369c7c"
    ),

    // ── Vendored official Swift wrapper (Apache-2.0, see NOTICE) ─────────────
    .target(
      name: "LiteRTLM",
      dependencies: [
        .target(name: "CLiteRTLM", condition: .when(platforms: [.iOS])),
        .target(name: "CLiteRTLM_mac", condition: .when(platforms: [.macOS])),
      ],
      path: "Sources/LiteRTLM"
      // No linkerSettings: the vendored binaries are dynamic (an iOS dynamic
      // framework and a macOS dylib), and unsafe flags would make the package
      // unusable as a remote SwiftPM dependency.
    ),

    // ── Our layer: Easy mode + downloader + model catalog (+ FM mode later) ──
    .target(
      name: "LiteRTFoundation",
      dependencies: ["LiteRTLM"],
      path: "Sources/LiteRTFoundation"
    ),
  ]
)
