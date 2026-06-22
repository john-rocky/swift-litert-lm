// swift-tools-version: 5.9
//
// LiteRTLMFoundationModels — the *lean* Apple Foundation Models backend for
// LiteRT-LM, carved out as a candidate to contribute upstream to
// google-ai-edge/litert-lm's Swift API.
//
// It depends on **only the LiteRT-LM core Swift wrapper** (the `LiteRTLM`
// product — `Engine` / `Conversation` / `Message` / …) plus Apple's
// `FoundationModels`. No app conveniences (no downloader, no model catalog, no
// `LiteRTChat`): those stay in the parent `swift-litert-lm` package. Everything
// here is gated by `#if canImport(FoundationModels)` + `@available(iOS 27/macOS
// 27)`, so on non-Apple platforms it compiles to nothing and the core is
// unaffected.
//
// This nested package exists to *prove the carve compiles against the core
// alone*. For the actual PR these files would be added as a gated module/target
// inside the upstream Swift package.

import PackageDescription

let package = Package(
  name: "LiteRTLMFoundationModels",
  platforms: [.iOS(.v16), .macOS(.v13)],
  products: [
    .library(name: "LiteRTLMFoundationModels", targets: ["LiteRTLMFoundationModels"])
  ],
  dependencies: [
    .package(name: "swift-litert-lm", path: "../..")
  ],
  targets: [
    .target(
      name: "LiteRTLMFoundationModels",
      dependencies: [.product(name: "LiteRTLM", package: "swift-litert-lm")]
    ),
    // Headless macOS functional test: drives the adapter end-to-end (respond /
    // guided / tools) so it can be verified without a device.
    //   swift run fmtest /path/to/model.litertlm
    .executableTarget(
      name: "fmtest",
      dependencies: [
        "LiteRTLMFoundationModels",
        .product(name: "LiteRTLM", package: "swift-litert-lm"),
      ]
    ),
  ]
)
