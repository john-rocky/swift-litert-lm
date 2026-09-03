// Copyright 2026 Daisuke Majima
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Tests that need no `.litertlm` on disk.
//
// `LiteRTExecutor.init` only builds a `LazyEngine`; the weights are read on the
// first `respond`. So executor/engine accounting, configuration identity, and
// capability derivation are all observable against a model path that does not
// exist — which makes them safe to run in CI.

#if canImport(FoundationModels) && compiler(>=6.4)

import CoreGraphics
import FoundationModels
import LiteRTLM
import XCTest

@testable import LiteRTLMFoundationModels

@available(iOS 27.0, macOS 27.0, *)
private struct StubTool: FoundationModels.Tool {
  let name = "get_temperature"
  let description = "Get the current temperature for a city."
  @Generable struct Arguments {
    @Guide(description: "The city name")
    var city: String
  }
  func call(arguments: Arguments) async throws -> String { "21°C" }
}

@available(iOS 27.0, macOS 27.0, *)
@Generable
private struct PrimaryColors {
  @Guide(description: "Exactly three additive primary colors")
  var colors: [String]
}

@available(iOS 27.0, macOS 27.0, *)
final class AdapterTests: XCTestCase {
  private static let modelPath = "/tmp/litertlm-tests-nonexistent.litertlm"

  override func setUp() async throws {
    try await super.setUp()
    await LiteRTLanguageModel.releaseCachedEngines()
  }

  override func tearDown() async throws {
    await LiteRTLanguageModel.releaseCachedEngines()
    try await super.tearDown()
  }

  // MARK: - Engine sharing

  /// Foundation Models builds one executor per session — a plain session and a
  /// tool-enabled session over the same model yield two. They must resolve to a
  /// single engine, or the multi-GB weights load twice and the app OOMs.
  func testPlainAndToolSessionsShareOneEngine() throws {
    let model = LiteRTLanguageModel(
      engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .cpu()))
    XCTAssertEqual(EngineCache.shared.count, 0)

    let plain = LanguageModelSession(model: model)
    plain.prewarm()
    let tooled = LanguageModelSession(model: model, tools: [StubTool()])
    tooled.prewarm()

    withExtendedLifetime((plain, tooled)) {
      XCTAssertEqual(
        EngineCache.shared.count, 1,
        "two sessions over one model must share a single engine")
    }
  }

  /// Two models over the *same* file but different backends genuinely need
  /// different engines, and must not collide in the cache.
  func testDifferentBackendsDoNotShareAnEngine() throws {
    let cpu = LiteRTLanguageModel(
      engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .cpu()))
    let gpu = LiteRTLanguageModel(
      engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .gpu))

    let a = LanguageModelSession(model: cpu)
    a.prewarm()
    let b = LanguageModelSession(model: gpu)
    b.prewarm()

    withExtendedLifetime((a, b)) {
      XCTAssertEqual(EngineCache.shared.count, 2)
    }
  }

  /// `visualTokenBudget` is a conversation-level setting, so two models over the
  /// same engine configuration with different budgets share one engine (the
  /// multi-GB weights load once).
  func testDifferentVisualTokenBudgetsShareOneEngine() throws {
    let config = try EngineConfig(modelPath: Self.modelPath, backend: .cpu())
    let small = LiteRTLanguageModel(engineConfig: config, visualTokenBudget: 70)
    let large = LiteRTLanguageModel(engineConfig: config, visualTokenBudget: 280)

    let a = LanguageModelSession(model: small)
    a.prewarm()
    let b = LanguageModelSession(model: large)
    b.prewarm()

    withExtendedLifetime((a, b)) {
      XCTAssertEqual(EngineCache.shared.count, 1)
    }
  }

  // MARK: - Configuration identity

  /// Regression: `Configuration` once hashed on `modelPath` alone, so a `.gpu`
  /// model silently received a `.cpu` engine.
  func testConfigurationDistinguishesBackend() throws {
    let cpu = LiteRTExecutor.Configuration(
      engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .cpu()))
    let gpu = LiteRTExecutor.Configuration(
      engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .gpu))
    XCTAssertNotEqual(cpu, gpu)
  }

  func testConfigurationDistinguishesMaxNumTokens() throws {
    let small = LiteRTExecutor.Configuration(
      engineConfig: try EngineConfig(modelPath: Self.modelPath, maxNumTokens: 512))
    let large = LiteRTExecutor.Configuration(
      engineConfig: try EngineConfig(modelPath: Self.modelPath, maxNumTokens: 4096))
    XCTAssertNotEqual(small, large)
  }

  func testConfigurationDistinguishesVisualTokenBudget() throws {
    let config = try EngineConfig(modelPath: Self.modelPath)
    XCTAssertNotEqual(
      LiteRTExecutor.Configuration(engineConfig: config, visualTokenBudget: 64),
      LiteRTExecutor.Configuration(engineConfig: config, visualTokenBudget: 256))
  }

  func testIdenticalConfigurationsCompareEqual() throws {
    let a = LiteRTExecutor.Configuration(
      engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .gpu))
    let b = LiteRTExecutor.Configuration(
      engineConfig: try EngineConfig(modelPath: Self.modelPath, backend: .gpu))
    XCTAssertEqual(a, b)
    XCTAssertEqual(a.hashValue, b.hashValue)
  }

  /// Regression: the adapter used to mirror a subset of `EngineConfig`'s fields
  /// into its own `Configuration`, so `cacheDir` / `loraRank` / `audioLoraRank` /
  /// `maxNumImages` were dropped on the way to the engine.
  func testConfigurationCarriesEveryEngineConfigField() throws {
    let engineConfig = try EngineConfig(
      modelPath: Self.modelPath,
      backend: .gpu,
      visionBackend: .gpu,
      audioBackend: .cpu(threadCount: 2),
      maxNumTokens: 4096,
      cacheDir: "/tmp/litertlm-tests-cache",
      loraRank: 8,
      audioLoraRank: 4,
      maxNumImages: 3)

    let carried = LiteRTLanguageModel(engineConfig: engineConfig)
      .executorConfiguration.engineConfig

    XCTAssertEqual(carried, engineConfig)
    XCTAssertEqual(carried.cacheDir, "/tmp/litertlm-tests-cache")
    XCTAssertEqual(carried.loraRank, 8)
    XCTAssertEqual(carried.audioLoraRank, 4)
    XCTAssertEqual(carried.maxNumImages, 3)
    XCTAssertEqual(carried.maxNumTokens, 4096)
    XCTAssertEqual(carried.audioBackend, .cpu(threadCount: 2))
  }

  /// `init(engineConfig:)` honours the caller's `cacheDir`; the convenience
  /// initializer supplies the app's Caches directory when none is given.
  func testCacheDirIsHonouredAndDefaulted() throws {
    let explicit = LiteRTLanguageModel(
      engineConfig: try EngineConfig(modelPath: Self.modelPath, cacheDir: "/tmp/explicit"))
    XCTAssertEqual(explicit.executorConfiguration.engineConfig.cacheDir, "/tmp/explicit")

    let sugared = try LiteRTLanguageModel(modelPath: Self.modelPath)
    let defaulted = try XCTUnwrap(sugared.executorConfiguration.engineConfig.cacheDir)
    XCTAssertTrue(defaulted.contains("Caches"), "got \(defaulted)")
  }

  // MARK: - Capabilities

  func testVisionCapabilityTracksVisionBackend() throws {
    let textOnly = LiteRTLanguageModel(
      engineConfig: try EngineConfig(modelPath: Self.modelPath))
    XCTAssertFalse(textOnly.capabilities.contains(.vision))
    XCTAssertTrue(textOnly.capabilities.contains(.guidedGeneration))
    XCTAssertTrue(textOnly.capabilities.contains(.toolCalling))

    let vision = LiteRTLanguageModel(
      engineConfig: try EngineConfig(modelPath: Self.modelPath, visionBackend: .gpu))
    XCTAssertTrue(vision.capabilities.contains(.vision))
  }

  /// An image attachment on a model that did not declare `.vision` is refused
  /// with `LanguageModelError.unsupportedCapability(.vision)` before the engine
  /// is touched — never silently dropped. Runs without a model file because the
  /// check precedes engine creation.
  func testImageOnTextOnlyModelIsUnsupportedCapability() async throws {
    let textOnly = LiteRTLanguageModel(
      engineConfig: try EngineConfig(modelPath: Self.modelPath))
    let image = try XCTUnwrap(Self.makeImage())
    let transcript = Transcript(entries: [
      .prompt(
        .init(segments: [
          .text(.init(content: "What is in this picture?")),
          .attachment(.init(content: .image(.init(image)))),
        ]))
    ])
    let session = LanguageModelSession(model: textOnly, transcript: transcript)
    do {
      _ = try await session.respond(to: "Answer briefly.")
      XCTFail("expected unsupportedCapability")
    } catch let error as LanguageModelError {
      guard case .unsupportedCapability(let unsupported) = error else {
        return XCTFail("unexpected LanguageModelError: \(error)")
      }
      XCTAssertEqual(unsupported.capability, .vision)
    }
  }

  // MARK: - Guided generation prompt

  /// The guided prompt carries a field guide and a placeholder instance, never
  /// the raw schema. Gemma 4 E2B on macOS (Xcode 27 beta 5, 2026-09-04) echoed
  /// the schema dump back (`{"colors": {"type": "array", "items": ...}}`) and
  /// FM failed with "GeneratedContent does not contain an array"; no
  /// post-processing can turn a schema node into the array it stands for.
  func testGuidedInstructionsShowASkeletonNotTheSchema() throws {
    let json = String(
      decoding: try JSONEncoder().encode(PrimaryColors.generationSchema), as: UTF8.self)
    let text = LiteRTExecutor.guidedInstructions(fromSchemaJSON: json)
    XCTAssertTrue(text.contains("{\"colors\": [\"<string>\"]}"), text)
    XCTAssertTrue(
      text.contains("- colors (array of string): Exactly three additive primary colors"), text)
    for token in ["\"type\"", "\"properties\"", "x-order", "additionalProperties", "\"required\""] {
      XCTAssertFalse(text.contains(token), "schema leaked into the prompt: \(token)\n\(text)")
    }
  }

  /// `$ref` into `$defs`, `enum`, optionals (absent from `required`), nested
  /// objects and arrays of objects, in `x-order` — the shapes the beta 5
  /// `GenerationSchema` encoder emits for a `@Generable` struct.
  func testGuidedInstructionsFollowRefsEnumsAndOptionals() {
    let schema = """
      {"$defs": {"Address": {"type": "object", "x-order": ["city", "zip"], "required": ["city", "zip"],
         "properties": {"city": {"type": "string", "description": "City name"}, "zip": {"type": "integer"}}}},
       "type": "object", "title": "Person",
       "x-order": ["name", "active", "mood", "address", "nickname", "addresses"],
       "required": ["name", "active", "mood", "address", "addresses"],
       "properties": {
         "name": {"type": "string", "description": "Full name"},
         "active": {"type": "boolean"},
         "mood": {"type": "string", "enum": ["happy", "sad"]},
         "address": {"$ref": "#/$defs/Address"},
         "nickname": {"type": "string"},
         "addresses": {"type": "array", "items": {"$ref": "#/$defs/Address"}}}}
      """
    let text = LiteRTExecutor.guidedInstructions(fromSchemaJSON: schema)
    XCTAssertTrue(
      text.contains(
        "{\"name\": \"<string>\", \"active\": <true or false>, \"mood\": \"<happy | sad>\", "
          + "\"address\": {\"city\": \"<string>\", \"zip\": <integer>}, \"nickname\": \"<string>\", "
          + "\"addresses\": [{\"city\": \"<string>\", \"zip\": <integer>}]}"), text)
    XCTAssertTrue(text.contains("- name (string): Full name"), text)
    XCTAssertTrue(text.contains("- mood (one of \"happy\", \"sad\")"), text)
    XCTAssertTrue(text.contains("- nickname (string, optional)"), text)
    XCTAssertTrue(text.contains("- address.city (string): City name"), text)
    XCTAssertTrue(text.contains("- addresses (array of object)"), text)
    XCTAssertTrue(text.contains("- addresses[].zip (integer)"), text)
    XCTAssertFalse(text.contains("$ref"), text)
  }

  /// A reply whose field holds the field's own schema is reported by name;
  /// a real value, or an object-typed field, is not.
  func testSchemaEchoIsDetectedByField() throws {
    let json = String(
      decoding: try JSONEncoder().encode(PrimaryColors.generationSchema), as: UTF8.self)
    XCTAssertEqual(
      LiteRTExecutor.schemaEchoField(
        in: #"{"colors": {"items": {"type": "string"}, "type": "array"}}"#, schemaJSON: json),
      "colors")
    XCTAssertNil(
      LiteRTExecutor.schemaEchoField(in: #"{"colors": ["red", "green", "blue"]}"#, schemaJSON: json))
    XCTAssertNil(LiteRTExecutor.schemaEchoField(in: #"{"colors": ["red"]}"#, schemaJSON: nil))
    let objectField = #"{"type": "object", "properties": {"box": {"type": "object"}}}"#
    XCTAssertNil(
      LiteRTExecutor.schemaEchoField(in: #"{"box": {"type": "object"}}"#, schemaJSON: objectField))
  }

  private static func makeImage() -> CGImage? {
    let context = CGContext(
      data: nil, width: 2, height: 2, bitsPerComponent: 8, bytesPerRow: 0,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    context?.setFillColor(red: 1, green: 0, blue: 0, alpha: 1)
    context?.fill(CGRect(x: 0, y: 0, width: 2, height: 2))
    return context?.makeImage()
  }
}

#endif
