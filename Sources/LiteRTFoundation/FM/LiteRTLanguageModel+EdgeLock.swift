// swift-litert-lm — Foundation Models backend from an `edge.lock.json`
//
// The two developer lines an `edge add <component>` project ends with:
//
//   let model   = try await LiteRTLanguageModel.fromEdgeLock(lockURL, component: "gemma4")
//   let session = LanguageModelSession(model: model)      // Apple's exact API
//
// `fromEdgeLock` reads the lock's artifact block, hands it to `OdaiModelStore`
// (download on first use, sha256-verified, resumable, offline = explicit error)
// and builds the file-URL backend over the verified file. Errors surface as
// thrown errors here; nothing switches to another model or to the system model.

#if canImport(FoundationModels) && compiler(>=6.4)

import Foundation
import FoundationModels

@available(iOS 27.0, macOS 27.0, *)
extension LiteRTLanguageModel {

  /// Build the backend for the `component` an `edge add` wrote into `lockURL`.
  ///
  /// - Parameters:
  ///   - lockURL: The project's `edge.lock.json` (bundle it, or ship it as a resource).
  ///   - component: The `edge add` component (`gemma4`, `lfm`, …).
  ///   - device: The lock's `device_slug` to use when the lock has several
  ///     selections for the component; nil is fine for a single selection.
  ///   - store: Where verified files live; default `OdaiModelStore.shared`.
  ///   - offline: Never open a connection; throws `OdaiModelStore.Error.notCached`
  ///     when the verified file is not there.
  ///   - verify: Rehash a cached file even when its sidecar already matches
  ///     (seconds for a multi-GB file; `odai fetch --verify` on the host).
  ///   - modalities: Towers to enable on the engine. The lock's text-generation
  ///     component needs none; pass `.textImage` / `.all` for a vision model.
  ///   - onProgress: Download / verification progress on first use.
  public static func fromEdgeLock(
    _ lockURL: URL,
    component: String,
    device: String? = nil,
    store: OdaiModelStore = .shared,
    offline: Bool = false,
    verify: Bool = false,
    modalities: Modality = [],
    maxTokens: Int = 2048,
    onProgress: (@Sendable (OdaiModelStore.Progress) -> Void)? = nil
  ) async throws -> LiteRTLanguageModel {
    let ref = try OdaiModelStore.Ref.fromEdgeLock(lockURL, component: component, device: device)
    let file = try await store.ensure(ref, offline: offline, verify: verify, onProgress: onProgress)
    return try LiteRTLanguageModel(modelFileURL: file, modalities: modalities, maxTokens: maxTokens)
  }
}

#endif
