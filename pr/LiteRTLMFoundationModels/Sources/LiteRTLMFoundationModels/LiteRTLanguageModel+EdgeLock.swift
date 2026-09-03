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

// Foundation Models backend from an `edge.lock.json` (ODAI glue).
//
// The two developer lines an `edge add <component>` project ends with:
//
//   let model   = try await LiteRTLanguageModel.fromEdgeLock(lockURL, component: "gemma4", store: store)
//   let session = LanguageModelSession(model: model)      // Apple's exact API
//
// `fromEdgeLock` reads the lock's artifact block, hands it to `OdaiModelStore`
// (download on first use through the injected downloader, sha256-verified,
// offline = explicit error) and builds the backend over the verified file.
// Errors surface as thrown errors here; nothing switches to another model or
// to the system model.

#if canImport(FoundationModels) && compiler(>=6.4)

import Foundation
import FoundationModels
import LiteRTLM

@available(iOS 27.0, macOS 27.0, *)
extension LiteRTLanguageModel {

  /// Build the backend for the `component` an `edge add` wrote into `lockURL`.
  ///
  /// - Parameters:
  ///   - lockURL: The project's `edge.lock.json` (bundle it, or ship it as a resource).
  ///   - component: The `edge add` component (`gemma4`, `lfm`, ...).
  ///   - device: The lock's `device_slug` to use when the lock has several
  ///     selections for the component; nil is fine for a single selection.
  ///   - store: Where verified files live (this lean package has no default
  ///     downloader; the parent package's `ModelDownloader` conforms).
  ///   - offline: Never open a connection; throws `OdaiModelStore.Error.notCached`
  ///     when the verified file is not there.
  ///   - verify: Rehash a cached file even when its sidecar already matches.
  ///   - backend: Main compute backend for the engine (default `.gpu`).
  ///   - maxTokens: KV/context budget (nil = model/engine default).
  ///   - onProgress: Download / verification progress on first use.
  public static func fromEdgeLock(
    _ lockURL: URL,
    component: String,
    device: String? = nil,
    store: OdaiModelStore,
    offline: Bool = false,
    verify: Bool = false,
    backend: Backend = .gpu,
    maxTokens: Int? = 2048,
    onProgress: (@Sendable (OdaiModelStore.Progress) -> Void)? = nil
  ) async throws -> LiteRTLanguageModel {
    let ref = try OdaiModelStore.Ref.fromEdgeLock(lockURL, component: component, device: device)
    let file = try await store.ensure(ref, offline: offline, verify: verify, onProgress: onProgress)
    return try LiteRTLanguageModel(modelPath: file.path, backend: backend, maxTokens: maxTokens)
  }
}

#endif
