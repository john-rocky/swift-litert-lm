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

// Verified model store over an `edge.lock.json` artifact block (ODAI glue;
// not part of the Foundation Models adapter surface itself).
//
// Rules shared with the host `odai fetch` and the Android `ModelStore`
// (odai docs/weight-delivery.md): the lock's `selections[].artifact` —
// `file`, `source_url`, `sha256`, `size_bytes` from the registry's HF snapshot —
// is the only identity a model file has; a file is used only after its sha256
// (and size) matched; offline never opens a connection; a failed verification
// keeps the previously verified file; nothing ever sends a token.
//
// The transfer is injected (`ModelFileDownloading`). The parent package's
// `ModelDownloader` resumes across launches and keeps `Range` over the
// HF -> CDN redirect, but has no checksum, returns early whenever its
// destination exists — whatever the bytes — and never rolls back. So the store
// points the downloader at `<file>.new` (never the final name), hashes that
// file in a second pass with CryptoKit, and only then replaces `<file>`.
//
// Cache layout (flat, `Application Support/odai/models`, excluded from backup):
//
//   <file>                 verified artifact (publisher's file name)
//   <file>.odai.json       sidecar: sha256, size_bytes, source_url, component, variant, verified_at
//   <file>.new             downloaded, not yet verified (the downloader's destination)
//   <file>.new.partial     ModelDownloader's in-flight bytes
//   <file>.new.dl-bits     ModelDownloader's resume bitmap
//
// The sidecar has the same keys as the host's, so a directory filled by
// `odai fetch --verify` (`<dest>/<component>/`) is read as cached here.

import CryptoKit
import Foundation

/// Fetches one URL into one file. This lean package carries no downloader: the
/// parent `swift-litert-lm` conforms its `ModelDownloader` (resume bitmap +
/// Range over the HF redirect); the tests here use that one through this seam.
public protocol ModelFileDownloading: Sendable {
  /// Download `url` to `destination`, resuming a previous attempt if the
  /// implementation can. `expectedBytes` is the size the lock declares (used
  /// when the server reports none). `onProgress` receives (completed, total).
  func downloadArtifact(
    from url: URL, to destination: URL, expectedBytes: Int64?,
    onProgress: @escaping @Sendable (Int64, Int64) -> Void
  ) async throws
}

/// A directory of verified model files, keyed by the lock's artifact block.
public actor OdaiModelStore {

  /// `edge.lock.json` → `selections[].artifact` (odai lock schema v0.1, additive block).
  public struct Ref: Sendable, Codable, Equatable {
    public let file: String
    public let sourceURL: URL
    public let sha256: String
    public let sizeBytes: Int64?
    public let component: String?
    public let variant: String?
    public let format: String?

    public init(
      file: String, sourceURL: URL, sha256: String, sizeBytes: Int64? = nil,
      component: String? = nil, variant: String? = nil, format: String? = nil
    ) {
      self.file = file
      self.sourceURL = sourceURL
      self.sha256 = sha256.lowercased()
      self.sizeBytes = sizeBytes
      self.component = component
      self.variant = variant
      self.format = format
    }

    enum CodingKeys: String, CodingKey {
      case file, sha256, component, variant, format
      case sourceURL = "source_url"
      case sizeBytes = "size_bytes"
    }

    public init(from decoder: Decoder) throws {
      let c = try decoder.container(keyedBy: CodingKeys.self)
      self.init(
        file: try c.decode(String.self, forKey: .file),
        sourceURL: try c.decode(URL.self, forKey: .sourceURL),
        sha256: try c.decode(String.self, forKey: .sha256),
        sizeBytes: try c.decodeIfPresent(Int64.self, forKey: .sizeBytes),
        component: try c.decodeIfPresent(String.self, forKey: .component),
        variant: try c.decodeIfPresent(String.self, forKey: .variant),
        format: try c.decodeIfPresent(String.self, forKey: .format))
    }

    /// Read the artifact block of one selection out of an `edge.lock.json`
    /// (or an `odai.lock.json`): the selection whose `component` is `component`
    /// and, when given, whose `device_slug` is `device`. With several matching
    /// selections and no `device`, the call refuses rather than guessing.
    public static func fromEdgeLock(_ lockURL: URL, component: String, device: String? = nil) throws -> Ref {
      let lock = try JSONDecoder().decode(EdgeLock.self, from: Data(contentsOf: lockURL))
      let mine = lock.selections.filter { ($0.component ?? $0.artifact?.component) == component }
      guard !mine.isEmpty else { throw OdaiModelStore.Error.selectionNotFound(component: component, device: device) }
      let chosen: EdgeLock.Selection
      if let device {
        guard let s = mine.first(where: { $0.deviceSlug == device }) else {
          throw OdaiModelStore.Error.selectionNotFound(component: component, device: device)
        }
        chosen = s
      } else if mine.count == 1 {
        chosen = mine[0]
      } else {
        throw OdaiModelStore.Error.ambiguousSelection(component: component, devices: mine.map(\.deviceSlug))
      }
      guard let ref = chosen.artifact else {
        throw OdaiModelStore.Error.noArtifact(component: component, device: chosen.deviceSlug)
      }
      return ref
    }
  }

  /// The parts of a lock this store reads.
  public struct EdgeLock: Decodable, Sendable {
    public struct Selection: Decodable, Sendable {
      public let deviceSlug: String
      public let component: String?
      public let runtime: String?
      public let backend: String?
      public let os: String?
      public let artifact: Ref?
      enum CodingKeys: String, CodingKey {
        case component, runtime, backend, os, artifact
        case deviceSlug = "device_slug"
      }
    }
    public let selections: [Selection]
  }

  public enum Progress: Sendable {
    /// Bytes landed on disk so far (the downloader's own resume bitmap counts).
    case downloading(completed: Int64, total: Int64)
    /// Second pass: bytes hashed so far.
    case verifying(hashed: Int64, total: Int64)
  }

  public enum Error: Swift.Error, LocalizedError {
    /// Offline and no verified file for this ref (never a silent download).
    case notCached(Ref)
    /// The downloaded bytes did not hash to the lock's sha256. The staging file
    /// was deleted; `previousKept` says whether an older `<file>` is still there.
    case checksumMismatch(Ref, actual: String, previousKept: Bool)
    /// The downloaded file is not the lock's size (checked before hashing).
    case sizeMismatch(Ref, actual: Int64, previousKept: Bool)
    /// This store cannot fetch that ref (directory formats, non-https URLs).
    case notDistributed(Ref, reason: String)
    /// Another `ensure` of the same file is running on this store.
    case downloadInProgress(Ref)
    case selectionNotFound(component: String, device: String?)
    case ambiguousSelection(component: String, devices: [String])
    case noArtifact(component: String, device: String)

    public var errorDescription: String? {
      switch self {
      case .notCached(let r): return "offline and \(r.file) is not cached (verified) in the model store"
      case .checksumMismatch(let r, let actual, let kept):
        return "\(r.file): sha256 \(actual.prefix(12))… does not match the lock's \(r.sha256.prefix(12))…; download discarded, previous file \(kept ? "kept" : "absent")"
      case .sizeMismatch(let r, let actual, let kept):
        return "\(r.file): size \(actual) does not match the lock's \(r.sizeBytes.map(String.init) ?? "?"); download discarded, previous file \(kept ? "kept" : "absent")"
      case .notDistributed(let r, let reason): return "\(r.file) cannot be fetched by this store: \(reason)"
      case .downloadInProgress(let r): return "a download of \(r.file) is already in progress on this store"
      case .selectionNotFound(let c, let d): return "no selection for component \(c)\(d.map { " on \($0)" } ?? "") in the lock"
      case .ambiguousSelection(let c, let ds): return "component \(c) has \(ds.count) selections (\(ds.joined(separator: ", "))); pass device:"
      case .noArtifact(let c, let d): return "selection \(c) on \(d) has no artifact block (run edge add again, or odai fetch --lock cannot deliver it)"
      }
    }
  }

  /// Where the files live. Default: `Application Support/odai/models`.
  public nonisolated let directory: URL
  private let downloader: any ModelFileDownloading
  private var inFlight: Set<String> = []

  public init(directory: URL? = nil, downloader: any ModelFileDownloading) {
    self.directory = directory ?? Self.defaultDirectory()
    self.downloader = downloader
  }

  public static func defaultDirectory() -> URL {
    let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
      ?? URL(fileURLWithPath: NSTemporaryDirectory())
    return base.appendingPathComponent("odai", isDirectory: true).appendingPathComponent("models", isDirectory: true)
  }

  // MARK: - Public API

  /// Path the verified file will have (whether or not it exists yet).
  public nonisolated func url(for ref: Ref) -> URL { directory.appendingPathComponent(ref.file) }

  /// True when `<file>` exists and its sidecar names the same sha256 and size.
  /// No hashing (that is `ensure(_:verify: true)`).
  public func isCached(_ ref: Ref) -> Bool { sidecarMatches(url(for: ref), ref) }

  /// Return the verified file for `ref`, downloading and verifying it first
  /// when needed.
  ///
  /// - offline: never open a connection; throw `notCached` instead.
  /// - verify: rehash a cached file even when the sidecar already matches.
  public func ensure(
    _ ref: Ref, offline: Bool = false, verify: Bool = false,
    onProgress: (@Sendable (Progress) -> Void)? = nil
  ) async throws -> URL {
    let fm = FileManager.default
    let final = url(for: ref)
    let staging = directory.appendingPathComponent(ref.file + ".new")
    let total = ref.sizeBytes ?? -1

    // 1. Cached and trusted by the sidecar (optionally rehashed).
    if sidecarMatches(final, ref) {
      if !verify { return final }
      if try await hashMatches(final, ref, onProgress: onProgress) { return final }
      try? fm.removeItem(at: sidecarURL(final))  // the file changed under the sidecar: treat as not cached
    } else if fm.fileExists(atPath: final.path) {
      // Side-loaded (or written by an older version): the file is trusted only
      // after one hash; a match earns the sidecar, a mismatch is not cached.
      if try await hashMatches(final, ref, onProgress: onProgress) {
        try writeSidecar(final, ref)
        return final
      }
    }

    if offline { throw Error.notCached(ref) }
    try refuseUndistributable(ref)

    guard !inFlight.contains(ref.file) else { throw Error.downloadInProgress(ref) }
    inFlight.insert(ref.file)
    defer { inFlight.remove(ref.file) }

    try fm.createDirectory(at: directory, withIntermediateDirectories: true)
    Self.excludeFromBackup(directory)

    // 2. A staging file left by an earlier run that downloaded but never
    // verified (crash between the two passes): verify it before downloading.
    if fm.fileExists(atPath: staging.path) {
      if try await hashMatches(staging, ref, onProgress: onProgress) {
        return try promote(staging, to: final, ref: ref)
      }
      try? fm.removeItem(at: staging)  // ModelDownloader would return early on it
    }

    // 3. Transfer (resume is the downloader's), then the second pass.
    try await downloader.downloadArtifact(
      from: ref.sourceURL, to: staging, expectedBytes: ref.sizeBytes
    ) { done, all in
      onProgress?(.downloading(completed: done, total: all > 0 ? all : total))
    }

    let previousKept = fm.fileExists(atPath: final.path)
    let size = Self.fileSize(staging)
    if let want = ref.sizeBytes, want != size {
      try? fm.removeItem(at: staging)
      throw Error.sizeMismatch(ref, actual: size, previousKept: previousKept)
    }
    let digest = try await Self.sha256(of: staging, total: size) { hashed in
      onProgress?(.verifying(hashed: hashed, total: size))
    }
    guard digest == ref.sha256 else {
      try? fm.removeItem(at: staging)  // rollback: `final` (old weights, if any) untouched
      throw Error.checksumMismatch(ref, actual: digest, previousKept: previousKept)
    }
    return try promote(staging, to: final, ref: ref)
  }

  /// Delete the verified file and its sidecar (and any staging leftovers).
  public func evict(_ ref: Ref) throws {
    let fm = FileManager.default
    let final = url(for: ref)
    for u in [final, sidecarURL(final),
              directory.appendingPathComponent(ref.file + ".new"),
              directory.appendingPathComponent(ref.file + ".new.partial"),
              directory.appendingPathComponent(ref.file + ".new.dl-bits")] {
      if fm.fileExists(atPath: u.path) { try fm.removeItem(at: u) }
    }
  }

  // MARK: - Internals

  private func refuseUndistributable(_ ref: Ref) throws {
    if let f = ref.format, !["litertlm", "pte", "gguf"].contains(f.lowercased()) {
      throw Error.notDistributed(ref, reason: "unsupported_format \(f) (single files only in v0)")
    }
    guard ref.sourceURL.scheme?.lowercased() == "https" || ref.sourceURL.scheme?.lowercased() == "http" else {
      throw Error.notDistributed(ref, reason: "source_url is not http(s): \(ref.sourceURL)")
    }
    guard !ref.sha256.isEmpty, ref.sha256.count == 64 else {
      throw Error.notDistributed(ref, reason: "lock has no sha256 for this artifact")
    }
  }

  private func promote(_ staging: URL, to final: URL, ref: Ref) throws -> URL {
    let fm = FileManager.default
    if fm.fileExists(atPath: final.path) {
      _ = try fm.replaceItemAt(final, withItemAt: staging)  // atomic; old bytes gone only now
    } else {
      try fm.moveItem(at: staging, to: final)
    }
    Self.excludeFromBackup(final)
    try writeSidecar(final, ref)
    return final
  }

  private func hashMatches(_ file: URL, _ ref: Ref, onProgress: (@Sendable (Progress) -> Void)?) async throws -> Bool {
    let size = Self.fileSize(file)
    if let want = ref.sizeBytes, want != size { return false }
    let digest = try await Self.sha256(of: file, total: size) { hashed in
      onProgress?(.verifying(hashed: hashed, total: size))
    }
    return digest == ref.sha256
  }

  /// Streamed SHA-256 (8 MiB reads) off the actor, so other callers are not
  /// blocked for the seconds a multi-GB file takes.
  nonisolated static func sha256(
    of file: URL, total: Int64, onProgress: (@Sendable (Int64) -> Void)? = nil
  ) async throws -> String {
    try await Task.detached(priority: .utility) {
      let h = try FileHandle(forReadingFrom: file)
      defer { try? h.close() }
      var digest = SHA256()
      var hashed: Int64 = 0
      while let chunk = try h.read(upToCount: 8 << 20), !chunk.isEmpty {
        digest.update(data: chunk)
        hashed += Int64(chunk.count)
        onProgress?(hashed)
      }
      return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }.value
  }

  // Sidecar: same keys as the host's `<file>.odai.json` (odai fetch).
  private nonisolated func sidecarURL(_ final: URL) -> URL {
    final.deletingLastPathComponent().appendingPathComponent(final.lastPathComponent + ".odai.json")
  }

  private func sidecarMatches(_ final: URL, _ ref: Ref) -> Bool {
    let fm = FileManager.default
    guard fm.fileExists(atPath: final.path),
      let data = try? Data(contentsOf: sidecarURL(final)),
      let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
      let sha = obj["sha256"] as? String, sha.lowercased() == ref.sha256
    else { return false }
    let size = Self.fileSize(final)
    if let s = obj["size_bytes"] as? NSNumber, s.int64Value != size { return false }
    if let want = ref.sizeBytes, want != size { return false }
    return true
  }

  private func writeSidecar(_ final: URL, _ ref: Ref) throws {
    var obj: [String: Any] = [
      "sha256": ref.sha256,
      "size_bytes": Self.fileSize(final),
      "source_url": ref.sourceURL.absoluteString,
      "verified_at": ISO8601DateFormatter().string(from: Date()),
    ]
    obj["component"] = ref.component ?? NSNull()
    obj["variant"] = ref.variant ?? NSNull()
    let data = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys, .prettyPrinted])
    try data.write(to: sidecarURL(final), options: .atomic)
  }

  private nonisolated static func fileSize(_ url: URL) -> Int64 {
    ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? NSNumber)?.int64Value ?? -1
  }

  private nonisolated static func excludeFromBackup(_ url: URL) {
    var v = URLResourceValues()
    v.isExcludedFromBackup = true
    var u = url
    try? u.setResourceValues(v)
  }
}
