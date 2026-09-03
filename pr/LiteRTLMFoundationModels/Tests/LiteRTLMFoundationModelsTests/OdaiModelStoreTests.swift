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

// `OdaiModelStore` over the parent package's `ModelDownloader`, against a
// loopback HTTP/1.1 server (POSIX socket, one thread per connection). No model
// file, no network beyond 127.0.0.1, no token. The same three shapes the host
// `odai fetch` and the Android `ModelStore` tests cover: an interrupted
// transfer resumes with `Range` (here: the downloader's chunk bitmap), wrong
// bytes are discarded and the previously verified file is kept, offline never
// opens a connection.

import CryptoKit
import Foundation
import XCTest

import LiteRTFoundation
import LiteRTLMFoundationModels

private typealias Store = LiteRTLMFoundationModels.OdaiModelStore

// The lean package has no downloader; the parent's is the implementation of
// the seam (same method the root package's conformance provides).
extension ModelDownloader: LiteRTLMFoundationModels.ModelFileDownloading {}

// MARK: - Loopback HTTP/1.1 server

/// Serves one byte string with HEAD / GET / `Range`, `Connection: close` per
/// request. Requests are recorded; a gate can hold every ranged request at or
/// past a byte offset until the test releases it (to interrupt a transfer with
/// the earlier chunks landed and a later one still in flight).
private final class LoopbackServer: @unchecked Sendable {
  struct Request: Equatable { let method: String; let path: String; let range: String? }

  private(set) var port: UInt16 = 0
  private let lock = NSLock()
  private var body: Data
  private var requestLog: [Request] = []
  private var gateOffset: Int64? = nil
  private let gate = DispatchSemaphore(value: 0)
  private var gateReleased = false
  private var listenFD: Int32 = -1
  private var acceptThread: Thread?

  init(body: Data) { self.body = body }

  var requests: [Request] { lock.lock(); defer { lock.unlock() }; return requestLog }
  func clearRequests() { lock.lock(); requestLog.removeAll(); lock.unlock() }
  func setBody(_ d: Data) { lock.lock(); body = d; lock.unlock() }
  private func currentBody() -> Data { lock.lock(); defer { lock.unlock() }; return body }

  /// Hold ranged requests whose first byte is >= `offset` until `releaseGate()`.
  func holdRanges(from offset: Int64) { lock.lock(); gateOffset = offset; gateReleased = false; lock.unlock() }
  func releaseGate() {
    lock.lock(); gateReleased = true; gateOffset = nil; lock.unlock()
    for _ in 0..<64 { gate.signal() }
  }

  func start() throws {
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    guard fd >= 0 else { throw NSError(domain: "loopback", code: 1) }
    var one: Int32 = 1
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, socklen_t(MemoryLayout<Int32>.size))
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))
    var addr = sockaddr_in()
    addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = 0
    addr.sin_addr.s_addr = inet_addr("127.0.0.1")
    let bindResult = withUnsafePointer(to: &addr) {
      $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
    }
    guard bindResult == 0, listen(fd, 16) == 0 else { close(fd); throw NSError(domain: "loopback", code: 2) }
    var bound = sockaddr_in()
    var len = socklen_t(MemoryLayout<sockaddr_in>.size)
    _ = withUnsafeMutablePointer(to: &bound) {
      $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &len) }
    }
    port = UInt16(bigEndian: bound.sin_port)
    listenFD = fd
    let t = Thread { [weak self] in self?.acceptLoop(fd) }
    t.name = "loopback-accept"
    t.start()
    acceptThread = t
  }

  func stop() {
    releaseGate()
    if listenFD >= 0 { close(listenFD); listenFD = -1 }
  }

  var baseURL: URL { URL(string: "http://127.0.0.1:\(port)")! }

  private func acceptLoop(_ fd: Int32) {
    while true {
      let c = accept(fd, nil, nil)
      if c < 0 { return }
      var one: Int32 = 1
      setsockopt(c, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))
      let t = Thread { [weak self] in self?.serve(c) }
      t.start()
    }
  }

  private func readHeaders(_ c: Int32) -> String? {
    var buf = Data()
    var chunk = [UInt8](repeating: 0, count: 4096)
    while true {
      let n = read(c, &chunk, chunk.count)
      if n <= 0 { return nil }
      buf.append(chunk, count: n)
      if let r = buf.range(of: Data("\r\n\r\n".utf8)) { return String(decoding: buf[..<r.lowerBound], as: UTF8.self) }
      if buf.count > 65536 { return nil }
    }
  }

  private func writeAll(_ c: Int32, _ data: Data) {
    data.withUnsafeBytes { raw in
      var off = 0
      while off < raw.count {
        let n = write(c, raw.baseAddress! + off, raw.count - off)
        if n <= 0 { return }
        off += n
      }
    }
  }

  private func serve(_ c: Int32) {
    defer { close(c) }
    guard let head = readHeaders(c) else { return }
    let lines = head.split(separator: "\r\n", omittingEmptySubsequences: false).map(String.init)
    let parts = lines.first?.split(separator: " ").map(String.init) ?? []
    guard parts.count >= 2 else { return }
    let method = parts[0], path = parts[1]
    var range: String? = nil
    for l in lines.dropFirst() {
      if let colon = l.firstIndex(of: ":"), l[..<colon].lowercased() == "range" {
        range = l[l.index(after: colon)...].trimmingCharacters(in: .whitespaces)
      }
    }
    lock.lock(); requestLog.append(Request(method: method, path: path, range: range)); lock.unlock()

    let full = currentBody()
    let total = Int64(full.count)
    var status = 200
    var start: Int64 = 0, end: Int64 = total - 1
    if let r = range, r.hasPrefix("bytes=") {
      let spec = r.dropFirst("bytes=".count).split(separator: "-", omittingEmptySubsequences: false)
      if let s = Int64(spec[0]) {
        start = s
        if spec.count > 1, let e = Int64(spec[1]) { end = min(e, total - 1) }
        status = 206
      }
    }
    // Gate: hold this ranged request until the test releases it.
    lock.lock()
    let held = status == 206 && gateOffset.map { start >= $0 } == true && !gateReleased
    lock.unlock()
    if held { gate.wait() }

    let slice = full[Int(start)...Int(end)]
    var h = "HTTP/1.1 \(status) \(status == 206 ? "Partial Content" : "OK")\r\n"
    h += "Accept-Ranges: bytes\r\nConnection: close\r\nContent-Type: application/octet-stream\r\n"
    h += "Content-Length: \(status == 206 ? slice.count : full.count)\r\n"
    if status == 206 { h += "Content-Range: bytes \(start)-\(end)/\(total)\r\n" }
    h += "\r\n"
    writeAll(c, Data(h.utf8))
    if method != "HEAD" { writeAll(c, Data(slice)) }
  }
}

// MARK: - Helpers

private func deterministicBytes(_ count: Int, seed: UInt64) -> Data {
  var d = Data(count: count)
  var x = seed
  d.withUnsafeMutableBytes { raw in
    let p = raw.bindMemory(to: UInt8.self)
    for i in 0..<count {
      x = x &* 6364136223846793005 &+ 1442695040888963407
      p[i] = UInt8(truncatingIfNeeded: x >> 33)
    }
  }
  return d
}

private func hex(_ d: Data) -> String { SHA256.hash(data: d).map { String(format: "%02x", $0) }.joined() }

private final class ProgressBox: @unchecked Sendable {
  private let lock = NSLock()
  private(set) var events: [Store.Progress] = []
  var onDownloading: (@Sendable (Int64, Int64) -> Void)?
  func record(_ p: Store.Progress) {
    lock.lock(); events.append(p); lock.unlock()
    if case .downloading(let done, let total) = p { onDownloading?(done, total) }
  }
  var maxDownloaded: Int64 {
    lock.lock(); defer { lock.unlock() }
    return events.compactMap { if case .downloading(let d, _) = $0 { return d } else { return nil } }.max() ?? 0
  }
}

private final class TaskBox: @unchecked Sendable {
  private let lock = NSLock()
  private var task: Task<URL, Error>?
  func set(_ t: Task<URL, Error>) { lock.lock(); task = t; lock.unlock() }
  func cancel() { lock.lock(); task?.cancel(); lock.unlock() }
}

// MARK: - Tests

final class OdaiModelStoreTests: XCTestCase {
  private var dir: URL!
  private var server: LoopbackServer!

  override func setUp() async throws {
    try await super.setUp()
    dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("odai-store-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
  }

  override func tearDown() async throws {
    server?.stop()
    try? FileManager.default.removeItem(at: dir)
    try await super.tearDown()
  }

  private func ref(_ body: Data, file: String = "model.litertlm") -> Store.Ref {
    Store.Ref(
      file: file, sourceURL: server.baseURL.appendingPathComponent("resolve/abc/\(file)"),
      sha256: hex(body), sizeBytes: Int64(body.count), component: "gemma4",
      variant: "litert-lm-test", format: "litertlm")
  }

  private func store() -> Store { Store(directory: dir, downloader: ModelDownloader.shared) }

  private func fileSize(_ u: URL) -> Int64 {
    ((try? FileManager.default.attributesOfItem(atPath: u.path))?[.size] as? NSNumber)?.int64Value ?? -1
  }

  /// Interrupt a chunked transfer after the first chunks landed; the next
  /// `ensure` sends `Range` only for the missing chunk, the verified file
  /// lands with its sidecar, and the staging files are gone.
  func testInterruptedTransferResumesWithRangeAndVerifies() async throws {
    // 40 MiB → three 16 MiB-geometry chunks (the downloader splits above 16 MiB).
    let body = deterministicBytes(40 << 20, seed: 1)
    server = LoopbackServer(body: body); try server.start()
    let r = ref(body)
    let s = store()

    // Chunks at offset >= 32 MiB are held at the server; cancel once 32 MiB landed.
    server.holdRanges(from: 32 << 20)
    let progress = ProgressBox()
    let handle = TaskBox()
    progress.onDownloading = { done, _ in if done >= (32 << 20) { handle.cancel() } }
    let first = Task<URL, Error> { try await s.ensure(r) { progress.record($0) } }
    handle.set(first)
    do {
      _ = try await first.value
      XCTFail("the first transfer should have been cancelled")
    } catch {
      // cancelled (CancellationError or URLError.cancelled through the downloader)
    }
    server.releaseGate()
    XCTAssertGreaterThanOrEqual(progress.maxDownloaded, 32 << 20)
    let staging = dir.appendingPathComponent("model.litertlm.new")
    XCTAssertFalse(FileManager.default.fileExists(atPath: dir.appendingPathComponent("model.litertlm").path))
    XCTAssertTrue(FileManager.default.fileExists(atPath: staging.appendingPathExtension("partial").path), "partial kept")
    XCTAssertTrue(FileManager.default.fileExists(atPath: staging.appendingPathExtension("dl-bits").path), "bitmap kept")
    let firstRanges = Set(server.requests.compactMap(\.range))
    XCTAssertEqual(firstRanges, ["bytes=0-16777215", "bytes=16777216-33554431", "bytes=33554432-41943039"])

    // Second run: only the missing chunk is requested.
    server.clearRequests()
    let second = ProgressBox()
    let url = try await s.ensure(r) { second.record($0) }
    let gets = server.requests.filter { $0.method == "GET" }
    XCTAssertEqual(gets.map(\.range), ["bytes=33554432-41943039"], "resume asks for the missing chunk only")
    XCTAssertEqual(server.requests.filter { $0.method == "HEAD" }.count, 1)
    XCTAssertEqual(url, dir.appendingPathComponent("model.litertlm"))
    XCTAssertEqual(fileSize(url), Int64(body.count))
    XCTAssertEqual(try hex(Data(contentsOf: url)), r.sha256)
    XCTAssertFalse(FileManager.default.fileExists(atPath: staging.path))
    XCTAssertFalse(FileManager.default.fileExists(atPath: staging.appendingPathExtension("partial").path))
    XCTAssertFalse(FileManager.default.fileExists(atPath: staging.appendingPathExtension("dl-bits").path))
    let sidecar = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("model.litertlm.odai.json"))) as? [String: Any]
    XCTAssertEqual(sidecar?["sha256"] as? String, r.sha256)
    XCTAssertEqual((sidecar?["size_bytes"] as? NSNumber)?.int64Value, Int64(body.count))
    XCTAssertEqual(sidecar?["source_url"] as? String, r.sourceURL.absoluteString)
    XCTAssertEqual(sidecar?["component"] as? String, "gemma4")
    XCTAssertTrue(second.events.contains { if case .verifying = $0 { return true } else { return false } }, "second pass reported")
    let cached = await s.isCached(r)
    XCTAssertTrue(cached)

    // Cached: no request at all, offline or not.
    server.clearRequests()
    _ = try await s.ensure(r, offline: true)
    _ = try await s.ensure(r)
    XCTAssertTrue(server.requests.isEmpty)
  }

  /// The lock names a new sha256; the server serves other bytes. The download
  /// is discarded, the previously verified file (old sha256, old sidecar) is
  /// still there and still what the store hands out for its own ref.
  func testChecksumMismatchDiscardsDownloadAndKeepsPreviousFile() async throws {
    let old = deterministicBytes(1 << 20, seed: 2)
    server = LoopbackServer(body: old); try server.start()
    let s = store()
    let oldRef = ref(old)
    let placed = try await s.ensure(oldRef)
    XCTAssertEqual(try Data(contentsOf: placed), old)

    // New lock: same file name, a different sha256 (a new model version).
    let wanted = deterministicBytes(1 << 20, seed: 3)
    let newRef = ref(wanted)
    server.setBody(deterministicBytes(1 << 20, seed: 4))  // wrong bytes, right size
    server.clearRequests()
    do {
      _ = try await s.ensure(newRef)
      XCTFail("wrong bytes must not be accepted")
    } catch let Store.Error.checksumMismatch(r, actual, previousKept) {
      XCTAssertEqual(r, newRef)
      XCTAssertEqual(actual, hex(deterministicBytes(1 << 20, seed: 4)))
      XCTAssertTrue(previousKept)
    }
    XCTAssertFalse(server.requests.filter { $0.method == "GET" }.isEmpty, "a transfer happened")
    XCTAssertEqual(try Data(contentsOf: placed), old, "old weights untouched")
    XCTAssertFalse(FileManager.default.fileExists(atPath: dir.appendingPathComponent("model.litertlm.new").path))
    let sidecar = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("model.litertlm.odai.json"))) as? [String: Any]
    XCTAssertEqual(sidecar?["sha256"] as? String, oldRef.sha256, "old sidecar untouched")
    let oldCached = await s.isCached(oldRef)
    let newCached = await s.isCached(newRef)
    XCTAssertTrue(oldCached)
    XCTAssertFalse(newCached)

    // Offline, the new ref is an explicit error (never the old bytes under the new name).
    server.clearRequests()
    do {
      _ = try await s.ensure(newRef, offline: true)
      XCTFail("offline + not cached must throw")
    } catch let Store.Error.notCached(r) {
      XCTAssertEqual(r, newRef)
    }
    XCTAssertTrue(server.requests.isEmpty)
  }

  /// Offline never opens a connection: an empty store throws `notCached`
  /// (no HEAD, no GET), and a side-loaded file is accepted only after one hash.
  func testOfflineIsAnExplicitErrorWithoutAConnection() async throws {
    let body = deterministicBytes(256 << 10, seed: 5)
    server = LoopbackServer(body: body); try server.start()
    let s = store()
    let r = ref(body)
    do {
      _ = try await s.ensure(r, offline: true)
      XCTFail("must throw notCached")
    } catch let Store.Error.notCached(got) {
      XCTAssertEqual(got, r)
    }
    XCTAssertTrue(server.requests.isEmpty, "offline opened a connection")
    XCTAssertFalse(FileManager.default.fileExists(atPath: dir.appendingPathComponent("model.litertlm").path))

    // Side-loaded (no sidecar): offline is fine once the hash matches; a
    // side-loaded file with other bytes is not cached.
    try body.write(to: dir.appendingPathComponent("model.litertlm"))
    let url = try await s.ensure(r, offline: true)
    XCTAssertEqual(url.lastPathComponent, "model.litertlm")
    XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("model.litertlm.odai.json").path))
    XCTAssertTrue(server.requests.isEmpty)

    let other = ref(deterministicBytes(256 << 10, seed: 6), file: "other.litertlm")
    try body.write(to: dir.appendingPathComponent("other.litertlm"))
    do {
      _ = try await s.ensure(other, offline: true)
      XCTFail("side-loaded bytes with another hash are not cached")
    } catch Store.Error.notCached {
    }
    XCTAssertTrue(server.requests.isEmpty)
  }

  /// `Ref.fromEdgeLock` reads the odai lock shape (`selections[].artifact`,
  /// snake_case keys) and refuses to guess between several selections.
  func testRefFromEdgeLockReadsTheArtifactBlock() throws {
    let lock = """
    {
      "edge_lock_version": "0.1",
      "selections": [
        {
          "artifact": {
            "component": "gemma4", "file": "gemma-4-E2B-it.litertlm", "format": "litertlm",
            "sha256": "181938105E0EEFD105961417E8DA75903EACDA102C4FCE9CE90F50B97139A63C",
            "size_bytes": 2588147712,
            "source_url": "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1/gemma-4-E2B-it.litertlm",
            "variant": "litert-lm-gemma-4-e2b-it"
          },
          "component": "gemma4", "device_slug": "iphone-17-pro", "os": "ios", "runtime": "litert-lm", "backend": "default"
        },
        {
          "artifact": {
            "component": "gemma4", "file": "gemma-4-E2B-it.litertlm", "format": "litertlm",
            "sha256": "181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c",
            "size_bytes": 2588147712,
            "source_url": "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1/gemma-4-E2B-it.litertlm",
            "variant": "litert-lm-gemma-4-e2b-it"
          },
          "component": "gemma4", "device_slug": "galaxy-s26", "os": "android", "runtime": "litert-lm", "backend": "gpu"
        },
        { "component": "lfm", "device_slug": "galaxy-s26", "os": "android", "runtime": "litert-lm" }
      ]
    }
    """
    let url = dir.appendingPathComponent("edge.lock.json")
    try Data(lock.utf8).write(to: url)

    let r = try Store.Ref.fromEdgeLock(url, component: "gemma4", device: "iphone-17-pro")
    XCTAssertEqual(r.file, "gemma-4-E2B-it.litertlm")
    XCTAssertEqual(r.sha256, "181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c", "lower-cased")
    XCTAssertEqual(r.sizeBytes, 2_588_147_712)
    XCTAssertEqual(r.sourceURL.host, "huggingface.co")
    XCTAssertEqual(r.variant, "litert-lm-gemma-4-e2b-it")
    XCTAssertEqual(r.format, "litertlm")

    XCTAssertThrowsError(try Store.Ref.fromEdgeLock(url, component: "gemma4")) { e in
      guard case Store.Error.ambiguousSelection(_, let devices) = e else { return XCTFail("\(e)") }
      XCTAssertEqual(devices, ["iphone-17-pro", "galaxy-s26"])
    }
    XCTAssertThrowsError(try Store.Ref.fromEdgeLock(url, component: "lfm")) { e in
      guard case Store.Error.noArtifact("lfm", "galaxy-s26") = e else { return XCTFail("\(e)") }
    }
    XCTAssertThrowsError(try Store.Ref.fromEdgeLock(url, component: "qwen35")) { e in
      guard case Store.Error.selectionNotFound("qwen35", nil) = e else { return XCTFail("\(e)") }
    }
    XCTAssertThrowsError(try Store.Ref.fromEdgeLock(url, component: "gemma4", device: "pixel-8a")) { e in
      guard case Store.Error.selectionNotFound("gemma4", "pixel-8a") = e else { return XCTFail("\(e)") }
    }
  }
}
