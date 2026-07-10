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

// CGImage → PNG bytes (cross-platform via ImageIO).

#if canImport(FoundationModels)

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// Encode a `CGImage` as PNG bytes. Returns nil on failure.
func pngData(from cgImage: CGImage) -> Data? {
  let data = NSMutableData()
  guard
    let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)
  else { return nil }
  CGImageDestinationAddImage(dest, cgImage, nil)
  guard CGImageDestinationFinalize(dest) else { return nil }
  return data as Data
}

#endif
