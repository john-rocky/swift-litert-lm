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
