// Corroborate an instance mask with a separate, built-in image classifier.
// Input: one local JPEG path per line. Output: ranked labels (JSON lines).
import Foundation
import Vision
let request = VNClassifyImageRequest()
request.usesCPUOnly = true
while let path = readLine() {
    do {
        try VNImageRequestHandler(url: URL(fileURLWithPath: path)).perform([request])
        let labels = (request.results ?? []).filter {$0.confidence >= 0.007}.map {[$0.identifier, String($0.confidence)]}
        print(String(decoding: try JSONEncoder().encode(labels), as: UTF8.self))
    } catch { print("[]") }
}
