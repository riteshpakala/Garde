//
//  HeadlessCLI.swift
//  Garde
//
//  CLI scoring mode (implied whenever `garde` is given any argument):
//
//      garde <image> [--patch 64|128|256] [--repeat N]
//                    [--json out.json | --json -]
//                    [--model model.xgb.json] [--cpu] [--exact-warp]
//                    [--no-color]
//                    [--dump-features out.json]
//                    [--verify-warp fixture.json]
//
//  Prints a color-coded patch map (P(generated) heat map + weight map) with
//  verdicts and per-run wall times. `--json path` additionally writes the
//  machine-readable result; `--json -` prints ONLY the JSON to stdout (for
//  piping). `--repeat` reruns inference on the same image; run 1 includes
//  Metal/JIT warmup, so use run 2+ for steady-state timing. `--cpu` forces the
//  MLX default device to CPU (A/B check that the hot path normally runs on the
//  GPU). Set CLR_TIMING=1 for per-phase timings on stderr. `--headless` is
//  accepted as a no-op for compatibility with the original CLR app.

import AppKit
import Foundation
import Frigate

private let usageText = """
Usage: garde <image> [options]          score an image (CLI mode)
       garde                            open the macOS app

Options:
  --patch N              outer patch size: 64 (default) | 128 | 256
  --repeat N             rerun inference N times (run 1 = warmup)
  --json out.json        also write machine-readable JSON to a file
  --json -               print ONLY JSON to stdout (for piping)
  --model m.xgb.json     override the bundled model
  --cpu                  force the MLX default device to CPU
  --exact-warp           disable cv2's 1/32-px coordinate quantization (debug)
  --no-color             plain text patch map
  --dump-features f.json write the raw [nTiles, 10] feature matrix (parity debug)
  --verify-warp fix.json check the warp pipeline against a cv2 fixture and exit
  --help, -h             show this help

Environment:
  CLR_TIMING=1           per-phase timings + device log on stderr
"""

func runHeadlessCLI() -> Never {
    var imagePath: String?
    var patch = 64
    var repeats = 1
    var jsonOut: String?
    var modelPath: String?
    var useCPU = false
    var exactWarp = false
    var noColor = false
    var dumpFeatures: String?

    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let a = it.next() {
        switch a {
        case "--headless":      break   // compat no-op: CLI mode is implied by any argument
        case "--help", "-h":
            FileHandle.standardOutput.write(Data((usageText + "\n").utf8))
            exit(0)
        case "--verify-warp":
            verifyWarpFixture(path: it.next() ?? "")   // prints diffs and exits
        case "--patch":         patch = Int(it.next() ?? "") ?? patch
        case "--repeat":        repeats = max(1, Int(it.next() ?? "") ?? 1)
        case "--json":          jsonOut = it.next()
        case "--model":         modelPath = it.next()
        case "--cpu":           useCPU = true
        case "--exact-warp":    exactWarp = true
        case "--no-color":      noColor = true
        case "--dump-features": dumpFeatures = it.next()
        default:
            if a.hasPrefix("--") {
                FileHandle.standardError.write(Data("Unknown flag: \(a)\n\(usageText)\n".utf8))
                exit(2)
            }
            imagePath = a
        }
    }
    guard let imagePath else {
        FileHandle.standardError.write(Data((usageText + "\n").utf8))
        exit(2)
    }

    let modelURL = modelPath.map { URL(fileURLWithPath: $0) } ?? InferenceConfig.defaultModelURL
    let dumpURL = dumpFeatures.map { URL(fileURLWithPath: $0) }
    let jsonOnly = jsonOut == "-"
    let useColor = !noColor && !jsonOnly && isatty(STDOUT_FILENO) == 1

    Task.detached {
        do {
            guard let nsImage = NSImage(contentsOf: URL(fileURLWithPath: imagePath)) else {
                FileHandle.standardError.write(Data("Cannot load image: \(imagePath)\n".utf8))
                exit(2)
            }
            var config = InferenceConfig(modelURL: modelURL, patchSize: patch)
            config.cv2QuantizedWarp = !exactWarp
            config.featureDumpURL = dumpURL

            var times = [Double]()
            var result: InferenceResult?
            let device = useCPU ? Device(.cpu) : Device(.gpu)
            try await Device.withDefaultDevice(device) {
                for _ in 0..<repeats {
                    let t0 = DispatchTime.now()
                    result = try await runInferenceFast(image: nsImage, config: config)
                    times.append(Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9)
                }
            }
            guard let r = result else { exit(1) }

            if let jsonOut {
                let data = try resultJSON(r, times: times, device: useCPU ? "cpu" : "gpu")
                if jsonOnly {
                    FileHandle.standardOutput.write(data)
                    FileHandle.standardOutput.write(Data("\n".utf8))
                    exit(0)
                }
                try data.write(to: URL(fileURLWithPath: jsonOut))
            }
            printPretty(r, imagePath: imagePath, modelURL: modelURL, times: times,
                        device: useCPU ? "cpu" : "gpu", color: useColor)
            exit(0)
        } catch {
            FileHandle.standardError.write(Data("Error: \(error)\n".utf8))
            exit(1)
        }
    }
    dispatchMain()
}

// MARK: - JSON output

private func resultJSON(_ r: InferenceResult, times: [Double], device: String) throws -> Data {
    var grid = [[Double]](), weights = [[Double]]()
    for i in 0..<r.gridRows {
        var row = [Double](), wrow = [Double]()
        for j in 0..<r.gridCols {
            let p = r.patches[i * r.gridCols + j]
            row.append(Double(p.pGenerated))
            wrow.append(Double(p.weight))
        }
        grid.append(row); weights.append(wrow)
    }
    let out: [String: Any] = [
        "device": device,
        "patch_size": r.patchSize,
        "grid": ["rows": r.gridRows, "cols": r.gridCols],
        "image": ["width": r.imageWidth, "height": r.imageHeight],
        "verdict": [
            "majority": ["prediction": r.majorityPrediction, "score": Double(r.majorityScore)],
            "weighted": ["prediction": r.weightedPrediction, "score": Double(r.weightedScore)],
        ],
        "patch_scores": grid,
        "weight_map": weights,
        "times_sec": times,
    ]
    return try JSONSerialization.data(withJSONObject: out, options: [.sortedKeys])
}

// MARK: - Pretty terminal output

/// Confidence label, mirroring Python's `_confidence_label`.
private func confidenceLabel(_ score: Float) -> String {
    let dev = abs(score - 0.5)
    if dev >= 0.3 { return "high" }
    if dev >= 0.15 { return "medium" }
    return "low"
}

/// Green (real) → yellow (uncertain) → red (generated).
private func heatColor(_ p: Float) -> (Int, Int, Int) {
    func lerp(_ a: (Int, Int, Int), _ b: (Int, Int, Int), _ u: Float) -> (Int, Int, Int) {
        (Int(Float(a.0) + (Float(b.0) - Float(a.0)) * u),
         Int(Float(a.1) + (Float(b.1) - Float(a.1)) * u),
         Int(Float(a.2) + (Float(b.2) - Float(a.2)) * u))
    }
    let green: (Int, Int, Int) = (35, 150, 65)
    let yellow: (Int, Int, Int) = (214, 175, 40)
    let red: (Int, Int, Int) = (205, 55, 48)
    let t = max(0, min(1, p))
    return t < 0.5 ? lerp(green, yellow, t * 2) : lerp(yellow, red, (t - 0.5) * 2)
}

/// Dark → bright blue for the weight map.
private func weightColor(_ frac: Float) -> (Int, Int, Int) {
    let t = max(0, min(1, frac))
    return (Int(25 + 35 * t), Int(35 + 90 * t), Int(55 + 180 * t))
}

/// Wrap `text` in 24-bit background color with auto black/white foreground.
private func cell(_ text: String, bg: (Int, Int, Int), color: Bool) -> String {
    guard color else { return text }
    let lum = 0.299 * Double(bg.0) + 0.587 * Double(bg.1) + 0.114 * Double(bg.2)
    let fg = lum > 145 ? "30" : "97"
    return "\u{1B}[48;2;\(bg.0);\(bg.1);\(bg.2)m\u{1B}[\(fg)m\(text)\u{1B}[0m"
}

private func verdictWord(_ word: String, color: Bool) -> String {
    guard color else { return word }
    let code = word == "generated" ? "31" : "32"
    return "\u{1B}[\(code);1m\(word)\u{1B}[0m"
}

private func printPretty(
    _ r: InferenceResult, imagePath: String, modelURL: URL,
    times: [Double], device: String, color: Bool
) {
    var out = ""
    let name = URL(fileURLWithPath: imagePath).lastPathComponent
    out += "Garde (CLR v1.8.2) — \(name)\n"
    out += "image \(r.imageWidth)×\(r.imageHeight) · grid \(r.gridRows)×\(r.gridCols) @ \(r.patchSize) px"
    out += " · device \(device) · model \(modelURL.lastPathComponent)\n\n"

    // Column header shared by both maps (6-char cells).
    var colHeader = "     "
    for j in 0..<r.gridCols { colHeader += String(format: "  %2d  ", j) }

    out += "P(generated) — green real · yellow uncertain · red generated\n"
    out += colHeader + "\n"
    for i in 0..<r.gridRows {
        var line = String(format: " %2d  ", i)
        for j in 0..<r.gridCols {
            let p = r.patches[i * r.gridCols + j].pGenerated
            line += cell(String(format: " %3.0f%% ", p * 100), bg: heatColor(p), color: color)
        }
        out += line + "\n"
    }

    out += "\nweight map — share of the weighted verdict (center × texture × confidence)\n"
    out += colHeader + "\n"
    let maxW = max(r.patches.map(\.weight).max() ?? 1, 1e-9)
    for i in 0..<r.gridRows {
        var line = String(format: " %2d  ", i)
        for j in 0..<r.gridCols {
            let w = r.patches[i * r.gridCols + j].weight
            line += cell(String(format: " %4.1f ", w * 100), bg: weightColor(w / maxW), color: color)
        }
        out += line + "\n"
    }

    out += "\nverdict  weighted  \(String(format: "%.4f", r.weightedScore))  "
    out += verdictWord(r.weightedPrediction, color: color)
    out += "  (\(confidenceLabel(r.weightedScore)) confidence)\n"
    out += "         majority  \(String(format: "%.4f", r.majorityScore))  "
    out += verdictWord(r.majorityPrediction, color: color)
    out += "  (\(confidenceLabel(r.majorityScore)) confidence — fraction of patches > 0.5)\n"

    let formatted = times.enumerated().map { i, t in
        String(format: "%.3fs%@", t, i == 0 && times.count > 1 ? " (warmup)" : "")
    }
    out += "time     " + formatted.joined(separator: " · ") + "\n"

    FileHandle.standardOutput.write(Data(out.utf8))
}

// MARK: - Warp fixture verification

/// Compare the Swift warp-residual pipeline against a cv2 fixture produced by
/// the Ra research repo's notebooks/clr/dump_reference.py: {"tile": [3072 HWC
/// floats], "R_low": [1024], "R_high": [1024]}. Prints max-abs diffs for the
/// quantized and exact-double coordinate paths and exits non-zero if the
/// quantized path is off.
private func verifyWarpFixture(path: String) -> Never {
    do {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let tile = (obj["tile"] as? [Any])?.map({ Float(($0 as? NSNumber)?.doubleValue ?? 0) }),
              let rLow = (obj["R_low"] as? [Any])?.map({ Float(($0 as? NSNumber)?.doubleValue ?? 0) }),
              let rHigh = (obj["R_high"] as? [Any])?.map({ Float(($0 as? NSNumber)?.doubleValue ?? 0) })
        else {
            FileHandle.standardError.write(Data("Bad fixture: \(path)\n".utf8))
            exit(2)
        }
        func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
            zip(a, b).map { abs($0 - $1) }.max() ?? .infinity
        }
        var report = [String: Double]()
        for quantize in [true, false] {
            let (low, high) = debugWarpResidualMaps(tileRGB: tile, quantizeLikeCV2: quantize)
            let key = quantize ? "cv2_quantized" : "exact_double"
            report["\(key)_low_maxdiff"] = Double(maxAbsDiff(low, rLow))
            report["\(key)_high_maxdiff"] = Double(maxAbsDiff(high, rHigh))
        }
        let out = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
        FileHandle.standardOutput.write(out)
        FileHandle.standardOutput.write(Data("\n".utf8))
        let pass = (report["cv2_quantized_low_maxdiff"] ?? 1) < 1e-3
            && (report["cv2_quantized_high_maxdiff"] ?? 1) < 1e-3
        exit(pass ? 0 : 1)
    } catch {
        FileHandle.standardError.write(Data("Error: \(error)\n".utf8))
        exit(2)
    }
}
