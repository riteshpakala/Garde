/// CLR v1.8.2 inference engine — Swift port of clr_inference.py.
///
/// Uses MLXAccelerate (Gaussian blur, Sobel, filter2D) from the Frigate module,
/// and FrigateBoost for XGBoost tree-ensemble scoring.
///
/// Feature order (must match training): FR, LV, CV, CM, CC, CR, SH, SV, SH_S, SV_S.

import Foundation
import AppKit
import os
import Frigate

// MARK: - Config

struct InferenceConfig: Sendable {
    var modelURL: URL
    var patchSize: Int   = 64
    var centerSharpness: Float = 8.0
    var texturePower: Float    = 3.0
    var textureBias: Float     = 0.2
    var saliencyCenter: Bool   = true
    var hotspotBlend: Float    = 0.35
    var hotspotThresh: Float   = 0.60
    var maxDim: Int            = 512
    /// Replicate cv2.warpPerspective's 1/32-pixel fixed-point coordinate quantization
    /// in the warp tap tables (bit-near parity with the Python/cv2 feature pipeline
    /// the model was trained on). False = exact double coordinates (≤ 1/64 px off
    /// cv2's sample points). Zero runtime cost either way.
    var cv2QuantizedWarp: Bool = true
    /// Debug: when set, the [nTiles, 10] feature matrix (row-major patch-then-tile
    /// order) is written here as JSON for parity comparison against Python.
    var featureDumpURL: URL?   = nil

    static let defaultModelURL: URL = {
        if let url = Bundle.module.url(forResource: "clr_v1.8", withExtension: "xgb.json") {
            return url
        }
        // SwiftPM strips dots from resource names; try the flat filename too.
        if let url = Bundle.module.url(forResource: "clr_v1.8.xgb", withExtension: "json") {
            return url
        }
        fatalError("clr_v1.8.xgb.json not found in bundle — ensure it is listed as a .copy resource in Package.swift")
    }()
}

// MARK: - Result

struct PatchInfo: Sendable {
    let row: Int, col: Int
    let pGenerated: Float
    let weight: Float
    let x: Int, y: Int, width: Int, height: Int
    var verdict: String { pGenerated > 0.5 ? "generated" : "real" }
}

struct InferenceResult: Sendable {
    let majorityPrediction: String
    let majorityScore: Float
    let weightedPrediction: String
    let weightedScore: Float
    let patches: [PatchInfo]
    let gridRows: Int, gridCols: Int
    let patchSize: Int
    let imageWidth: Int, imageHeight: Int
}

//// MARK: - v1.8 constants (must match training)
//
//private let TILE_SIZE  = 32
//private let N_TRIALS   = 16
//private let S_LOW: Float  = 0.025
//private let S_HIGH: Float = 0.05
//private let HP_KERNEL  = 7
//private let HP_SIGMA: Float = 1.0
//private let K_LV = 2    // local_var kernel size
//private let K_CV = 9    // circ_var kernel size
//private let STRIP_W    = 4
//private let SEAM_BINS  = 8
//
//// MARK: - Seeded RNG (replaces np.random.default_rng(42))
//
//private struct SeededRNG {
//    private var s0: UInt64
//    private var s1: UInt64
//
//    init(seed: UInt64 = 42) {
//        // Splitmix64 to initialise xorshift128+ state
//        func splitmix(_ x: inout UInt64) -> UInt64 {
//            x &+= 0x9e3779b97f4a7c15
//            var z = x
//            z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
//            z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
//            return z ^ (z >> 31)
//        }
//        var state = seed
//        s0 = splitmix(&state)
//        s1 = splitmix(&state)
//    }
//
//    mutating func nextUInt64() -> UInt64 {
//        var x = s0, y = s1
//        s0 = y
//        x ^= x << 23; x ^= x >> 18; x ^= y; x ^= y >> 5
//        s1 = x
//        return (s0 &+ s1)
//    }
//
//    // U(-1, 1) float
//    mutating func uniform() -> Float {
//        let bits = nextUInt64() >> 40          // 24 bits
//        let f = Float(bits) / Float(1 << 24)   // [0, 1)
//        return f * 2.0 - 1.0
//    }
//}
//
//// MARK: - Core feature extraction
//
///// Compute warp-noise residual map for a 32×32×3 float tile.
///// Mirrors `_warp_noise(tile, s)`.
//private func warpNoise(_ tile: MLXArray, s: Float) -> MLXArray {
//    let H = tile.shape[0], W = tile.shape[1]
//    var rng = SeededRNG(seed: 42)    // fresh seed each call — matches Python
//
//    var acc = MLXArray.zeros([H, W], type: Float.self)
//    let srcCorners: [[Float]] = [[0,0],[Float(W),0],[Float(W),Float(H)],[0,Float(H)]]
//
//    for _ in 0..<N_TRIALS {
//        var dst = [[Float]]()
//        for c in srcCorners {
//            let dx = rng.uniform() * s * Float(min(W, H))
//            let dy = rng.uniform() * s * Float(min(W, H))
//            dst.append([c[0]+dx, c[1]+dy])
//        }
//        let M    = getPerspectiveTransform(src: srcCorners, dst: dst)
//        let Minv = invertMatrix3x3(M)
//
//        let r    = perspectiveWarp(tile, forwardMatrix: M,    outputSize: (H, W))
//        let back = perspectiveWarp(r,    forwardMatrix: Minv, outputSize: (H, W))
//
//        // |tile - back|.mean(axis=2) — mean across colour channels
//        let diff = MLX.abs(tile.asType(.float32) - back.asType(.float32)).mean(axes: [2])
//        acc = acc + diff
//    }
//    let R = acc / Float(N_TRIALS)
//    // High-pass: R - GaussianBlur(R, 7, 1.0)
//    return R - gaussianBlur(R, kernelSize: HP_KERNEL, sigma: HP_SIGMA)
//}
//
///// Compute edge strip for seam features.  Returns 1-D Float array.
//private func edgeStrip(_ R: MLXArray, side: String) -> [Float] {
//    let H = R.shape[0], W = R.shape[1]
//    switch side {
//    case "top":    return R[0..<STRIP_W, 0...].mean(axes: [0]).asArray(Float.self)
//    case "bottom": return R[(H-STRIP_W)..., 0...].mean(axes: [0]).asArray(Float.self)
//    case "left":   return R[0..., 0..<STRIP_W].mean(axes: [1]).asArray(Float.self)
//    default:       return R[0..., (W-STRIP_W)...].mean(axes: [1]).asArray(Float.self)
//    }
//}
//
///// Pearson correlation coefficient between two flat arrays.
//private func pearsonR(_ a: MLXArray, _ b: MLXArray) -> Float {
//    let aF = a.flattened().asType(.float32)
//    let bF = b.flattened().asType(.float32)
//    let aVar: Float = aF.variance().item(Float.self)
//    let bVar: Float = bF.variance().item(Float.self)
//    let aStd = sqrtf(aVar), bStd = sqrtf(bVar)
//    guard aStd > 1e-8 && bStd > 1e-8 else { return 0 }
//    let aMean: Float = aF.mean().item(Float.self)
//    let bMean: Float = bF.mean().item(Float.self)
//    let num: Float = ((aF - aMean) * (bF - bMean)).mean().item(Float.self)
//    return num / (aStd * bStd)
//}
//
///// Synchronous score for use on GCD threads (no actor hop, no cooperative yield).
///// Identical logic to scorePatch but calls XGBoostTreeModel.predictBatch directly.
//private func scorePatchSync(_ patch: MLXArray, model: XGBoostTreeModel) -> Float {
//    let H = patch.shape[0], W = patch.shape[1]
//    let nr = H / TILE_SIZE, nc = W / TILE_SIZE
//    guard nr > 0, nc > 0 else { return 0 }
//
//    let kLV = MLXArray([Float](repeating: 1.0 / Float(K_LV * K_LV), count: K_LV * K_LV), [K_LV, K_LV])
//    let kCV = MLXArray([Float](repeating: 1.0 / Float(K_CV * K_CV), count: K_CV * K_CV), [K_CV, K_CV])
//
//    var maps   = [MLXArray?](repeating: nil, count: nr * nc)
//    var FR = [[Float]](repeating: [Float](repeating: 0, count: nc), count: nr)
//    var LV = [[Float]](repeating: [Float](repeating: 0, count: nc), count: nr)
//    var CV = [[Float]](repeating: [Float](repeating: 0, count: nc), count: nr)
//    var CM = [[Float]](repeating: [Float](repeating: 0, count: nc), count: nr)
//
//    for i in 0..<nr {
//        for j in 0..<nc {
//            let tile = patch[
//                (i * TILE_SIZE)..<((i+1) * TILE_SIZE),
//                (j * TILE_SIZE)..<((j+1) * TILE_SIZE),
//                0...
//            ]
//            let Rl = warpNoise(tile, s: S_LOW)
//            let Rh: Float = warpNoise(tile, s: S_HIGH).mean().item(Float.self)
//            maps[i * nc + j] = Rl
//
//            FR[i][j] = (Rh - Rl.mean().item(Float.self)) / (S_HIGH - S_LOW)
//
//            let lm  = filter2D(Rl, kernel: kLV)
//            let lm2 = filter2D(Rl * Rl, kernel: kLV)
//            LV[i][j] = MLX.maximum(lm2 - lm * lm, MLXArray(Float(0))).mean().item(Float.self)
//
//            let (gx, gy) = sobelGradients(Rl)
//            let ang  = MLX.atan2(gy, gx)
//            let Rf   = MLX.sqrt(filter2D(MLX.cos(ang), kernel: kCV).pow(2) + filter2D(MLX.sin(ang), kernel: kCV).pow(2))
//            let cvMap = MLXArray(Float(1.0)) - Rf
//            CV[i][j] = cvMap.mean().item(Float.self)
//            CM[i][j] = (MLX.abs(Rl) * (MLXArray(Float(1.0)) - cvMap)).mean().item(Float.self)
//        }
//    }
//
//    var CC = [[Float]](repeating: [Float](repeating: 0, count: nc), count: nr)
//    var CR = [[Float]](repeating: [Float](repeating: 0, count: nc), count: nr)
//    var pairs: [(Int,Int,Int,Int)] = []
//    let rm = nr-1, cm = nc-1
//    if rm > 0 && cm > 0 { pairs += [(0,0,rm,cm), (0,cm,rm,0)] }
//    if rm > 1 { pairs.append((0, nc/2, rm, nc/2)) }
//    if cm > 1 { pairs.append((nr/2, 0, nr/2, cm)) }
//    if pairs.isEmpty && nr * nc >= 2 { pairs = [(0,0,nr-1,nc-1)] }
//    for (i1,j1,i2,j2) in pairs {
//        guard let a = maps[i1*nc+j1], let b = maps[i2*nc+j2] else { continue }
//        let r = pearsonR(a, b)
//        if !r.isNaN { CC[i1][j1] = r; CC[i2][j2] = r }
//        let s1 = sqrtf(a.variance().item(Float.self))
//        let s2 = sqrtf(b.variance().item(Float.self))
//        CR[i1][j1] = min(s1,s2) / max(s1, s2, 1e-12)
//        CR[i2][j2] = CR[i1][j1]
//    }
//
//    var SH  = [[Float]](repeating: [Float](repeating: 0, count: nc), count: nr)
//    var SV  = [[Float]](repeating: [Float](repeating: 0, count: nc), count: nr)
//    var SHS = [[Float]](repeating: [Float](repeating: 0, count: nc), count: nr)
//    var SVS = [[Float]](repeating: [Float](repeating: 0, count: nc), count: nr)
//    for i in 0..<nr {
//        for j in 0..<nc {
//            guard let cur = maps[i*nc+j] else { continue }
//            if i+1 < nr, let below = maps[(i+1)*nc+j] {
//                let bot = edgeStrip(cur, side: "bottom"); let top = edgeStrip(below, side: "top")
//                SH[i][j]  = abs(bot.reduce(0,+)/Float(bot.count) - top.reduce(0,+)/Float(top.count))
//                SHS[i][j] = spectralDistance(bot, top, bins: SEAM_BINS)
//            }
//            if j+1 < nc, let right = maps[i*nc+j+1] {
//                let rgt = edgeStrip(cur, side: "right"); let lft = edgeStrip(right, side: "left")
//                SV[i][j]  = abs(rgt.reduce(0,+)/Float(rgt.count) - lft.reduce(0,+)/Float(lft.count))
//                SVS[i][j] = spectralDistance(rgt, lft, bins: SEAM_BINS)
//            }
//        }
//    }
//
//    var featureVectors = [[Float]]()
//    for i in 0..<nr {
//        for j in 0..<nc {
//            featureVectors.append([FR[i][j], LV[i][j], CV[i][j], CM[i][j],
//                                   CC[i][j], CR[i][j], SH[i][j], SV[i][j],
//                                   SHS[i][j], SVS[i][j]])
//        }
//    }
//    let probs = model.predictBatch(featureVectors)
//    return probs.reduce(0, +) / Float(probs.count)
//}
//
///// Score one outer patch.  Returns mean P(generated) over all inner tiles.
//private func scorePatch(_ patch: MLXArray, boost: FrigateBoost) async -> Float {
//    let H = patch.shape[0], W = patch.shape[1]
//    let nr = H / TILE_SIZE, nc = W / TILE_SIZE
//    guard nr > 0, nc > 0 else { return 0 }
//
//    let kLV = MLXArray(
//        [Float](repeating: 1.0 / Float(K_LV * K_LV), count: K_LV * K_LV),
//        [K_LV, K_LV]
//    )
//    let kCV = MLXArray(
//        [Float](repeating: 1.0 / Float(K_CV * K_CV), count: K_CV * K_CV),
//        [K_CV, K_CV]
//    )
//
//    // Pass 1: per-tile features (FR, LV, CV, CM) + store residual maps
//    var maps  = [[[Float]]?](repeating: nil, count: nr * nc)  // stored as flat arrays
//    var mapArr = [MLXArray?](repeating: nil, count: nr * nc)
//    var FR = [[Float]](repeating: [Float](repeating: 0, count: nc), count: nr)
//    var LV = [[Float]](repeating: [Float](repeating: 0, count: nc), count: nr)
//    var CV = [[Float]](repeating: [Float](repeating: 0, count: nc), count: nr)
//    var CM = [[Float]](repeating: [Float](repeating: 0, count: nc), count: nr)
//
//    for i in 0..<nr {
//        for j in 0..<nc {
//            let tile = patch[
//                (i * TILE_SIZE)..<((i+1) * TILE_SIZE),
//                (j * TILE_SIZE)..<((j+1) * TILE_SIZE),
//                0...
//            ]
//            let Rl = warpNoise(tile, s: S_LOW)
//            let Rh: Float = warpNoise(tile, s: S_HIGH).mean().item(Float.self)
//            mapArr[i * nc + j] = Rl
//
//            FR[i][j] = (Rh - Rl.mean().item(Float.self)) / (S_HIGH - S_LOW)
//
//            let lm  = filter2D(Rl, kernel: kLV)
//            let lm2 = filter2D(Rl * Rl, kernel: kLV)
//            let lvar = MLX.maximum(lm2 - lm * lm, MLXArray(Float(0)))
//            LV[i][j] = lvar.mean().item(Float.self)
//
//            let (gx, gy) = sobelGradients(Rl)
//            let ang = MLX.atan2(gy, gx)
//            let cosA = MLX.cos(ang), sinA = MLX.sin(ang)
//            let Rf = MLX.sqrt(
//                filter2D(cosA, kernel: kCV).pow(2) + filter2D(sinA, kernel: kCV).pow(2)
//            )
//            let cvMap = MLXArray(Float(1.0)) - Rf
//            CV[i][j] = cvMap.mean().item(Float.self)
//            CM[i][j] = (MLX.abs(Rl) * (MLXArray(Float(1.0)) - cvMap)).mean().item(Float.self)
//        }
//    }
//
//    // Pass 2: cross-region pairs (CC, CR)
//    var CC = [[Float]](repeating: [Float](repeating: 0, count: nc), count: nr)
//    var CR = [[Float]](repeating: [Float](repeating: 0, count: nc), count: nr)
//
//    var pairs: [(Int,Int,Int,Int)] = []
//    let rm = nr-1, cm = nc-1
//    if rm > 0 && cm > 0 {
//        pairs += [(0,0,rm,cm), (0,cm,rm,0)]
//    }
//    if rm > 1 { pairs.append((0, nc/2, rm, nc/2)) }
//    if cm > 1 { pairs.append((nr/2, 0, nr/2, cm)) }
//    if pairs.isEmpty && nr * nc >= 2 { pairs = [(0,0,nr-1,nc-1)] }
//
//    for (i1,j1,i2,j2) in pairs {
//        guard let a = mapArr[i1*nc+j1], let b = mapArr[i2*nc+j2] else { continue }
//        let r = pearsonR(a, b)
//        if !r.isNaN {
//            CC[i1][j1] = r; CC[i2][j2] = r
//        }
//        let s1 = sqrtf(a.variance().item(Float.self))
//        let s2 = sqrtf(b.variance().item(Float.self))
//        let ratio = min(s1,s2) / max(s1, s2, 1e-12)
//        CR[i1][j1] = ratio; CR[i2][j2] = ratio
//    }
//
//    // Pass 3: seam features (SH, SV, SH_S, SV_S)
//    var SH  = [[Float]](repeating: [Float](repeating: 0, count: nc), count: nr)
//    var SV  = [[Float]](repeating: [Float](repeating: 0, count: nc), count: nr)
//    var SHS = [[Float]](repeating: [Float](repeating: 0, count: nc), count: nr)
//    var SVS = [[Float]](repeating: [Float](repeating: 0, count: nc), count: nr)
//
//    for i in 0..<nr {
//        for j in 0..<nc {
//            guard let cur = mapArr[i*nc+j] else { continue }
//            if i+1 < nr, let below = mapArr[(i+1)*nc+j] {
//                let bot = edgeStrip(cur,   side: "bottom")
//                let top = edgeStrip(below, side: "top")
//                SH[i][j]  = abs(bot.reduce(0,+)/Float(bot.count) - top.reduce(0,+)/Float(top.count))
//                SHS[i][j] = spectralDistance(bot, top, bins: SEAM_BINS)
//            }
//            if j+1 < nc, let right = mapArr[i*nc+j+1] {
//                let rgt = edgeStrip(cur,   side: "right")
//                let lft = edgeStrip(right, side: "left")
//                SV[i][j]  = abs(rgt.reduce(0,+)/Float(rgt.count) - lft.reduce(0,+)/Float(lft.count))
//                SVS[i][j] = spectralDistance(rgt, lft, bins: SEAM_BINS)
//            }
//        }
//    }
//
//    // Batch all feature vectors and run XGBoost
//    var featureVectors = [[Float]]()
//    for i in 0..<nr {
//        for j in 0..<nc {
//            featureVectors.append([FR[i][j], LV[i][j], CV[i][j], CM[i][j],
//                                   CC[i][j], CR[i][j], SH[i][j], SV[i][j],
//                                   SHS[i][j], SVS[i][j]])
//        }
//    }
//    let probs = await boost.predict(features: featureVectors)
//    return probs.reduce(0, +) / Float(probs.count)
//}
//
//// MARK: - Main inference entry point
//
///// Run CLR v1.8 inference on an NSImage.
/////
///// - Parameters:
/////   - image: Source image (any size — will be downscaled to `config.maxDim` if needed).
/////   - config: Inference configuration.
/////   - progress: Called after each outer patch with `(completed, total, lastScore)`.
///// - Returns: `InferenceResult` ready for display.
//func runInference(
//    image nsImage: NSImage,
//    config: InferenceConfig,
//    progress: @Sendable @escaping (Int, Int, Float) -> Void = { _,_,_ in }
//) async throws -> InferenceResult {
//    // Load model once — XGBoostTreeModel is Sendable and synchronous; no actor needed.
//    let syncModel: XGBoostTreeModel
//    do { syncModel = try XGBoostTreeModel(url: config.modelURL) } catch {
//        throw NSError(domain: "CLR", code: 1,
//                      userInfo: [NSLocalizedDescriptionKey: "Cannot load model: \(config.modelURL.path)"])
//    }
//
//    // NSImage → pixel buffer (float32, 0–255, RGB)
//    let (imgArr, imgW, imgH) = try nsImageToMLXArray(nsImage, maxDim: config.maxDim)
//
//    let ps = config.patchSize
//    let Pr = imgH / ps, Pc = imgW / ps
//    guard Pr > 0, Pc > 0 else {
//        throw NSError(domain: "CLR", code: 2,
//                      userInfo: [NSLocalizedDescriptionKey:
//                        "Image \(imgW)×\(imgH) too small for patchSize=\(ps)"])
//    }
//
//    // Materialise full image as flat [Float] — Sendable, safe to read from any GCD thread.
//    let imgData: [Float] = imgArr.asArray(Float.self)
//
//    var patchGrid  = [[Float]](repeating: [Float](repeating: 0, count: Pc), count: Pr)
//    var texWeights = [[Float]](repeating: [Float](repeating: 0, count: Pc), count: Pr)
//
//    // Parallel patch scoring on GCD thread pool.
//    // withCheckedContinuation suspends the Swift cooperative task (releases its thread),
//    // then concurrentPerform runs blocking MLX evals on GCD threads — no pool starvation.
//    // OSAllocatedUnfairLock provides a Sendable atomic counter for live progress.
//    let (flatScores, flatTex): ([Float], [Float]) = await withCheckedContinuation { continuation in
//        DispatchQueue.global(qos: .userInitiated).async {
//            var scores  = [Float](repeating: 0, count: Pr * Pc)
//            var tex     = [Float](repeating: 0, count: Pr * Pc)
//            let counter = OSAllocatedUnfairLock(initialState: 0)
//            scores.withUnsafeMutableBufferPointer { sBuf in
//                tex.withUnsafeMutableBufferPointer { tBuf in
//                    DispatchQueue.concurrentPerform(iterations: Pr * Pc) { idx in
//                        let I = idx / Pc, J = idx % Pc
//                        var patchData = [Float](repeating: 0, count: ps * ps * 3)
//                        for row in 0..<ps {
//                            let src = ((I * ps + row) * imgW + J * ps) * 3
//                            let dst = row * ps * 3
//                            for k in 0..<(ps * 3) { patchData[dst + k] = imgData[src + k] }
//                        }
//                        let patch = MLXArray(patchData, [ps, ps, 3])
//                        let score = scorePatchSync(patch, model: syncModel)
//                        let gray = rgbToGray(patch)
//                        let (gx, gy) = sobelGradients(gray)
//                        let tw = MLX.sqrt(gx*gx + gy*gy).mean().item(Float.self)
//                        sBuf[idx] = score
//                        tBuf[idx] = tw
//                        let c: Int = counter.withLock { n in n += 1; return n }
//                        progress(c, Pr * Pc, score)
//                    }
//                }
//            }
//            continuation.resume(returning: (scores, tex))
//        }
//    }
//
//    for I in 0..<Pr {
//        for J in 0..<Pc {
//            patchGrid[I][J]  = flatScores[I * Pc + J]
//            texWeights[I][J] = flatTex[I * Pc + J]
//        }
//    }
//
//    let weightMap = buildWeightMap(
//        patchGrid: patchGrid, texWeights: texWeights,
//        Pr: Pr, Pc: Pc, config: config
//    )
//    let weightedScore = weightedVerdict(patchGrid: patchGrid, weightMap: weightMap, config: config)
//    let majorityScore = patchGrid.flatMap { $0 }.filter { $0 > 0.5 }.count.f / Float(Pr * Pc)
//
//    var patches = [PatchInfo]()
//    for I in 0..<Pr {
//        for J in 0..<Pc {
//            patches.append(PatchInfo(
//                row: I, col: J,
//                pGenerated: patchGrid[I][J],
//                weight: weightMap[I][J],
//                x: J*ps, y: I*ps, width: ps, height: ps
//            ))
//        }
//    }
//
//    return InferenceResult(
//        majorityPrediction: majorityScore > 0.5 ? "generated" : "real",
//        majorityScore: majorityScore,
//        weightedPrediction: weightedScore > 0.5 ? "generated" : "real",
//        weightedScore: weightedScore,
//        patches: patches,
//        gridRows: Pr, gridCols: Pc,
//        patchSize: ps,
//        imageWidth: imgW, imageHeight: imgH
//    )
//}
//
//// MARK: - Weight map (mirrors _build_weight_map)
//
//private func buildWeightMap(
//    patchGrid: [[Float]], texWeights: [[Float]],
//    Pr: Int, Pc: Int, config: InferenceConfig
//) -> [[Float]] {
//    // Normalise texture
//    let flat = texWeights.flatMap { $0 }
//    let tMin = flat.min() ?? 0, tMax = flat.max() ?? 1
//    var tw = [[Float]](repeating: [Float](repeating: 0, count: Pc), count: Pr)
//    for i in 0..<Pr {
//        for j in 0..<Pc {
//            tw[i][j] = (texWeights[i][j] - tMin) / (tMax - tMin + 1e-8)
//        }
//    }
//
//    // Saliency centre
//    var cy: Float, cx: Float
//    if config.saliencyCenter {
//        var sumW: Float = 0, sumY: Float = 0, sumX: Float = 0
//        for i in 0..<Pr {
//            for j in 0..<Pc {
//                sumY += Float(i) * tw[i][j]; sumX += Float(j) * tw[i][j]; sumW += tw[i][j]
//            }
//        }
//        cy = sumY / (sumW + 1e-8); cx = sumX / (sumW + 1e-8)
//    } else {
//        cy = Float(Pr) / 2; cx = Float(Pc) / 2
//    }
//
//    // Gaussian centre weight
//    let sigma = Float(max(Pr, Pc)) / config.centerSharpness
//    var combined = [[Float]](repeating: [Float](repeating: 0, count: Pc), count: Pr)
//    var total: Float = 0
//    for i in 0..<Pr {
//        for j in 0..<Pc {
//            let dx = Float(j) - cx, dy = Float(i) - cy
//            let cw = Foundation.exp(-(dx*dx + dy*dy) / (2*sigma*sigma))
//            let texW = config.textureBias + (1 - config.textureBias) * pow(tw[i][j], config.texturePower)
//            let conf = abs(patchGrid[i][j] - 0.5)
//            let boost = 1.0 + 2.0 * conf
//            combined[i][j] = cw * texW * boost
//            total += combined[i][j]
//        }
//    }
//    // Normalise
//    for i in 0..<Pr { for j in 0..<Pc { combined[i][j] /= max(total, 1e-12) } }
//    return combined
//}
//
//// MARK: - Weighted verdict (mirrors _weighted_verdict)
//
//private func weightedVerdict(patchGrid: [[Float]], weightMap: [[Float]], config: InferenceConfig) -> Float {
//    var score: Float = 0
//    let Pr = patchGrid.count, Pc = patchGrid[0].count
//    for i in 0..<Pr { for j in 0..<Pc { score += patchGrid[i][j] * weightMap[i][j] } }
//    // Hotspot correction
//    var hotspots = [Float]()
//    for i in 0..<Pr { for j in 0..<Pc { if patchGrid[i][j] > config.hotspotThresh { hotspots.append(patchGrid[i][j]) } } }
//    if !hotspots.isEmpty {
//        let hMean = hotspots.reduce(0,+) / Float(hotspots.count)
//        score = (1 - config.hotspotBlend) * score + config.hotspotBlend * hMean
//    }
//    return score
//}
//
//// MARK: - Image loading helpers
//
//private func nsImageToMLXArray(_ nsImage: NSImage, maxDim: Int) throws -> (MLXArray, Int, Int) {
//    guard let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
//        throw NSError(domain: "CLR", code: 3, userInfo: [NSLocalizedDescriptionKey: "Cannot get CGImage"])
//    }
//    var w = cgImage.width, h = cgImage.height
//    if maxDim > 0 && max(w, h) > maxDim {
//        let scale = Double(maxDim) / Double(max(w, h))
//        w = Int(Double(w) * scale); h = Int(Double(h) * scale)
//    }
//    let bytesPerPixel = 4
//    var pixelData = [UInt8](repeating: 0, count: w * h * bytesPerPixel)
//    let colorSpace = CGColorSpaceCreateDeviceRGB()
//    guard let ctx = CGContext(
//        data: &pixelData, width: w, height: h,
//        bitsPerComponent: 8, bytesPerRow: w * bytesPerPixel,
//        space: colorSpace,
//        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
//    ) else {
//        throw NSError(domain: "CLR", code: 4, userInfo: [NSLocalizedDescriptionKey: "Cannot create CGContext"])
//    }
//    ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
//
//    // Convert to float32 (0–255) RGB, shape (H, W, 3)
//    var floats = [Float](repeating: 0, count: h * w * 3)
//    for i in 0..<h {
//        for j in 0..<w {
//            let base = (i * w + j) * bytesPerPixel
//            floats[(i * w + j) * 3 + 0] = Float(pixelData[base + 0])
//            floats[(i * w + j) * 3 + 1] = Float(pixelData[base + 1])
//            floats[(i * w + j) * 3 + 2] = Float(pixelData[base + 2])
//        }
//    }
//    return (MLXArray(floats, [h, w, 3]), w, h)
//}
//
//private func rgbToGray(_ img: MLXArray) -> MLXArray {
//    // Luminance weights: 0.299R + 0.587G + 0.114B
//    let r = img[0..., 0..., 0..<1].squeezed(axis: -1)
//    let g = img[0..., 0..., 1..<2].squeezed(axis: -1)
//    let b = img[0..., 0..., 2..<3].squeezed(axis: -1)
//    return r * Float(0.299) + g * Float(0.587) + b * Float(0.114)
//}
//
//// MARK: - Swift numeric helpers
//
//private extension Int {
//    var f: Float { Float(self) }
//}
//
//private func max(_ a: Float, _ b: Float, _ c: Float) -> Float { Swift.max(a, Swift.max(b, c)) }
