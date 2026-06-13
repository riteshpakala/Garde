//
//  CLRInference.swift
//  CLR
//
//  Created by Ritesh Pakala Rao on 6/8/26.
//

import Foundation
import AppKit
import Frigate

// MARK: - v1.8 constants (must match training)

private let TILE_SIZE  = 32
private let N_TRIALS   = 16
private let HP_KERNEL  = 7
private let HP_SIGMA: Float = 1.0
private let K_LV       = 2      // local_var kernel size
private let K_CV       = 9      // circ_var kernel size
private let STRIP_W    = 4
private let SEAM_BINS  = 8

/// Tiles per MLX.eval group. 256 tiles ≈ 150 MB of transient gather buffers; with
/// maxDim=512 the whole image fits in one group (ps=64 → 4 tiles/patch × 64 patches).
private let EVAL_TILE_BUDGET = 256

// Float scales used only as the FR normaliser (matches Python self._S_HIGH/_S_LOW).
private let S_LOW:  Float = 0.025
private let S_HIGH: Float = 0.05

// Double scales used to build the corner displacements, so the multiply
// `WARP_DISP * s * 32` happens in float64 and is then cast to Float — exactly
// matching Python's `(rng.uniform(...) * s * min(w,h)).astype(np.float32)`.
private let S_LOW_D:  Double = 0.025
private let S_HIGH_D: Double = 0.05

// MARK: - Deterministic warp displacements
// Exact output of numpy `np.random.default_rng(42).uniform(-1, 1, (16, 4, 2))`.
// Verified equal to 16 sequential `uniform(-1,1,(4,2))` draws (numpy fills C-order),
// which is what `_warp_noise` does. Index order: [trial][corner][axis(x,y)].

private let WARP_DISP: [[[Double]]] = [
    [[0.5479120971119267, -0.12224312049589536], [0.7171958398227649, 0.3947360581187278], [-0.8116453042247009, 0.9512447032735118], [0.5222794039807059, 0.5721286105539076]],
    [[-0.7437727346489083, -0.09922812420886573], [-0.25840395153483753, 0.8535299776972036], [0.2877302401613291, 0.64552322654166], [-0.11317160234533774, -0.5455225564304462]],
    [[0.1091695740316696, -0.8723654877916494], [0.6552623439851641, 0.2633287982441297], [0.5161754801707477, -0.2909480637402633], [0.9413960487898065, 0.7862422426443954]],
    [[0.5567669941475237, -0.6107225842960649], [-0.06655799254593164, -0.9123924684255424], [-0.6914210158649043, 0.36609790648490925], [0.48952431181563427, 0.93501946486842]],
    [[-0.3483492837236961, -0.25908058793026223], [-0.06088837744838416, -0.6210572818314286], [-0.7401569893290567, -0.04859014754813251], [-0.5461813018982318, 0.3396279893650207]],
    [[-0.12569616225533853, 0.6653563921156749], [0.40053020400449824, -0.37526671723591787], [0.6645196027904021, 0.6095287149936037], [-0.22504324193965108, -0.4233437921395118]],
    [[0.364991007949951, -0.7204950327813804], [-0.6001835950497834, -0.985275460497989], [0.5738487550042768, 0.32970171318406427], [0.4103307572526702, 0.5614580620439358]],
    [[-0.08216844892332009, 0.13748239190578748], [-0.7204060037446851, -0.7709398529280531], [0.3368059235809433, -0.057807587713734954], [0.13047221296237765, 0.5299977148320512]],
    [[0.2694366400011816, 0.10715880131599165], [0.11841432149082709, -0.3920998038747756], [-0.9383643308641212, -0.12656522153527527], [-0.5708306543609416, -0.1829427125507277]],
    [[0.7068061465363322, -0.5321210282693185], [-0.883394516621868, -0.4372322159560069], [-0.41281248446663277, 0.3238330294537901], [0.11406430468255668, 0.567796418212827]],
    [[0.3286270806547751, -0.18722627711985895], [0.6280407693320693, -0.6660541601845922], [-0.954575853732279, -0.8199042784487165], [0.4447187011929006, -0.07624553949722523]],
    [[-0.6774564419327964, 0.0020895502067270755], [-0.6953757945736632, 0.39264075015547206], [-0.1076874488519386, -0.23795754780703504], [-0.39697582170424695, 0.2605651862377769]],
    [[-0.27637477889321915, -0.824700161367798], [-0.7639881957589694, 0.9237953290990291], [0.8171613814152141, 0.3994142676214991], [-0.46826007708096085, 0.9383527546954478]],
    [[0.5575018079315892, 0.4337803783179912], [-0.10127699571242266, -0.45551687630968196], [-0.8072180756930014, 0.8052047930876833], [-0.08844742033277786, -0.5952732704095394]],
    [[-0.388086751698695, 0.15843913788379194], [-0.6464544341215366, 0.713228568184751], [0.5170390596704202, 0.4389259119018736], [-0.13581392044979257, 0.2546176814048864]],
    [[0.16819593782547115, 0.29969320310964], [-0.8311113577202218, -0.16838519565878074], [-0.916771652276215, -0.012018361510962139], [-0.34027757533442937, -0.7109516222679062]],
]

// MARK: - Warp tap tables

/// Build the (forward, inverse) bicubic tap tables for the 16 trials at scale `s`.
///
/// Python per trial: M = getPerspectiveTransform(src, src+d); fwd warp passes M and
/// cv2 inverts it internally; back warp passes np.linalg.inv(M) and cv2 inverts THAT
/// internally. So the dst→src maps are inv(M) and inv(inv(M)) — both replicated in
/// float64, including the double inversion round-trip (do not shortcut to M).
private func buildWarpTables(scale s: Double, quantizeLikeCV2: Bool)
    -> (fwd: WarpTapTable, back: WarpTapTable) {

    let t = Float(TILE_SIZE)
    let srcCorners: [[Float]] = [[0, 0], [t, 0], [t, t], [0, t]]
    var fwdMats = [[Double]]()
    var backMats = [[Double]]()
    for trial in 0..<N_TRIALS {
        let dst: [[Float]] = (0..<4).map { c in
            // float64 multiply, then cast to Float — matches numpy `.astype(np.float32)`.
            let dx = Float(WARP_DISP[trial][c][0] * s * Double(TILE_SIZE))
            let dy = Float(WARP_DISP[trial][c][1] * s * Double(TILE_SIZE))
            return [srcCorners[c][0] + dx, srcCorners[c][1] + dy]
        }
        let M = getPerspectiveTransformD(src: srcCorners, dst: dst)
        fwdMats.append(invertMatrix3x3D(M))
        backMats.append(invertMatrix3x3D(invertMatrix3x3D(M)))
    }
    let fwd = buildBicubicWarpTable(
        dstToSrc: fwdMats, height: TILE_SIZE, width: TILE_SIZE,
        sourceIsTrialStacked: false, quantizeLikeCV2: quantizeLikeCV2)
    let back = buildBicubicWarpTable(
        dstToSrc: backMats, height: TILE_SIZE, width: TILE_SIZE,
        sourceIsTrialStacked: true, quantizeLikeCV2: quantizeLikeCV2)
    return (fwd: fwd, back: back)
}

/// Warp-noise residual maps for a whole tile batch, high-pass filtered.
/// Mirrors `_warp_noise` for every tile at once: forward warp all 16 trials, warp
/// back, |tile − back| meaned over channels then trials, minus a Gaussian blur.
/// `tiles` is [N, 1024, 3]; returns [N, 32, 32].
private func warpResidualBatch(
    _ tiles: MLXArray, fwd: WarpTapTable, back: WarpTapTable
) -> MLXArray {
    let N = tiles.shape[0]
    let warped = applyWarpTable(tiles, fwd)                    // [N, 16·1024, 3]
    let returned = applyWarpTable(warped, back)                // [N, 16·1024, 3]
    let diff = MLX.abs(
        tiles.reshaped([N, 1, TILE_SIZE * TILE_SIZE, 3])
        - returned.reshaped([N, N_TRIALS, TILE_SIZE * TILE_SIZE, 3]))
    let R = diff.mean(axes: [3]).mean(axes: [1])               // channels, then trials
        .reshaped([N, TILE_SIZE, TILE_SIZE])
    return R - batchedGaussianBlur(R, kernelSize: HP_KERNEL, sigma: HP_SIGMA)
}

/// Debug hook for HeadlessCLI `--verify-warp`: run the full warp-residual pipeline
/// on one 32×32×3 tile (row-major HWC floats) and return the high-passed residual
/// maps for both scales, for comparison against a cv2-generated fixture.
func debugWarpResidualMaps(tileRGB: [Float], quantizeLikeCV2: Bool) -> (low: [Float], high: [Float]) {
    let tiles = MLXArray(tileRGB, [1, TILE_SIZE * TILE_SIZE, 3])
    let (lowFwd, lowBack)   = buildWarpTables(scale: S_LOW_D,  quantizeLikeCV2: quantizeLikeCV2)
    let (highFwd, highBack) = buildWarpTables(scale: S_HIGH_D, quantizeLikeCV2: quantizeLikeCV2)
    let Rl = warpResidualBatch(tiles, fwd: lowFwd, back: lowBack)
    let Rh = warpResidualBatch(tiles, fwd: highFwd, back: highBack)
    MLX.eval(Rl, Rh)
    return (low: Rl.asArray(Float.self), high: Rh.asArray(Float.self))
}

// MARK: - Geometry

/// Geometric tile pairs within a patch. Mirrors `_geo_pairs` (+ Python's 2-tile
/// fallback). For the square grids produced here (nr == nc ≥ 2) the pair members
/// never overlap, so the one-hot scatter in `scoreGroup` is exact.
private func geoPairs(_ nr: Int, _ nc: Int) -> [(Int, Int, Int, Int)] {
    var pairs = [(Int, Int, Int, Int)]()
    let rm = nr - 1, cm = nc - 1
    if rm > 0 && cm > 0 { pairs += [(0, 0, rm, cm), (0, cm, rm, 0)] }
    if rm > 1 { pairs.append((0, nc / 2, rm, nc / 2)) }
    if cm > 1 { pairs.append((nr / 2, 0, nr / 2, cm)) }
    if pairs.isEmpty && nr * nc >= 2 { pairs = [(0, 0, nr - 1, nc - 1)] }
    return pairs
}

/// cv2 cvtColor(RGB2GRAY) on uint8 data: (R·4899 + G·9617 + B·1868 + 8192) >> 14.
/// Inputs hold exact uint8 values in float32, every intermediate stays < 2^24, so
/// this floor-division reproduces cv2's rounded uint8 gray bit-exactly.
private func cv2GrayU8(_ img: MLXArray) -> MLXArray {
    let r = img[.ellipsis, 0..<1].squeezed(axis: -1)
    let g = img[.ellipsis, 1..<2].squeezed(axis: -1)
    let b = img[.ellipsis, 2..<3].squeezed(axis: -1)
    return MLX.floor((r * Float(4899) + g * Float(9617) + b * Float(1868) + Float(8192)) / Float(16384))
}

// MARK: - Score a group of patches (single GPU sync per group)

/// Constant arrays shared by every group, built once per run.
private struct GroupConstants {
    let lowFwd: WarpTapTable, lowBack: WarpTapTable
    let highFwd: WarpTapTable, highBack: WarpTapTable
    let kLV: MLXArray, kCV: MLXArray
    let pairA: MLXArray, pairB: MLXArray     // [P] int32 flat tile indices
    let pairMask: MLXArray                   // [P, n] one-hot scatter (both members)
    let nr: Int, nc: Int
    var n: Int { nr * nc }

    init(nr: Int, nc: Int, quantizeLikeCV2: Bool) {
        self.nr = nr; self.nc = nc
        (lowFwd, lowBack)   = buildWarpTables(scale: S_LOW_D,  quantizeLikeCV2: quantizeLikeCV2)
        (highFwd, highBack) = buildWarpTables(scale: S_HIGH_D, quantizeLikeCV2: quantizeLikeCV2)
        kLV = MLXArray([Float](repeating: 1.0 / Float(K_LV * K_LV), count: K_LV * K_LV), [K_LV, K_LV])
        kCV = MLXArray([Float](repeating: 1.0 / Float(K_CV * K_CV), count: K_CV * K_CV), [K_CV, K_CV])

        let pairs = geoPairs(nr, nc)
        let n = nr * nc
        pairA = MLXArray(pairs.map { Int32($0.0 * nc + $0.1) })
        pairB = MLXArray(pairs.map { Int32($0.2 * nc + $0.3) })
        var mask = [Float](repeating: 0, count: pairs.count * n)
        for (p, (i1, j1, i2, j2)) in pairs.enumerated() {
            mask[p * n + i1 * nc + j1] = 1
            mask[p * n + i2 * nc + j2] = 1
        }
        pairMask = MLXArray(mask, [pairs.count, n])
        MLX.eval(kLV, kCV, pairA, pairB, pairMask)
    }
}

/// Score `G` patches in one lazy graph and one `MLX.eval`.
/// `patches` is [G, ps, ps, 3]; returns per-patch scores, texture weights, and the
/// raw [G·n, 10] feature rows (for the optional debug dump).
private func scoreGroup(
    _ patches: MLXArray,
    c: GroupConstants,
    model: XGBoostTreeModel
) -> (scores: [Float], tex: [Float], features: [[Float]]) {

    let G = patches.shape[0]
    let nr = c.nr, nc = c.nc, n = c.n
    let N = G * n
    let ts = TILE_SIZE

    // [G, ps, ps, 3] → [G·n, 1024, 3] tile stack (row-major i,j — matches Python).
    let tiles = patches
        .reshaped([G, nr, ts, nc, ts, 3])
        .transposed(0, 1, 3, 2, 4, 5)
        .reshaped([N, ts * ts, 3])

    // Warp residuals for both scales.
    let Rl = warpResidualBatch(tiles, fwd: c.lowFwd, back: c.lowBack)    // [N, 32, 32]
    let Rh = warpResidualBatch(tiles, fwd: c.highFwd, back: c.highBack)

    // Per-tile features → [N].
    let FR = (Rh.mean(axes: [1, 2]) - Rl.mean(axes: [1, 2])) / (S_HIGH - S_LOW)

    let lm  = batchedFilter2D(Rl, kernel: c.kLV)
    let lm2 = batchedFilter2D(Rl * Rl, kernel: c.kLV)
    let LV  = MLX.maximum(lm2 - lm * lm, MLXArray(Float(0))).mean(axes: [1, 2])

    let (gx, gy) = batchedSobelGradients(Rl)
    let ang = MLX.atan2(gy, gx)
    let Rf = MLX.sqrt(batchedFilter2D(MLX.cos(ang), kernel: c.kCV).pow(2)
                      + batchedFilter2D(MLX.sin(ang), kernel: c.kCV).pow(2))
    let cvMap = MLXArray(Float(1.0)) - Rf
    let CV = cvMap.mean(axes: [1, 2])
    // Python: (|R_l| * (1 - cv_map)).mean() — keep the double rounding, not just Rf.
    let CM = (MLX.abs(Rl) * (MLXArray(Float(1.0)) - cvMap)).mean(axes: [1, 2])

    // Cross-region pairs (CC, CR): batched Pearson + std ratio, scattered to both
    // members via the one-hot mask matmul.
    let maps = Rl.reshaped([G, n, ts * ts])
    let a = maps.take(c.pairA, axis: 1)        // [G, P, 1024]
    let b = maps.take(c.pairB, axis: 1)
    let acm = a - a.mean(axes: [2], keepDims: true)
    let bcm = b - b.mean(axes: [2], keepDims: true)
    let num = (acm * bcm).mean(axes: [2])
    let den = MLX.sqrt((acm * acm).mean(axes: [2]) * (bcm * bcm).mean(axes: [2]) + Float(1e-12))
    let CC = MLX.matmul(num / den, c.pairMask)                       // [G, n]
    let s1 = MLX.sqrt(a.variance(axes: [2]))
    let s2 = MLX.sqrt(b.variance(axes: [2]))
    let ratio = MLX.minimum(s1, s2) / MLX.maximum(MLX.maximum(s1, s2), MLXArray(Float(1e-12)))
    let CR = MLX.matmul(ratio, c.pairMask)                           // [G, n]

    // Seam features. Edge strips of every residual map → [N, 32]; their means and
    // 8-bin rfft magnitudes are differenced between vertically / horizontally
    // adjacent tiles, with zero rows/cols for the last row/col (Python leaves 0).
    let topS = Rl[0..., 0..<STRIP_W, 0...].mean(axes: [1])
    let botS = Rl[0..., (ts - STRIP_W)..., 0...].mean(axes: [1])
    let lftS = Rl[0..., 0..., 0..<STRIP_W].mean(axes: [2])
    let rgtS = Rl[0..., 0..., (ts - STRIP_W)...].mean(axes: [2])

    func seamDiffs(_ near: MLXArray, _ far: MLXArray, axis: Int)
        -> (mean: MLXArray, spec: MLXArray) {
        // near/far: [N, 32] strips; "near" of tile k pairs with "far" of the next
        // tile along `axis` (1 = below, 2 = right) in the [G, nr, nc] grid.
        let nearM = near.mean(axes: [1]).reshaped([G, nr, nc])
        let farM  = far.mean(axes: [1]).reshaped([G, nr, nc])
        let nearF = MLX.abs(MLXFFT.rfft(near, axis: 1))[0..., 0..<SEAM_BINS]
            .reshaped([G, nr, nc, SEAM_BINS])
        let farF  = MLX.abs(MLXFFT.rfft(far, axis: 1))[0..., 0..<SEAM_BINS]
            .reshaped([G, nr, nc, SEAM_BINS])
        if axis == 1 {
            let zero = MLXArray.zeros([G, 1, nc], type: Float.self)
            let m = MLX.concatenated(
                [MLX.abs(nearM[0..., 0..<(nr - 1), 0...] - farM[0..., 1..., 0...]), zero], axis: 1)
            let s = MLX.concatenated(
                [MLX.abs(nearF[0..., 0..<(nr - 1), 0..., 0...] - farF[0..., 1..., 0..., 0...])
                    .mean(axes: [3]), zero], axis: 1)
            return (m, s)
        } else {
            let zero = MLXArray.zeros([G, nr, 1], type: Float.self)
            let m = MLX.concatenated(
                [MLX.abs(nearM[0..., 0..., 0..<(nc - 1)] - farM[0..., 0..., 1...]), zero], axis: 2)
            let s = MLX.concatenated(
                [MLX.abs(nearF[0..., 0..., 0..<(nc - 1), 0...] - farF[0..., 0..., 1..., 0...])
                    .mean(axes: [3]), zero], axis: 2)
            return (m, s)
        }
    }
    let (SH, SHS) = seamDiffs(botS, topS, axis: 1)
    let (SV, SVS) = seamDiffs(rgtS, lftS, axis: 2)

    // Texture weight: Sobel magnitude mean over cv2's uint8 gray.
    let gray = cv2GrayU8(patches)                                    // [G, ps, ps]
    let (tgx, tgy) = batchedSobelGradients(gray)
    let tex = MLX.sqrt(tgx * tgx + tgy * tgy).mean(axes: [1, 2])     // [G]

    // [G, n, 10] in training feature order.
    let featAll = MLX.stacked(
        [FR.reshaped([G, n]), LV.reshaped([G, n]), CV.reshaped([G, n]), CM.reshaped([G, n]),
         CC, CR,
         SH.reshaped([G, n]), SV.reshaped([G, n]),
         SHS.reshaped([G, n]), SVS.reshaped([G, n])],
        axis: 2)

    // ── The single GPU sync for the whole group ──
    MLX.eval(featAll, tex)
    let flat = featAll.asArray(Float.self)                           // G·n·10
    let texOut = tex.asArray(Float.self)

    var rows = [[Float]]()
    rows.reserveCapacity(N)
    for t in 0..<N { rows.append(Array(flat[(t * 10)..<(t * 10 + 10)])) }
    let probs = model.predictBatch(rows)

    var scores = [Float](repeating: 0, count: G)
    for g in 0..<G {
        var s: Float = 0
        for t in 0..<n { s += probs[g * n + t] }
        scores[g] = s / Float(n)
    }
    return (scores, texOut, rows)
}

// MARK: - Main inference entry point

/// Batched, fidelity-corrected CLR v1.8 inference. Companion to `runInference(...)`.
func runInferenceFast(
    image nsImage: NSImage,
    config: InferenceConfig,
    progress: @Sendable @escaping (Int, Int, Float) -> Void = { _, _, _ in }
) async throws -> InferenceResult {

    let timing = ProcessInfo.processInfo.environment["CLR_TIMING"] == "1"
    func tlog(_ label: String, _ t0: DispatchTime) {
        guard timing else { return }
        let ms = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e6
        FileHandle.standardError.write(Data("[clr] \(label): \(String(format: "%.1f", ms)) ms\n".utf8))
    }

    let model: XGBoostTreeModel
    do { model = try XGBoostTreeModel(url: config.modelURL) } catch {
        throw NSError(domain: "CLR", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "Cannot load model: \(config.modelURL.path)"])
    }

    if timing {
        FileHandle.standardError.write(Data("[clr] device: \(Device.defaultDevice())\n".utf8))
    }

    let (img, realW, realH, padW, padH) =
        try loadImageMLX(nsImage, maxDim: config.maxDim, patchSize: config.patchSize)

    let ps = config.patchSize
    guard realW > 0, realH > 0 else {
        throw NSError(domain: "CLR", code: 2,
                      userInfo: [NSLocalizedDescriptionKey: "Empty image"])
    }
    // Grid is computed on the PADDED dims, so every patch is a full ps×ps block with full
    // 32×32 inner tiles. Right/bottom edge patches hold real pixels plus a thin reflected
    // border (pad < ps); they are reported below with their real, clipped footprint.
    let Pr = padH / ps, Pc = padW / ps
    let nr = ps / TILE_SIZE, nc = ps / TILE_SIZE

    var t0 = DispatchTime.now()
    let consts = GroupConstants(nr: nr, nc: nc, quantizeLikeCV2: config.cv2QuantizedWarp)
    tlog("warp tap tables", t0)

    // [Pr·Pc, ps, ps, 3] patch stack in row-major patch order.
    let allPatches = img
        .reshaped([Pr, ps, Pc, ps, 3])
        .transposed(0, 2, 1, 3, 4)
        .reshaped([Pr * Pc, ps, ps, 3])

    let total = Pr * Pc
    let tilesPerPatch = nr * nc
    let groupSize = max(1, EVAL_TILE_BUDGET / tilesPerPatch)

    var flatScores = [Float](repeating: 0, count: total)
    var flatTex = [Float](repeating: 0, count: total)
    var dumpRows = config.featureDumpURL != nil ? [[Float]]() : nil

    var done = 0
    var g0 = 0
    while g0 < total {
        let g1 = min(g0 + groupSize, total)
        t0 = DispatchTime.now()
        let (scores, tex, rows) = scoreGroup(allPatches[g0..<g1], c: consts, model: model)
        tlog("group \(g0)..<\(g1) (\((g1 - g0) * tilesPerPatch) tiles)", t0)
        for k in 0..<(g1 - g0) {
            flatScores[g0 + k] = scores[k]
            flatTex[g0 + k] = tex[k]
            done += 1
            progress(done, total, scores[k])
        }
        dumpRows?.append(contentsOf: rows)
        g0 = g1
    }

    if let url = config.featureDumpURL, let rows = dumpRows {
        let obj: [String: Any] = [
            "grid": ["rows": Pr, "cols": Pc, "tiles_per_patch": tilesPerPatch],
            "feature_order": ["FR", "LV", "CV", "CM", "CC", "CR", "SH", "SV", "SH_S", "SV_S"],
            "features": rows.map { $0.map { Double($0) } },
        ]
        try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]).write(to: url)
    }

    var patchGrid  = [[Float]](repeating: [Float](repeating: 0, count: Pc), count: Pr)
    var texWeights = [[Float]](repeating: [Float](repeating: 0, count: Pc), count: Pr)
    for I in 0..<Pr {
        for J in 0..<Pc {
            patchGrid[I][J]  = flatScores[I * Pc + J]
            texWeights[I][J] = flatTex[I * Pc + J]
        }
    }

    let weightMap = buildWeightMap(patchGrid: patchGrid, texWeights: texWeights,
                                   Pr: Pr, Pc: Pc, config: config)
    let weightedScore = weightedVerdict(patchGrid: patchGrid, weightMap: weightMap, config: config)
    let generatedCount = patchGrid.flatMap { $0 }.filter { $0 > 0.5 }.count
    let majorityScore = Float(generatedCount) / Float(Pr * Pc)

    var patches = [PatchInfo]()
    for I in 0..<Pr {
        for J in 0..<Pc {
            let x = J * ps, y = I * ps
            // Scored on a padded ps×ps block; the visible footprint is clipped to the real
            // image so edge cells render shorter/narrower and land exactly on the picture.
            let wReal = min(ps, realW - x)
            let hReal = min(ps, realH - y)
            patches.append(PatchInfo(
                row: I, col: J,
                pGenerated: patchGrid[I][J],
                weight: weightMap[I][J],
                x: x, y: y, width: wReal, height: hReal
            ))
        }
    }

    return InferenceResult(
        majorityPrediction: majorityScore > 0.5 ? "generated" : "real",
        majorityScore: majorityScore,
        weightedPrediction: weightedScore > 0.5 ? "generated" : "real",
        weightedScore: weightedScore,
        patches: patches,
        gridRows: Pr, gridCols: Pc,
        patchSize: ps,
        imageWidth: realW, imageHeight: realH
    )
}

// MARK: - Weight map (host-side; mirrors _build_weight_map)

private func buildWeightMap(
    patchGrid: [[Float]], texWeights: [[Float]],
    Pr: Int, Pc: Int, config: InferenceConfig
) -> [[Float]] {
    let flat = texWeights.flatMap { $0 }
    let tMin = flat.min() ?? 0, tMax = flat.max() ?? 1
    var tw = [[Float]](repeating: [Float](repeating: 0, count: Pc), count: Pr)
    for i in 0..<Pr { for j in 0..<Pc { tw[i][j] = (texWeights[i][j] - tMin) / (tMax - tMin + 1e-8) } }

    var cy: Float, cx: Float
    if config.saliencyCenter {
        var sumW: Float = 0, sumY: Float = 0, sumX: Float = 0
        for i in 0..<Pr { for j in 0..<Pc { sumY += Float(i) * tw[i][j]; sumX += Float(j) * tw[i][j]; sumW += tw[i][j] } }
        cy = sumY / (sumW + 1e-8); cx = sumX / (sumW + 1e-8)
    } else {
        cy = Float(Pr) / 2; cx = Float(Pc) / 2
    }

    let sigma = Float(max(Pr, Pc)) / config.centerSharpness
    var combined = [[Float]](repeating: [Float](repeating: 0, count: Pc), count: Pr)
    var total: Float = 0
    for i in 0..<Pr {
        for j in 0..<Pc {
            let dx = Float(j) - cx, dy = Float(i) - cy
            let cw = Foundation.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
            let texW = config.textureBias + (1 - config.textureBias) * pow(tw[i][j], config.texturePower)
            let conf = abs(patchGrid[i][j] - 0.5)
            let boost = 1.0 + 2.0 * conf
            combined[i][j] = cw * texW * boost
            total += combined[i][j]
        }
    }
    for i in 0..<Pr { for j in 0..<Pc { combined[i][j] /= max(total, 1e-12) } }
    return combined
}

private func weightedVerdict(patchGrid: [[Float]], weightMap: [[Float]], config: InferenceConfig) -> Float {
    var score: Float = 0
    let Pr = patchGrid.count, Pc = patchGrid[0].count
    for i in 0..<Pr { for j in 0..<Pc { score += patchGrid[i][j] * weightMap[i][j] } }
    var hotspots = [Float]()
    for i in 0..<Pr { for j in 0..<Pc { if patchGrid[i][j] > config.hotspotThresh { hotspots.append(patchGrid[i][j]) } } }
    if !hotspots.isEmpty {
        let hMean = hotspots.reduce(0, +) / Float(hotspots.count)
        score = (1 - config.hotspotBlend) * score + config.hotspotBlend * hMean
    }
    return score
}

// MARK: - Image loading helpers

/// Loads an NSImage as a float32 RGB MLXArray, then reflect-pads the right/bottom edges up to
/// the next multiple of `patchSize`. This makes the patch/tile grid cover the WHOLE image with
/// full ps×ps patches and full, in-distribution 32×32 inner tiles — no dropped edge strip and
/// no sub-32 tiles handed to the model. Pad amount is always < patchSize (≪ image), so a single
/// reflect suffices. Returns the padded array plus both the real (post-resize, pre-pad) and
/// padded dimensions; the caller scores on the padded dims and reports on the real ones.
private func loadImageMLX(_ nsImage: NSImage, maxDim: Int, patchSize: Int) throws
    -> (array: MLXArray, realW: Int, realH: Int, padW: Int, padH: Int) {

    guard let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        throw NSError(domain: "CLR", code: 3, userInfo: [NSLocalizedDescriptionKey: "Cannot get CGImage"])
    }
    var w = cgImage.width, h = cgImage.height
    if maxDim > 0 && max(w, h) > maxDim {
        let scale = Double(maxDim) / Double(max(w, h))
        w = Int(Double(w) * scale); h = Int(Double(h) * scale)
    }
    let bytesPerPixel = 4
    var pixelData = [UInt8](repeating: 0, count: w * h * bytesPerPixel)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
        data: &pixelData, width: w, height: h,
        bitsPerComponent: 8, bytesPerRow: w * bytesPerPixel,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    ) else {
        throw NSError(domain: "CLR", code: 4, userInfo: [NSLocalizedDescriptionKey: "Cannot create CGContext"])
    }
    ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))

    // Real (unpadded) RGB float buffer.
    var real = [Float](repeating: 0, count: h * w * 3)
    for i in 0..<h {
        for j in 0..<w {
            let base = (i * w + j) * bytesPerPixel
            real[(i * w + j) * 3 + 0] = Float(pixelData[base + 0])
            real[(i * w + j) * 3 + 1] = Float(pixelData[base + 1])
            real[(i * w + j) * 3 + 2] = Float(pixelData[base + 2])
        }
    }

    // Pad right/bottom up to the next multiple of patchSize via reflect-101.
    let padW = ((w + patchSize - 1) / patchSize) * patchSize
    let padH = ((h + patchSize - 1) / patchSize) * patchSize
    if padW == w && padH == h {
        return (MLXArray(real, [h, w, 3]), w, h, w, h)   // already aligned — no copy needed
    }
    let colMap = (0..<padW).map { reflect101Index($0, w) }
    let rowMap = (0..<padH).map { reflect101Index($0, h) }
    var padded = [Float](repeating: 0, count: padH * padW * 3)
    for i in 0..<padH {
        let si = rowMap[i]
        for j in 0..<padW {
            let sj = colMap[j]
            let src = (si * w + sj) * 3
            let dst = (i * padW + j) * 3
            padded[dst + 0] = real[src + 0]
            padded[dst + 1] = real[src + 1]
            padded[dst + 2] = real[src + 2]
        }
    }
    return (MLXArray(padded, [padH, padW, 3]), w, h, padW, padH)
}
