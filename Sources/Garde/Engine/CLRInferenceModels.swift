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