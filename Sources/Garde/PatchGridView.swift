/// PatchGridView — NSView that overlays the inference result on the source image.
///
/// Two modes toggled by `displayMode`:
///   .pGenerated  — cells coloured green→red by P(generated)
///   .weightMap   — cells coloured by patch weight (purple→yellow)

import AppKit

enum DisplayMode { case pGenerated, weightMap }

final class PatchGridView: NSView {
    var sourceImage: NSImage? { didSet { needsDisplay = true } }
    var result: InferenceResult? { didSet { needsDisplay = true } }
    var displayMode: DisplayMode = .pGenerated { didSet { needsDisplay = true } }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        NSColor.black.setFill()
        bounds.fill()

        guard let img = sourceImage else { return }
        let imgRect = aspectFitRect(for: NSSize(width: img.size.width,
                                                 height: img.size.height),
                                    in: bounds)
        img.draw(in: imgRect)
        guard let r = result else { return }

        // Scale from inference-image coords to view coords
        let scaleX = imgRect.width  / CGFloat(r.imageWidth)
        let scaleY = imgRect.height / CGFloat(r.imageHeight)

        let weights = r.patches.map(\.weight)
        let wMin = weights.min() ?? 0, wMax = weights.max() ?? 1

        let ps  = CGFloat(r.patchSize)
        let scoredW = CGFloat(r.gridCols) * ps * scaleX
        let scoredH = CGFloat(r.gridRows) * ps * scaleY

        // ── Partial-strip fills (right column, bottom row, corner) ───────────
        // Show unscorable edge strips with a subtle neutral overlay so the grid
        // visually fills the whole image, matching the Python visualisation.
        let partialRight  = imgRect.width  - scoredW
        let partialBottom = imgRect.height - scoredH
        if partialRight > 0.5 || partialBottom > 0.5 {
            NSColor.white.withAlphaComponent(0.06).setFill()
            if partialRight > 0.5 {
                NSBezierPath.fill(CGRect(x: imgRect.minX + scoredW, y: imgRect.minY + partialBottom,
                                         width: partialRight, height: scoredH))
            }
            if partialBottom > 0.5 {
                NSBezierPath.fill(CGRect(x: imgRect.minX, y: imgRect.minY,
                                         width: imgRect.width, height: partialBottom))
            }
        }

        // ── Scored patch cells ───────────────────────────────────────────────
        for patch in r.patches {
            let rect = CGRect(
                x: imgRect.minX + CGFloat(patch.x) * scaleX,
                y: imgRect.maxY - CGFloat(patch.y + patch.height) * scaleY,
                width:  CGFloat(patch.width)  * scaleX,
                height: CGFloat(patch.height) * scaleY
            )

            let color: NSColor
            let alpha: CGFloat
            switch displayMode {
            case .pGenerated:
                let p = CGFloat(patch.pGenerated)
                let hue = (1 - p) * 120.0 / 360.0
                alpha = 0.15 + 0.55 * abs(p - 0.5) * 2
                color = NSColor(hue: hue, saturation: 0.9, brightness: 0.75, alpha: alpha)
            case .weightMap:
                let norm = wMax > wMin ? CGFloat((patch.weight - wMin) / (wMax - wMin)) : 0
                let hue = (270 - norm * 200) / 360.0
                alpha = 0.1 + 0.7 * norm
                color = NSColor(hue: hue, saturation: 0.8, brightness: 0.75, alpha: alpha)
            }

            color.setFill()
            NSBezierPath.fill(rect)

            // Score label
            let value = displayMode == .pGenerated ? patch.pGenerated : patch.weight
            let label = displayMode == .pGenerated
                ? String(format: "%.2f\n%@", value, patch.verdict.prefix(1).uppercased())
                : String(format: "%.3f", value)
            let fontSize = max(8.0, min(12.0, rect.width * 0.25))
            let attr: [NSAttributedString.Key: Any] = [
                .font: NSFont.monospacedSystemFont(ofSize: fontSize, weight: .regular),
                .foregroundColor: NSColor.white,
                .shadow: {
                    let s = NSShadow(); s.shadowColor = .black; s.shadowBlurRadius = 2; return s
                }()
            ]
            let str = NSAttributedString(string: label, attributes: attr)
            let sz = str.size()
            str.draw(at: NSPoint(x: rect.midX - sz.width/2, y: rect.midY - sz.height/2))
        }

        // ── Full-image grid lines ────────────────────────────────────────────
        // Draw lines across the entire image extent, including edge strips.
        let gridColor = NSColor.white.withAlphaComponent(0.25)
        gridColor.setStroke()
        let lines = NSBezierPath()
        lines.lineWidth = 0.5

        // Vertical lines at every patch column boundary + right image edge
        var x = imgRect.minX
        while x <= imgRect.maxX + 0.5 {
            lines.move(to: NSPoint(x: x, y: imgRect.minY))
            lines.line(to: NSPoint(x: x, y: imgRect.maxY))
            x += ps * scaleX
        }
        // Ensure the right image edge always has a line
        lines.move(to: NSPoint(x: imgRect.maxX, y: imgRect.minY))
        lines.line(to: NSPoint(x: imgRect.maxX, y: imgRect.maxY))

        // Horizontal lines at every patch row boundary + bottom image edge
        var y = imgRect.maxY
        while y >= imgRect.minY - 0.5 {
            lines.move(to: NSPoint(x: imgRect.minX, y: y))
            lines.line(to: NSPoint(x: imgRect.maxX, y: y))
            y -= ps * scaleY
        }
        lines.move(to: NSPoint(x: imgRect.minX, y: imgRect.minY))
        lines.line(to: NSPoint(x: imgRect.maxX, y: imgRect.minY))

        lines.stroke()
    }

    private func aspectFitRect(for imageSize: NSSize, in viewRect: CGRect) -> CGRect {
        let sx = viewRect.width  / imageSize.width
        let sy = viewRect.height / imageSize.height
        let scale = min(sx, sy)
        let w = imageSize.width  * scale
        let h = imageSize.height * scale
        return CGRect(
            x: viewRect.midX - w/2,
            y: viewRect.midY - h/2,
            width: w, height: h
        )
    }
}

// MARK: - Verdict Bar

final class VerdictBarView: NSView {
    var result: InferenceResult? { didSet { updateLabels() } }

    private let majorityLabel = NSTextField(labelWithString: "")
    private let weightedLabel = NSTextField(labelWithString: "")

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor
        [majorityLabel, weightedLabel].forEach {
            $0.alignment = .center
            $0.font = .systemFont(ofSize: 14, weight: .semibold)
            addSubview($0)
        }
    }
    required init?(coder: NSCoder) { fatalError() }

    override func layout() {
        super.layout()
        let half = bounds.width / 2
        majorityLabel.frame = CGRect(x: 0, y: 0, width: half, height: bounds.height)
        weightedLabel.frame = CGRect(x: half, y: 0, width: half, height: bounds.height)
    }

    private func updateLabels() {
        guard let r = result else { majorityLabel.stringValue = ""; weightedLabel.stringValue = ""; return }
        func label(_ pred: String, _ score: Float, _ kind: String) -> NSAttributedString {
            let conf = abs(score - 0.5) >= 0.3 ? "high" : abs(score - 0.5) >= 0.15 ? "med" : "low"
            let text = "\(kind): \(pred.uppercased())  \(String(format: "%.3f", score))  (\(conf))"
            let color: NSColor = pred == "generated" ? .systemRed : .systemGreen
            return NSAttributedString(string: text, attributes: [
                .foregroundColor: color,
                .font: NSFont.systemFont(ofSize: 14, weight: .semibold)
            ])
        }
        majorityLabel.attributedStringValue = label(r.majorityPrediction, r.majorityScore, "Majority")
        weightedLabel.attributedStringValue = label(r.weightedPrediction, r.weightedScore, "Weighted")
    }
}
