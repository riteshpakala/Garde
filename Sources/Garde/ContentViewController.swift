import AppKit
import Frigate

@MainActor
final class ContentViewController: NSViewController {

    // MARK: - UI elements

    private let gridView   = PatchGridView()
    private let verdictBar = VerdictBarView()
    private let modeControl: NSSegmentedControl = {
        let c = NSSegmentedControl(labels: ["P(generated)", "Weight map"],
                                   trackingMode: .selectOne,
                                   target: nil, action: nil)
        c.selectedSegment = 0
        return c
    }()
    private let openButton    = NSButton(title: "Open Image…", target: nil, action: nil)
    private let runButton     = NSButton(title: "Run Inference", target: nil, action: nil)
    private let progressLabel = NSTextField(labelWithString: "Ready.")
    private let configPanel   = ConfigPanelView()
    private var currentImage: NSImage?

    // MARK: - View lifecycle

    override func loadView() {
        view = NSView(frame: NSRect(x: 0, y: 0, width: 880, height: 760))
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        buildLayout()
        openButton.target = self; openButton.action = #selector(openImage)
        runButton.target  = self; runButton.action  = #selector(runInference)
        modeControl.target = self; modeControl.action = #selector(modeChanged)
        runButton.isEnabled = false
    }

    // MARK: - Layout

    private func buildLayout() {
        let topBar = NSStackView(views: [openButton, modeControl, NSView(), progressLabel])
        topBar.spacing = 10
        topBar.distribution = .fill
        topBar.orientation = .horizontal

        runButton.bezelStyle = .rounded
        openButton.bezelStyle = .rounded

        let btnRow = NSStackView(views: [configPanel, NSView(), runButton])
        btnRow.spacing = 12
        btnRow.orientation = .horizontal

        let stack = NSStackView(views: [topBar, btnRow, gridView, verdictBar])
        stack.orientation = .vertical
        stack.spacing = 8
        stack.edgeInsets = NSEdgeInsets(top: 12, left: 12, bottom: 12, right: 12)
        stack.translatesAutoresizingMaskIntoConstraints = false

        view.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            stack.topAnchor.constraint(equalTo: view.topAnchor),
            stack.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            gridView.heightAnchor.constraint(greaterThanOrEqualToConstant: 500),
            verdictBar.heightAnchor.constraint(equalToConstant: 36),
            topBar.heightAnchor.constraint(equalToConstant: 28),
            btnRow.heightAnchor.constraint(equalToConstant: 28),
        ])
    }

    // MARK: - Actions

    @objc private func openImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.jpeg, .png, .heic, .image]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.begin { [weak self] response in
            guard response == .OK, let url = panel.url else { return }
            DispatchQueue.main.async {
                self?.loadImage(from: url)
            }
        }
    }

    private func loadImage(from url: URL) {
        guard let img = NSImage(contentsOf: url) else { return }
        currentImage = img
        gridView.sourceImage = img
        gridView.result = nil
        verdictBar.result = nil
        progressLabel.stringValue = "Image loaded: \(url.lastPathComponent)"
        runButton.isEnabled = true
    }

    @objc private func runInference() {
        guard let img = currentImage else { return }
        runButton.isEnabled = false
        openButton.isEnabled = false
        progressLabel.stringValue = "Running…"

        let cfg = configPanel.currentConfig

        Task {
            do {
                let result = try await runInferenceFast(image: img, config: cfg) { done, total, score in
                    Task { @MainActor in
                        self.progressLabel.stringValue = "Patch \(done)/\(total)  P=\(String(format: "%.3f", score))"
                    }
                }
                await MainActor.run {
                    gridView.result    = result
                    verdictBar.result  = result
                    progressLabel.stringValue = "Done — \(result.gridRows)×\(result.gridCols) patches"
                    runButton.isEnabled  = true
                    openButton.isEnabled = true
                }
            } catch {
                await MainActor.run {
                    progressLabel.stringValue = "Error: \(error.localizedDescription)"
                    runButton.isEnabled  = true
                    openButton.isEnabled = true
                }
            }
        }
    }

    @objc private func modeChanged() {
        gridView.displayMode = modeControl.selectedSegment == 0 ? .pGenerated : .weightMap
    }
}

// MARK: - Config panel

@MainActor
final class ConfigPanelView: NSView {
    private let patchSizePop = NSPopUpButton()
    private let sharpField   = labeledField(label: "sharpness", value: "8.0")
    private let powerField   = labeledField(label: "tex_power", value: "3.0")

    var currentConfig: InferenceConfig {
        let ps = [64, 128, 256][max(0, patchSizePop.indexOfSelectedItem)]
        return InferenceConfig(
            modelURL:        InferenceConfig.defaultModelURL,
            patchSize:       ps,
            centerSharpness: Float(sharpField.1.stringValue) ?? 8.0,
            texturePower:    Float(powerField.1.stringValue) ?? 3.0
        )
    }

    override init(frame: NSRect) {
        super.init(frame: frame)
        patchSizePop.addItems(withTitles: ["64 px", "128 px", "256 px"])
        let row = NSStackView(views: [
            NSTextField(labelWithString: "patch:"), patchSizePop,
            sharpField.0, sharpField.1,
            powerField.0, powerField.1,
        ])
        row.spacing = 6; row.orientation = .horizontal
        addSubview(row)
        row.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            row.leadingAnchor.constraint(equalTo: leadingAnchor),
            row.trailingAnchor.constraint(equalTo: trailingAnchor),
            row.centerYAnchor.constraint(equalTo: centerYAnchor),
        ])
    }
    required init?(coder: NSCoder) { fatalError() }
}

private func labeledField(label: String, value: String) -> (NSTextField, NSTextField) {
    let lbl = NSTextField(labelWithString: label)
    lbl.setContentHuggingPriority(.defaultHigh, for: .horizontal)

    let fld = NSTextField(string: value)
    fld.setContentHuggingPriority(.defaultHigh, for: .horizontal)
    fld.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
    fld.widthAnchor.constraint(equalToConstant: 42).isActive = true
    return (lbl, fld)
}
