import AppKit
import Foundation

// Load garde's resource bundle before anything MLX-related starts.
// This registers it in NSBundle.allBundles(), which is where MLX's Metal backend
// searches for the mlx-swift_Cmlx.bundle sub-bundle containing default.metallib.
// Without this, the Metal device init fires before the bundle is registered
// and MLX throws "Failed to load the default metallib."
_ = Bundle.module

// Dual-mode dispatch: any argument means CLI mode (`--headless` is accepted for
// compatibility but no longer required); no arguments boots the macOS app.
if CommandLine.arguments.count > 1 {
    runHeadlessCLI()
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
