// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Garde",
    platforms: [.macOS("14.0")],
    dependencies: [
        .package(url: "https://github.com/rao-studios/Frigate.git", branch: "main")
    ],
    targets: [
        // Single dual-mode executable: `garde` with no arguments boots the macOS
        // app; with an image path / flags it runs the headless CLI scorer.
        .executableTarget(
            name: "garde",
            dependencies: [
                .product(name: "Frigate", package: "Frigate"),
            ],
            path: "Sources/Garde",
            resources: [
                .copy("clr_v1.8.xgb.json"),
                .copy("clr_tile32_v1.8_t9_final_4k_20k.xgb.json"),
                // MLX Metal shader library, bundled as the sub-bundle name MLX searches for.
                // See SWIFTPM_BUNDLE = "mlx-swift_Cmlx" in Frigate/Package.swift.
                .copy("mlx-swift_Cmlx.bundle"),
            ]
        )
    ]
)
