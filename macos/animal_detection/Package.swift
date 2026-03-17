// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "animal_detection",
    platforms: [
        .macOS("11.0")
    ],
    products: [
        .library(name: "animal-detection", targets: ["animal_detection"])
    ],
    dependencies: [],
    targets: [
        .target(
            name: "animal_detection",
            dependencies: [],
            resources: [
                .process("PrivacyInfo.xcprivacy"),
            ]
        )
    ]
)
