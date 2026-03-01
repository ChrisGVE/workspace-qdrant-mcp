// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Bookshelf",
    targets: [
        .executableTarget(
            name: "Bookshelf",
            path: "Sources/Bookshelf"
        ),
    ]
)
