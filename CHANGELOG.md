## 1.3.2

* Update flutter_litert -> 3.4.1

## 1.3.1

* Update flutter_litert -> 3.3.1

## 1.3.0

* Update flutter_litert -> 3.2.0
* Import native-only flutter_litert APIs via `package:flutter_litert/native.dart` so they resolve under static analysis (flutter_litert 3.2.0 moved `InterpreterPool` and `InterpreterFactory` behind the native conditional export). No runtime or API change.

## 1.2.3

* Update flutter_litert -> 3.1.1

## 1.2.2

* Update flutter_litert -> 3.1.0

## 1.2.1

* Update flutter_litert -> 2.8.3

## 1.2.0

* Update flutter_litert -> 2.8.0
* Complete Swift Package Manager migration: example apps build via SPM without CocoaPods

## 1.1.1

* Remove unused Darwin podspecs for Dart-only iOS/macOS plugin registration.

## 1.1.0

* Update flutter_litert -> 2.5.8

## 1.0.12

* Update flutter_litert -> 2.5.5

## 1.0.11

* Update flutter_litert -> 2.5.4

## 1.0.10

* Update flutter_litert -> 2.5.3

## 1.0.9

* Update flutter_litert -> 2.5.2

## 1.0.8

* Update flutter_litert -> 2.5.0

## 1.0.7

* Update flutter_litert -> 2.4.1

## 1.0.6

* Update flutter_litert -> 2.4.0

## 1.0.5

* Update flutter_litert -> 2.3.0

## 1.0.4

* Update flutter_litert -> 2.2.0

## 1.0.3

* Update flutter_litert -> 2.1.0

## 1.0.2

* Update flutter_litert to 2.0.13

## 1.0.1

* Update flutter_litert -> 2.0.12

## 1.0.0

* First stable release. On-device animal detection, species/breed classification, and 24-point body pose estimation using TensorFlow Lite. Supports Android, iOS, macOS, Windows, and Linux with automatic hardware acceleration.

## 0.0.8

* Update documentation

## 0.0.7

* Update flutter_litert 2.0.8 -> 2.0.10

## 0.0.6

* Enable auto hardware acceleration by default (XNNPACK on all native platforms, Metal GPU on iOS)
* Update flutter_litert 2.0.6 -> 2.0.8

## 0.0.5

* Propagate useIsolateInterpreter flag through model initialization

## 0.0.4

* Add macOS Swift Package Manager support.

## 0.0.3

* Add shared face detection infrastructure for species-specific packages

## 0.0.2

* Add iOS Swift Package Manager support.

## 0.0.1

* Initial release: SSD body detection, species classification, and SuperAnimal pose estimation.
