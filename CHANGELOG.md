## 2.0.0

* Replace the boxed nested input and output tensors in every model class with
  reused flat `Float32List`s handed to TFLite as `ByteBuffer`s, matching the
  approach in face_detection_tflite, pose_detection and hand_detection. Measured
  end to end on the cat_detection pipeline over a 3264x2448 photo in profile
  mode with `PerformanceMode.auto`, the full pipeline drops from 438 ms/frame to
  109 ms/frame and poseOnly from about 44 ms to 15 ms. Model outputs are
  unchanged.

  The two new helpers `ImageUtils.matToFloat32Simd` and
  `matToFloat32ImageNetSimd` use OpenCV's vectorized path and accept an optional
  caller buffer. Both are asserted equal to the per-pixel loops they replace:
  worst deviation 5.96e-8 (one float32 ULP) for the plain path, 7.15e-7 for the
  ImageNet affine. `ImageUtils.matToFloat32` and `matToFloat32ImageNet` are
  retained.

* Fix `ImageUtils.cropAndResize` returning `CropMetadata` built from
  pre-truncation floats while cropping an integral region. Callers mapped
  normalized coordinates against an origin up to 1px from where the crop
  actually began, and against a slightly too-large extent. Measured over the
  311-image CatFLW holdout with real localizer boxes, this placed landmarks
  +0.61px right and +0.52px down of ground truth and cost 0.255 NME_IOD,
  rising to 1.14 at the 95th percentile, with 72% of images improved by the
  fix. The Python training pipelines normalize against the integer crop, so
  this also aligns inference with how the models were trained.

  This changes returned landmark coordinates. Downstream packages should bump
  their own pipeline version so cached detections re-evaluate.

* Fix `ModelDownloader.modelHrnet`, which requested
  `superanimal_hrnet_w32_256_float16.tflite` while the release publishes
  `superanimal_hrnet_w32_float16.tflite`. `AnimalPoseModel.hrnet` therefore
  failed with an HTTP 404 on first use and had never worked. The private
  filename constant now derives from the public one so the two cannot drift.

* Reject a configured input size that disagrees with the bundled model, via the
  new internal `assertSquareInputSize`. `Interpreter.resizeInputTensor` accepts
  a shape the model was not trained for without reporting an error, and
  inference then returns finite but meaningless values: feeding a 256px
  landmark model at 384px measured NME_IOD 67.3 against a correct 3.5 while
  every output stayed in range. All six model classes now validate against
  `getInputTensor(0).shape` at initialization.

* Stop tracking `.DS_Store` files, which were included in the published
  archive.

## 1.4.0

* SSD anchors are now generated at runtime instead of shipping as a literal
  table. `lib/src/models/ssd_anchors.dart` was 12,944 lines, of which 12,936
  were a single float literal each; the values are the deterministic output of
  TF OD API's `create_ssd_anchors`, which `flutter_litert` already exports as
  `generateAnchors`. The library drops from 15,222 to 2,355 lines and the
  compiled binary shrinks by about 32 KB.
* Detection output is unchanged. Verified against the real SSDLite320 model
  over 9 images at 100 runs each: identical detection counts, bit-identical
  scores, and a worst-case box coordinate delta of 9.3e-05 px, which comes from
  the previous table having been rounded to 6 decimals while the generator is
  full float64. End-to-end timing is unchanged.
* Anchors are now stored in centre form (`cx, cy, w, h`), which is what
  `generateAnchors` emits and what the box decoder consumes, removing a
  corner round-trip that ran on every anchor of every frame.
* The exported anchor table is retained under `test/fixtures/` as the
  equivalence reference. `test/ssd_anchors_test.dart` regenerates the anchors
  and diffs all 3,234 against it on every run, so a change to the generator or
  its configuration fails immediately.
* Update flutter_litert -> 3.6.0.

## 1.3.3

* Update flutter_litert -> 3.5.0

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
