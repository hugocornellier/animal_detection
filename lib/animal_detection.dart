/// On-device animal detection, species classification, and body pose estimation
/// using TensorFlow Lite.
///
/// This library provides a Flutter plugin for detecting animals in images using
/// a multi-stage TFLite pipeline: SSD body detection, species classification,
/// and body pose estimation using SuperAnimal keypoints.
///
/// **Quick Start:**
/// ```dart
/// import 'package:animal_detection/animal_detection.dart';
///
/// final detector = AnimalDetector();
/// await detector.initialize();
///
/// final animals = await detector.detect(imageBytes);
/// for (final animal in animals) {
///   print('${animal.species} at ${animal.boundingBox} score=${animal.score}');
///   if (animal.pose != null) {
///     final tail = animal.pose!.getLandmark(AnimalPoseLandmarkType.tailEnd);
///     print('Tail: (${tail?.x}, ${tail?.y})');
///   }
/// }
///
/// await detector.dispose();
/// ```
///
/// **Main Classes:**
/// - [AnimalDetector]: Main API for animal detection
/// - [Animal]: Top-level detection result with body, species and pose info
/// - [AnimalPose]: Body pose result with 24 SuperAnimal keypoints
/// - [AnimalPoseLandmark]: Single body keypoint with 2D coordinates and confidence
/// - [BoundingBox]: Axis-aligned rectangle in pixel coordinates
///
/// **Pose Model Variants:**
/// - [AnimalPoseModel.rtmpose]: RTMPose-S (11.6MB, bundled). Fast SimCC-based decoder.
/// - [AnimalPoseModel.hrnet]: HRNet-w32 (54.6MB, downloaded on demand). Most accurate.
///
/// **Skeleton Connections:**
/// - [animalPoseConnections]: Body pose skeleton edges (SuperAnimal topology)
library;

export 'src/types.dart';
export 'src/animal_detector.dart' show AnimalDetector;
export 'src/util/model_downloader.dart' show ModelDownloader;
export 'src/dart_registration.dart';

// Re-export cv.Mat for users who want to use detectFromMat directly
export 'package:opencv_dart/opencv_dart.dart' show Mat, imdecode, IMREAD_COLOR;
export 'src/util/image_utils.dart' show ImageUtils;
export 'package:flutter_litert/flutter_litert.dart'
    show
        PerformanceMode,
        PerformanceConfig,
        BoundingBox,
        Point,
        LetterboxParams,
        computeLetterboxParams,
        scaleFromLetterbox;
export 'src/models/face_localizer_model.dart' show FaceLocalizerModel;
export 'src/models/landmark_model_runner.dart' show LandmarkModelRunnerBase;
export 'src/models/ensemble_landmark_model.dart'
    show EnsembleLandmarkModelBase, EnsembleModelGetter;
export 'src/util/species_model_downloader.dart' show SpeciesModelDownloader;
export 'src/models/single_interpreter_model.dart' show SingleInterpreterModel;
export 'src/util/math_utils.dart' show softmaxConfidence, argmaxSoftmax;
