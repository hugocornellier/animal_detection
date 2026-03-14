import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'types.dart';
import 'util/image_utils.dart';
import 'util/model_downloader.dart';
import 'models/animal_body_detector.dart';
import 'models/species_classifier.dart';
import 'models/body_pose_estimator.dart';

/// On-device animal detection using a multi-stage TensorFlow Lite pipeline.
///
/// Runs SSD body detection, species classification, and optionally body pose
/// estimation. Returns a list of [Animal] objects with bounding boxes, species
/// labels, and pose keypoints.
///
/// Usage:
/// ```dart
/// final detector = AnimalDetector();
/// await detector.initialize();
/// final animals = await detector.detect(imageBytes);
/// await detector.dispose();
/// ```
class AnimalDetector {
  AnimalBodyDetector? _bodyDetector;
  SpeciesClassifier? _classifier;
  BodyPoseEstimator? _poseEstimator;

  /// Body pose model variant.
  final AnimalPoseModel poseModel;

  /// Whether to run pose estimation.
  final bool enablePose;

  /// Margin fraction added to each side of the body bounding box before cropping.
  final double cropMargin;

  /// SSD detection score threshold.
  final double detThreshold;

  /// Performance configuration for TensorFlow Lite inference.
  final PerformanceConfig performanceConfig;

  bool _isInitialized = false;

  /// Creates an animal detector with the specified configuration.
  AnimalDetector({
    this.poseModel = AnimalPoseModel.rtmpose,
    this.enablePose = true,
    this.cropMargin = 0.20,
    this.detThreshold = 0.5,
    this.performanceConfig = PerformanceConfig.disabled,
  });

  /// Initializes the detector by loading TensorFlow Lite models.
  ///
  /// Must be called before [detect] or [detectFromMat].
  ///
  /// When [poseModel] is [AnimalPoseModel.hrnet], the HRNet model (~54.6 MB) is
  /// downloaded from GitHub Releases on first use and cached locally.
  ///
  /// [onDownloadProgress] is called during any model download with
  /// (modelName, bytesReceived, totalBytes).
  Future<void> initialize({
    void Function(String model, int received, int total)? onDownloadProgress,
  }) async {
    if (_isInitialized) {
      await dispose();
    }

    _bodyDetector = AnimalBodyDetector();
    await _bodyDetector!.initialize(performanceConfig);

    _classifier = SpeciesClassifier();
    await _classifier!.initialize(performanceConfig);

    if (enablePose) {
      _poseEstimator = BodyPoseEstimator(model: poseModel);
      if (poseModel == AnimalPoseModel.hrnet) {
        final hrnetBytes = await ModelDownloader.getHrnetModel(
          onProgress: onDownloadProgress != null
              ? (r, t) => onDownloadProgress(ModelDownloader.modelHrnet, r, t)
              : null,
        );
        await _poseEstimator!
            .initializeFromBuffer(hrnetBytes, performanceConfig);
      } else {
        await _poseEstimator!.initialize(performanceConfig);
      }
    }

    _isInitialized = true;
  }

  /// Initializes the detector from pre-loaded model bytes.
  ///
  /// Used for initialization within a background isolate where Flutter asset
  /// loading is not available.
  Future<void> initializeFromBuffers({
    required Uint8List bodyDetectorBytes,
    required Uint8List classifierBytes,
    required String speciesMappingJson,
    Uint8List? poseModelBytes,
  }) async {
    if (_isInitialized) {
      await dispose();
    }

    _bodyDetector = AnimalBodyDetector();
    await _bodyDetector!.initializeFromBuffer(
      bodyDetectorBytes,
      performanceConfig,
    );

    _classifier = SpeciesClassifier();
    await _classifier!.initializeFromBuffer(
      classifierBytes,
      speciesMappingJson,
      performanceConfig,
    );

    if (enablePose && poseModelBytes != null) {
      _poseEstimator = BodyPoseEstimator(model: poseModel);
      await _poseEstimator!
          .initializeFromBuffer(poseModelBytes, performanceConfig);
    }

    _isInitialized = true;
  }

  /// Returns true if the detector has been initialized and is ready to use.
  bool get isInitialized => _isInitialized;

  /// Returns true if the HRNet model is already cached locally.
  static Future<bool> isHrnetCached() => ModelDownloader.isHrnetCached();

  /// Releases all resources used by the detector.
  Future<void> dispose() async {
    _bodyDetector?.dispose();
    _classifier?.dispose();
    _poseEstimator?.dispose();
    _bodyDetector = null;
    _classifier = null;
    _poseEstimator = null;
    _isInitialized = false;
  }

  /// Detects animals in an image from raw bytes.
  ///
  /// Decodes the image bytes using OpenCV and runs the detection pipeline.
  ///
  /// Returns a list of [Animal] objects. Returns an empty list if image decoding
  /// fails or no animals are detected.
  ///
  /// Throws [StateError] if called before [initialize].
  Future<List<Animal>> detect(Uint8List imageBytes) async {
    if (!_isInitialized) {
      throw StateError(
          'AnimalDetector not initialized. Call initialize() first.');
    }
    try {
      final mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
      if (mat.isEmpty) return <Animal>[];
      try {
        return await detectFromMat(
          mat,
          imageWidth: mat.cols,
          imageHeight: mat.rows,
        );
      } finally {
        mat.dispose();
      }
    } catch (e) {
      return <Animal>[];
    }
  }

  /// Detects animals in an OpenCV Mat image.
  ///
  /// Runs the pipeline: SSD detection -> classify -> optional pose estimation.
  ///
  /// Returns a list of [Animal] objects.
  ///
  /// Throws [StateError] if called before [initialize].
  Future<List<Animal>> detectFromMat(
    cv.Mat image, {
    required int imageWidth,
    required int imageHeight,
  }) async {
    if (!_isInitialized) {
      throw StateError(
          'AnimalDetector not initialized. Call initialize() first.');
    }

    // Stage 1: SSD body detection
    final detections = await _bodyDetector!.detect(
      image,
      scoreThreshold: detThreshold,
    );
    if (detections.isEmpty) return <Animal>[];

    final animals = <Animal>[];

    for (final (bbox, score) in detections) {
      String? species;
      String? breed;
      double? speciesConfidence;
      AnimalPose? pose;

      // Stage 2: species classification on the original (unexpanded) bbox
      final origBw = (bbox.right - bbox.left).toInt();
      final origBh = (bbox.bottom - bbox.top).toInt();
      if (origBw >= 1 && origBh >= 1) {
        final classifyCrop = image.region(
          cv.Rect(
            bbox.left.toInt(),
            bbox.top.toInt(),
            origBw,
            origBh,
          ),
        );
        try {
          final (sp, br, conf) = await _classifier!.classify(classifyCrop);
          species = sp;
          breed = br;
          speciesConfidence = conf;
        } finally {
          classifyCrop.dispose();
        }
      }

      // Stage 3: body pose estimation on the expanded crop
      if (enablePose && _poseEstimator != null) {
        final (cx1, cy1, cx2, cy2) = ImageUtils.expandBox(
          bbox.left,
          bbox.top,
          bbox.right,
          bbox.bottom,
          cropMargin,
          imageWidth,
          imageHeight,
        );

        final int cropW = cx2 - cx1;
        final int cropH = cy2 - cy1;
        if (cropW >= 1 && cropH >= 1) {
          final expandedCrop = image.region(cv.Rect(cx1, cy1, cropW, cropH));
          try {
            pose = await _poseEstimator!.estimate(
              expandedCrop,
              cropX: cx1,
              cropY: cy1,
            );
          } finally {
            expandedCrop.dispose();
          }
        }
      }

      animals.add(Animal(
        boundingBox: bbox,
        score: score,
        species: species,
        breed: breed,
        speciesConfidence: speciesConfidence,
        pose: pose,
        imageWidth: imageWidth,
        imageHeight: imageHeight,
      ));
    }

    return animals;
  }
}
