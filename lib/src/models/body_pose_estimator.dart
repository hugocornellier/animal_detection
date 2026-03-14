import 'dart:async';
import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';
import '../types.dart';
import '../util/image_utils.dart';
import '../util/math_utils.dart';
import 'single_interpreter_model.dart';

/// SuperAnimal body pose estimator supporting RTMPose-S (SimCC) and HRNet-w32 (heatmap).
///
/// RTMPose-S (bundled): fast SimCC-based decoder, 11.6 MB.
/// HRNet-w32 (downloaded on demand): heatmap-based, most accurate, 54.6 MB.
///
/// Both models take a 256x256 ImageNet-normalized letterboxed crop as input
/// and output 39 SuperAnimal keypoints. Only body keypoints 15-38 are exposed
/// as [AnimalPoseLandmarkType] values.
class BodyPoseEstimator extends SingleInterpreterModel {
  /// Input spatial dimension for both pose models (256x256).
  static const int inputSize = 256;

  static const int _numKeypoints = 39;
  static const double _simccSplitRatio = 2.0;
  static const int _heatmapSize = 64;
  static const int _bodyKeypointStart = 15;
  static const int _simccBins = 512;

  static const String _rtmposePath =
      'packages/animal_detection/assets/models/superanimal_rtmpose_s_float16.tflite';

  /// The pose model variant this estimator is configured for.
  final AnimalPoseModel model;

  late List<List<List<List<double>>>> _inputTensor;

  // RTMPose output buffers: simcc_x [1, 39, 512] and simcc_y [1, 39, 512]
  late List<List<List<double>>> _simccX;
  late List<List<List<double>>> _simccY;

  // HRNet output buffer: heatmaps [1, 64, 64, 39]
  late List<List<List<List<double>>>> _heatmapBuffer;

  Float32List? _rgbBuffer;

  /// Creates a pose estimator for the given [model] variant.
  BodyPoseEstimator({required this.model});

  /// Initializes the estimator by loading the TFLite model from Flutter assets.
  ///
  /// Only valid for [AnimalPoseModel.rtmpose]. For [AnimalPoseModel.hrnet], the model
  /// bytes must be provided via [initializeFromBuffer].
  Future<void> initialize(PerformanceConfig performanceConfig) async {
    if (model == AnimalPoseModel.hrnet) {
      throw StateError(
        'HRNet requires a downloaded model. '
        'Use initializeFromBuffer() with bytes from ModelDownloader.',
      );
    }
    await initInterpreterFromAsset(_rtmposePath, performanceConfig);
    _allocBuffers();
  }

  /// Initializes the estimator from pre-loaded model bytes.
  Future<void> initializeFromBuffer(
    Uint8List bytes,
    PerformanceConfig performanceConfig,
  ) async {
    await initInterpreterFromBuffer(bytes, performanceConfig);
    _allocBuffers();
  }

  void _allocBuffers() {
    _inputTensor = createNHWCTensor4D(inputSize, inputSize);
    if (model == AnimalPoseModel.rtmpose) {
      _simccX = allocTensorShape([1, _numKeypoints, _simccBins])
          as List<List<List<double>>>;
      _simccY = allocTensorShape([1, _numKeypoints, _simccBins])
          as List<List<List<double>>>;
    } else {
      _heatmapBuffer =
          allocTensorShape([1, _heatmapSize, _heatmapSize, _numKeypoints])
              as List<List<List<List<double>>>>;
    }
  }

  /// Run pose estimation on an animal crop.
  ///
  /// [crop]: OpenCV Mat of the animal region (already cropped from the full image).
  /// [cropX], [cropY]: Offset of the crop in the original image for coordinate mapping.
  ///
  /// Returns an [AnimalPose] with up to 24 body landmarks in original image coordinates.
  Future<AnimalPose> estimate(
    cv.Mat crop, {
    required int cropX,
    required int cropY,
  }) async {
    // 1. Letterbox resize to 256x256, preserving aspect ratio.
    final (padded, params) = ImageUtils.letterboxResize(crop, inputSize);

    // 2. ImageNet normalize (BGR->RGB with (x/255 - mean) / std).
    _rgbBuffer = ImageUtils.matToFloat32ImageNet(padded);
    fillNHWC4D(_rgbBuffer!, _inputTensor, inputSize, inputSize);
    padded.dispose();

    // 3. Run model and decode keypoints in 256px letterbox space.
    final List<({double x, double y, double confidence})> kps;
    if (model == AnimalPoseModel.rtmpose) {
      kps = await _runRTMPose();
    } else {
      kps = await _runHRNet();
    }

    // 4. Un-letterbox: map from 256px letterbox space -> crop space -> original image.
    final double scale = params.scale;
    final double padLeft = params.padLeft.toDouble();
    final double padTop = params.padTop.toDouble();

    // 5. Extract body keypoints (indices 15-38) and map to AnimalPoseLandmarkType.
    final landmarks = <AnimalPoseLandmark>[];
    for (int i = _bodyKeypointStart; i < _numKeypoints; i++) {
      final kp = kps[i];
      final double xOrig = (kp.x - padLeft) / scale + cropX;
      final double yOrig = (kp.y - padTop) / scale + cropY;
      final landmarkType =
          AnimalPoseLandmarkType.values[i - _bodyKeypointStart];
      landmarks.add(AnimalPoseLandmark(
        type: landmarkType,
        x: xOrig,
        y: yOrig,
        confidence: kp.confidence,
      ));
    }

    return AnimalPose(landmarks: landmarks);
  }

  Future<List<({double x, double y, double confidence})>> _runRTMPose() async {
    final outputs = <int, Object>{0: _simccX, 1: _simccY};

    await runInference([_inputTensor], outputs);

    final result = <({double x, double y, double confidence})>[];
    for (int kp = 0; kp < _numKeypoints; kp++) {
      final xRow = _simccX[0][kp];
      final yRow = _simccY[0][kp];

      // argmax
      int xArgmax = 0;
      int yArgmax = 0;
      double xMax = xRow[0];
      double yMax = yRow[0];
      for (int b = 1; b < _simccBins; b++) {
        if (xRow[b] > xMax) {
          xMax = xRow[b];
          xArgmax = b;
        }
        if (yRow[b] > yMax) {
          yMax = yRow[b];
          yArgmax = b;
        }
      }

      // pixel in [0, 255] space
      final double xPixel = xArgmax / _simccSplitRatio;
      final double yPixel = yArgmax / _simccSplitRatio;

      // softmax confidence = max(softmax(simcc_x[kp])) * max(softmax(simcc_y[kp]))
      final double xConf = softmaxConfidence(xRow, xArgmax);
      final double yConf = softmaxConfidence(yRow, yArgmax);
      final double confidence = xConf * yConf;

      result.add((x: xPixel, y: yPixel, confidence: confidence));
    }
    return result;
  }

  Future<List<({double x, double y, double confidence})>> _runHRNet() async {
    final outputs = <int, Object>{0: _heatmapBuffer};

    await runInference([_inputTensor], outputs);

    final double scale = inputSize / _heatmapSize.toDouble(); // 4.0

    final result = <({double x, double y, double confidence})>[];
    for (int kp = 0; kp < _numKeypoints; kp++) {
      int bestRow = 0;
      int bestCol = 0;
      double bestVal = _heatmapBuffer[0][0][0][kp];

      for (int row = 0; row < _heatmapSize; row++) {
        for (int col = 0; col < _heatmapSize; col++) {
          final double v = _heatmapBuffer[0][row][col][kp];
          if (v > bestVal) {
            bestVal = v;
            bestRow = row;
            bestCol = col;
          }
        }
      }

      final double xPixel = bestCol * scale;
      final double yPixel = bestRow * scale;

      result.add((x: xPixel, y: yPixel, confidence: bestVal));
    }
    return result;
  }
}
