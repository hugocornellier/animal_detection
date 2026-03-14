import 'dart:async';
import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';
import '../types.dart';
import '../util/image_utils.dart';

/// Generic face landmark regression model runner.
///
/// Runs an EfficientNetV2-based model that takes a cropped face image and
/// outputs normalized (x,y) pairs for each landmark. The raw output is
/// denormalized to original image coordinates using crop metadata.
///
/// Used by both cat (48 landmarks, 256px) and dog (46 landmarks, 384px)
/// face landmark pipelines. Species-specific landmark types are created
/// by the caller from the raw coordinate pairs.
class LandmarkModelRunnerBase {
  /// Input spatial dimension (width and height).
  final int inputSize;

  /// Number of landmarks the model predicts.
  final int numLandmarks;

  /// Flutter asset path for the TFLite model.
  final String modelPath;

  final InterpreterPool _pool;

  /// Creates a landmark model runner.
  LandmarkModelRunnerBase({
    required this.inputSize,
    required this.numLandmarks,
    required this.modelPath,
    int poolSize = 1,
  }) : _pool = InterpreterPool(poolSize: poolSize);

  /// Initializes the model from Flutter assets.
  Future<void> initialize(PerformanceConfig performanceConfig) async {
    final path = modelPath;
    await _pool.initialize(
      (options, _) async {
        final interpreter = await Interpreter.fromAsset(path, options: options);
        interpreter.resizeInputTensor(0, [1, inputSize, inputSize, 3]);
        interpreter.allocateTensors();
        return interpreter;
      },
      performanceConfig: performanceConfig,
    );
  }

  /// Initializes the model from pre-loaded bytes (for isolate use).
  Future<void> initializeFromBuffer(
    Uint8List bytes,
    PerformanceConfig performanceConfig,
  ) async {
    await _pool.initialize(
      (options, _) async {
        final interpreter = Interpreter.fromBuffer(bytes, options: options);
        interpreter.resizeInputTensor(0, [1, inputSize, inputSize, 3]);
        interpreter.allocateTensors();
        return interpreter;
      },
      performanceConfig: performanceConfig,
    );
  }

  /// Runs landmark prediction and returns denormalized (x, y) coordinate pairs.
  ///
  /// The returned list has [numLandmarks] entries, each a (x, y) record in
  /// original image pixel coordinates (mapped back via [meta]).
  Future<List<(double, double)>> predictRaw(
    cv.Mat crop,
    CropMetadata meta,
  ) async {
    return _pool.withInterpreter((interpreter, isolateInterpreter) async {
      final inputTensor = createNHWCTensor4D(inputSize, inputSize);
      final outputBuffer = allocTensorShape([1, numLandmarks * 2]);

      final rgb = ImageUtils.matToFloat32(crop);
      fillNHWC4D(rgb, inputTensor, inputSize, inputSize);

      final outputs = {0: outputBuffer};
      if (isolateInterpreter != null) {
        await isolateInterpreter.runForMultipleInputs([inputTensor], outputs);
      } else {
        interpreter.runForMultipleInputs([inputTensor], outputs);
      }

      final raw = (outputBuffer as List)[0] as List;
      final coords = <(double, double)>[];

      for (int i = 0; i < numLandmarks; i++) {
        final xNorm = (raw[i * 2] as double).clamp(0.0, 1.0);
        final yNorm = (raw[i * 2 + 1] as double).clamp(0.0, 1.0);
        coords.add(
            (xNorm * meta.cropW + meta.cx1, yNorm * meta.cropH + meta.cy1));
      }

      return coords;
    });
  }

  /// Releases native resources.
  void dispose() {
    _pool.dispose();
  }
}
