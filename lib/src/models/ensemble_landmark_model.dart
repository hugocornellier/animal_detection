import 'dart:async';
import 'package:flutter/services.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';
import '../util/image_utils.dart';
import '../types.dart';

/// Callback type for downloading ensemble model weights.
typedef EnsembleModelGetter = Future<(Uint8List, Uint8List)> Function({
  void Function(String model, int received, int total)? onProgress,
});

/// 3-model ensemble landmark runner with multi-scale + flip TTA.
///
/// Runs 3 models (256px + 320px + 384px) x 3 scales (0.9, 1.0, 1.1) x
/// 2 orientations (original + horizontal flip) = 18 inference passes,
/// averaged in normalized [0,1] space for robust landmark prediction.
///
/// Species-specific parameters (landmark count, flip index, model paths,
/// download function) are passed at construction time.
class EnsembleLandmarkModelBase {
  static const int _size256 = 256;
  static const int _size320 = 320;
  static const int _size384 = 384;

  static const List<double> _scales = [0.9, 1.0, 1.1];

  /// Number of landmarks the model predicts.
  final int numLandmarks;

  /// Landmark index permutation for horizontal flip.
  final List<int> flipIndex;

  /// Flutter asset path for the bundled 384px model.
  final String bundledModelPath;

  /// Function to download/cache the 256px and 320px ensemble models.
  final EnsembleModelGetter getEnsembleModels;

  final int _poolSize;

  InterpreterPool? _pool256;
  InterpreterPool? _pool320;
  InterpreterPool? _pool384;

  /// Creates an ensemble model runner.
  EnsembleLandmarkModelBase({
    required this.numLandmarks,
    required this.flipIndex,
    required this.bundledModelPath,
    required this.getEnsembleModels,
    int poolSize = 1,
  }) : _poolSize = poolSize;

  /// Initializes all 3 models.
  ///
  /// The 384px model is loaded from bundled assets.
  /// The 256px and 320px models are obtained via [getEnsembleModels].
  Future<void> initialize(
    PerformanceConfig performanceConfig, {
    void Function(String model, int received, int total)? onDownloadProgress,
  }) async {
    final (bytes256, bytes320) = await getEnsembleModels(
      onProgress: onDownloadProgress,
    );

    final byteData384 = await rootBundle.load(bundledModelPath);
    final bytes384 = byteData384.buffer.asUint8List();

    await _initializeFromBytes(
      bytes256: bytes256,
      bytes320: bytes320,
      bytes384: bytes384,
      performanceConfig: performanceConfig,
    );
  }

  /// Initializes from pre-loaded model bytes (for isolate use).
  Future<void> initializeFromBuffers({
    required Uint8List bytes256,
    required Uint8List bytes320,
    required Uint8List bytes384,
    required PerformanceConfig performanceConfig,
  }) async {
    await _initializeFromBytes(
      bytes256: bytes256,
      bytes320: bytes320,
      bytes384: bytes384,
      performanceConfig: performanceConfig,
    );
  }

  Future<void> _initializeFromBytes({
    required Uint8List bytes256,
    required Uint8List bytes320,
    required Uint8List bytes384,
    required PerformanceConfig performanceConfig,
  }) async {
    _pool256 = InterpreterPool(poolSize: _poolSize);
    _pool320 = InterpreterPool(poolSize: _poolSize);
    _pool384 = InterpreterPool(poolSize: _poolSize);

    await Future.wait([
      _initPool(_pool256!, bytes256, _size256, performanceConfig),
      _initPool(_pool320!, bytes320, _size320, performanceConfig),
      _initPool(_pool384!, bytes384, _size384, performanceConfig),
    ]);
  }

  Future<void> _initPool(
    InterpreterPool pool,
    Uint8List bytes,
    int inputSize,
    PerformanceConfig config,
  ) async {
    await pool.initialize(
      (options, _) async {
        final interpreter = Interpreter.fromBuffer(bytes, options: options);
        interpreter.resizeInputTensor(0, [1, inputSize, inputSize, 3]);
        interpreter.allocateTensors();
        return interpreter;
      },
      performanceConfig: config,
    );
  }

  /// The input size used for cropping (largest model size = 384).
  int get inputSize => _size384;

  /// Runs the 18-pass ensemble and returns denormalized (x, y) coordinate pairs.
  Future<List<(double, double)>> predictRaw(
    cv.Mat crop384,
    CropMetadata meta,
  ) async {
    final crop256 = cv.resize(crop384, (_size256, _size256));
    final crop320 = cv.resize(crop384, (_size320, _size320));

    final tempMats = <cv.Mat>[];
    final futures = <Future<List<double>>>[];
    final outputLen = numLandmarks * 2;

    for (final (pool, crop, size) in [
      (_pool256!, crop256, _size256),
      (_pool320!, crop320, _size320),
      (_pool384!, crop384, _size384),
    ]) {
      for (final scale in _scales) {
        final scaled = _scaleCrop(crop, scale, size);
        if ((scale - 1.0).abs() >= 1e-6) tempMats.add(scaled);

        futures.add(
          _runModelRaw(pool, scaled, size, outputLen).then(
            (raw) => _unscaleCoords(raw, scale),
          ),
        );

        final flipped = cv.flip(scaled, 1);
        tempMats.add(flipped);

        futures.add(
          _runModelRaw(pool, flipped, size, outputLen).then((raw) {
            final remapped = List<double>.filled(outputLen, 0.0);
            for (int i = 0; i < numLandmarks; i++) {
              final srcIdx = flipIndex[i];
              remapped[i * 2] = 1.0 - raw[srcIdx * 2];
              remapped[i * 2 + 1] = raw[srcIdx * 2 + 1];
            }
            return _unscaleCoords(remapped, scale);
          }),
        );
      }
    }

    try {
      final allPreds = await Future.wait(futures);

      final coords = <(double, double)>[];
      for (int i = 0; i < numLandmarks; i++) {
        double xSum = 0.0;
        double ySum = 0.0;
        for (final pred in allPreds) {
          xSum += pred[i * 2];
          ySum += pred[i * 2 + 1];
        }
        final xNorm = (xSum / allPreds.length).clamp(0.0, 1.0);
        final yNorm = (ySum / allPreds.length).clamp(0.0, 1.0);
        coords.add(
            (xNorm * meta.cropW + meta.cx1, yNorm * meta.cropH + meta.cy1));
      }

      return coords;
    } finally {
      for (final mat in tempMats) {
        mat.dispose();
      }
      crop256.dispose();
      crop320.dispose();
    }
  }

  cv.Mat _scaleCrop(cv.Mat crop, double scale, int inputSize) {
    if ((scale - 1.0).abs() < 1e-6) return crop;

    if (scale < 1.0) {
      final newH = (inputSize * scale).round();
      final newW = (inputSize * scale).round();
      final offsetY = (inputSize - newH) ~/ 2;
      final offsetX = (inputSize - newW) ~/ 2;
      final cropped = crop.region(cv.Rect(offsetX, offsetY, newW, newH));
      final resized = cv.resize(cropped, (inputSize, inputSize));
      cropped.dispose();
      return resized;
    } else {
      final padH = (inputSize * (scale - 1.0) / 2.0).round();
      final padW = (inputSize * (scale - 1.0) / 2.0).round();
      final padded = cv.copyMakeBorder(
        crop,
        padH,
        padH,
        padW,
        padW,
        cv.BORDER_REFLECT_101,
      );
      final resized = cv.resize(padded, (inputSize, inputSize));
      padded.dispose();
      return resized;
    }
  }

  List<double> _unscaleCoords(List<double> coords, double scale) {
    if ((scale - 1.0).abs() < 1e-6) return coords;
    return List<double>.generate(
      coords.length,
      (i) => (coords[i] - 0.5) * scale + 0.5,
    );
  }

  Future<List<double>> _runModelRaw(
    InterpreterPool pool,
    cv.Mat crop,
    int inputSize,
    int outputLen,
  ) async {
    return pool.withInterpreter((interpreter, isolateInterpreter) async {
      final inputTensor = createNHWCTensor4D(inputSize, inputSize);
      final outputBuffer = allocTensorShape([1, outputLen]);

      final rgb = ImageUtils.matToFloat32(crop);
      fillNHWC4D(rgb, inputTensor, inputSize, inputSize);

      final outputs = {0: outputBuffer};
      if (isolateInterpreter != null) {
        await isolateInterpreter.runForMultipleInputs([inputTensor], outputs);
      } else {
        interpreter.runForMultipleInputs([inputTensor], outputs);
      }

      final raw = (outputBuffer as List)[0] as List;
      return List<double>.generate(
        outputLen,
        (i) => (raw[i] as double).clamp(0.0, 1.0),
      );
    });
  }

  /// Releases native resources held by all three interpreter pools.
  void dispose() {
    _pool256?.dispose();
    _pool320?.dispose();
    _pool384?.dispose();
  }
}
