import 'dart:async';
import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/native.dart';
import '../types.dart';
import '../util/image_utils.dart';
import '../util/input_shape.dart';

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

  /// Pool size, retained so the CompiledModel path can build a matching pool.
  final int poolSize;

  final InterpreterPool _pool;

  /// CompiledModel pool, used instead of [_pool] when initialized with
  /// `useCompiledModel: true`. Unused in interpreter mode, which remains the
  /// default and the verified path.
  final CompiledModelPool _compiledPool = CompiledModelPool();

  bool _useCompiled = false;

  /// Reusable input tensors, one per pool slot, keyed by the slot's
  /// interpreter. Avoids reallocating `inputSize * inputSize * 3` floats on
  /// every frame while keeping concurrent slots isolated from each other.
  final Map<Interpreter, Float32List> _inputBuffers = {};

  /// Reusable flat output tensors, one per pool slot.
  final Map<Interpreter, Float32List> _outputBuffers = {};

  /// Creates a landmark model runner.
  LandmarkModelRunnerBase({
    required this.inputSize,
    required this.numLandmarks,
    required this.modelPath,
    this.poolSize = 1,
  }) : _pool = InterpreterPool(poolSize: poolSize);

  /// Initializes the model from Flutter assets.
  Future<void> initialize(
    PerformanceConfig performanceConfig, {
    bool useIsolateInterpreter = true,
  }) async {
    final path = modelPath;
    await _pool.initialize(
      (options, _) async {
        final interpreter = await Interpreter.fromAsset(path, options: options);
        assertSquareInputSize(
            interpreter, inputSize, 'LandmarkModelRunnerBase');
        interpreter.resizeInputTensor(0, [1, inputSize, inputSize, 3]);
        interpreter.allocateTensors();
        return interpreter;
      },
      performanceConfig: performanceConfig,
      useIsolateInterpreter: useIsolateInterpreter,
    );
  }

  /// Initializes the model from pre-loaded bytes (for isolate use).
  Future<void> initializeFromBuffer(
    Uint8List bytes,
    PerformanceConfig performanceConfig, {
    bool useIsolateInterpreter = true,
  }) async {
    await _pool.initialize(
      (options, _) async {
        final interpreter = Interpreter.fromBuffer(bytes, options: options);
        assertSquareInputSize(
            interpreter, inputSize, 'LandmarkModelRunnerBase');
        interpreter.resizeInputTensor(0, [1, inputSize, inputSize, 3]);
        interpreter.allocateTensors();
        return interpreter;
      },
      performanceConfig: performanceConfig,
      useIsolateInterpreter: useIsolateInterpreter,
    );
  }

  /// Initializes a pool of LiteRT Next [CompiledModel] instances instead of an
  /// [InterpreterPool].
  ///
  /// Opt-in and off by default: the interpreter path is what this package's
  /// benchmarks and integration suites cover. Provided for parity with
  /// face_detection_tflite, pose_detection and hand_detection, which all plumb
  /// CompiledModel as an alternative backend and all default it off.
  ///
  /// [forceCpu] pins each model to CPU. Otherwise the default accelerator set
  /// attempts GPU and falls back to CPU, reporting via [onGpuFallback].
  Future<void> initializeCompiledFromBuffer(
    Uint8List bytes, {
    bool forceCpu = false,
    Set<Accelerator> accelerators = const {Accelerator.cpu},
    Precision precision = Precision.fp32,
    void Function(Object error)? onGpuFallback,
  }) async {
    _compiledPool.initialize(
      poolSize: poolSize,
      inputFloats: inputSize * inputSize * 3,
      create: () => compiledModelFromBufferAuto(
        bytes,
        accelerators: accelerators,
        precision: precision,
        forceCpu: forceCpu,
        onGpuFallback: onGpuFallback,
      ),
    );
    _useCompiled = true;
  }

  /// Runs landmark prediction and returns denormalized (x, y) coordinate pairs.
  ///
  /// The returned list has [numLandmarks] entries, each a (x, y) record in
  /// original image pixel coordinates (mapped back via [meta]).
  Future<List<(double, double)>> predictRaw(
    cv.Mat crop,
    CropMetadata meta,
  ) async {
    if (_useCompiled) return _predictCompiled(crop, meta);
    return _pool.withInterpreter((interpreter, isolateInterpreter) async {
      // Convert straight into a flat Float32List and hand TFLite its
      // ByteBuffer, rather than filling a nested List<List<List<List<double>>>>.
      // The per-pixel Dart loop plus boxed fill measured 10.5 ms at 384x384 in
      // profile mode; the SIMD path into a reused flat buffer is 0.08 ms.
      //
      // The buffer is keyed on the interpreter because each pool slot holds a
      // distinct instance and is checked out exclusively for the duration of
      // this callback, so slots never share a buffer.
      final rgb = _inputBuffers[interpreter] = ImageUtils.matToFloat32Simd(
        crop,
        buffer: _inputBuffers[interpreter],
      );

      final out = _outputBuffers[interpreter] ??= Float32List(numLandmarks * 2);
      final outputs = {0: out.buffer};
      if (isolateInterpreter != null) {
        await isolateInterpreter.runForMultipleInputs([rgb.buffer], outputs);
      } else {
        interpreter.runForMultipleInputs([rgb.buffer], outputs);
      }

      final coords = <(double, double)>[];

      for (int i = 0; i < numLandmarks; i++) {
        final xNorm = out[i * 2].clamp(0.0, 1.0);
        final yNorm = out[i * 2 + 1].clamp(0.0, 1.0);
        coords.add(
            (xNorm * meta.cropW + meta.cx1, yNorm * meta.cropH + meta.cy1));
      }

      return coords;
    });
  }

  /// CompiledModel variant of [predictRaw]. `run` returns freshly allocated
  /// output lists rather than filling caller buffers, so there is no output
  /// buffer to reuse here; the per-slot input buffer comes from the pool.
  Future<List<(double, double)>> _predictCompiled(
    cv.Mat crop,
    CropMetadata meta,
  ) async {
    return _compiledPool.withModel((model, input) async {
      ImageUtils.matToFloat32Simd(crop, buffer: input);
      // runAsync, matching pose_detection: the blocking native call runs on a
      // per-model helper isolate, and concurrent calls are serialized against
      // the model's shared native I/O buffers.
      final List<Float32List> out = await model.runAsync([input]);
      final Float32List raw = out[0];
      final coords = <(double, double)>[];
      for (int i = 0; i < numLandmarks; i++) {
        final xNorm = raw[i * 2].clamp(0.0, 1.0);
        final yNorm = raw[i * 2 + 1].clamp(0.0, 1.0);
        coords.add(
            (xNorm * meta.cropW + meta.cx1, yNorm * meta.cropH + meta.cy1));
      }
      return coords;
    });
  }

  /// Releases native resources.
  void dispose() {
    _pool.dispose();
    _compiledPool.dispose();
  }
}
