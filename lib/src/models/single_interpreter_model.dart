import 'dart:typed_data';
import 'package:flutter_litert/native.dart';

/// Base class for TFLite model classes that use a single interpreter.
///
/// Provides shared interpreter initialization, inference dispatch, and disposal
/// for the body detector, species classifier, pose estimator, and face localizer.
abstract class SingleInterpreterModel {
  Interpreter? _interpreter;
  IsolateInterpreter? _isolateInterpreter;
  Delegate? _delegate;

  CompiledModel? _compiled;

  /// True when this model was initialized onto the LiteRT Next
  /// [CompiledModel] backend rather than the default [Interpreter].
  bool get usingCompiledModel => _compiled != null;

  /// Reusable flat input buffer for the CompiledModel path, sized from the
  /// model's own reported input byte size rather than assumed.
  Float32List? _compiledInput;

  /// Input buffer to fill before calling [runCompiled]. Null in interpreter
  /// mode.
  Float32List? get compiledInput => _compiledInput;

  /// Byte size of each output tensor on the CompiledModel backend, or null in
  /// interpreter mode.
  ///
  /// CompiledModel reports sizes but not shapes, so subclasses whose decode
  /// needs shapes must derive them from these.
  List<int>? get compiledOutputByteSizes => _compiled?.outputByteSizes;

  /// Exposes the underlying interpreter for subclasses that need to query
  /// tensor metadata (e.g. shapes) after initialization.
  Interpreter? get interpreter => _interpreter;

  /// Initializes the interpreter from a Flutter asset path.
  Future<void> initInterpreterFromAsset(
    String assetPath,
    PerformanceConfig config, {
    bool useIsolateInterpreter = true,
  }) async {
    final (options, delegate) = InterpreterFactory.create(config);
    _delegate = delegate;
    _interpreter = await Interpreter.fromAsset(assetPath, options: options);
    _isolateInterpreter = await InterpreterFactory.createIsolateIfNeeded(
      _interpreter!,
      _delegate,
      useIsolateInterpreter: useIsolateInterpreter,
    );
  }

  /// Initializes the interpreter from pre-loaded model bytes.
  Future<void> initInterpreterFromBuffer(
    Uint8List bytes,
    PerformanceConfig config, {
    bool useIsolateInterpreter = true,
  }) async {
    final (options, delegate) = InterpreterFactory.create(config);
    _delegate = delegate;
    _interpreter = Interpreter.fromBuffer(bytes, options: options);
    _isolateInterpreter = await InterpreterFactory.createIsolateIfNeeded(
      _interpreter!,
      _delegate,
      useIsolateInterpreter: useIsolateInterpreter,
    );
  }

  /// Initializes the LiteRT Next [CompiledModel] backend from model bytes.
  ///
  /// Opt-in; the [Interpreter] path remains the default. Defaults follow the
  /// benchmarks in flutter_litert's test/benchmark/RESULTS.md:
  ///
  /// Defaults to CPU-only, deliberately, despite flutter_litert's
  /// RESULTS.md calling the GPU-then-CPU set "the production config". That
  /// guidance is about compiling successfully, not numerical correctness. The
  /// mixed partition diverges from the CPU reference by 10.08 on
  /// species_classifier and 0.238 on superanimal_rtmpose, to six identical
  /// decimal places across Metal (macOS, iOS) and Vulkan (Linux) — so it is
  /// deterministic, not GPU rounding. On this pipeline that flipped a cat into
  /// unknown_animal while still reporting score 1.0000. Pass
  /// `accelerators: {Accelerator.gpu, Accelerator.cpu}` only with a parity
  /// check against the interpreter for your own models.
  /// - [precision] defaults to fp32. Every model in this package emits
  ///   coordinates or boxes, and those results measure fp16 GPU paths
  ///   diverging ~9e-3 from the CPU reference against ~3e-6 for fp32, noting
  ///   that "models emitting pixel-space coordinates (landmarks) should prefer
  ///   fp32". fp32 costs more at compile time, which is paid once here.
  Future<void> initCompiledFromBuffer(
    Uint8List bytes, {
    bool forceCpu = false,
    Set<Accelerator> accelerators = const {Accelerator.cpu},
    Precision precision = Precision.fp32,
    void Function(Object error)? onGpuFallback,
  }) async {
    CompiledModel create(
      Set<Accelerator> requestedAccelerators, {
      required bool requestedForceCpu,
    }) =>
        compiledModelFromBufferAuto(
          bytes,
          accelerators: requestedAccelerators,
          precision: precision,
          forceCpu: requestedForceCpu,
          onGpuFallback: onGpuFallback,
        );

    var model = create(accelerators, requestedForceCpu: forceCpu);
    var verification = verifyCompiledModel(bytes, model);
    if (!verification.agrees &&
        !forceCpu &&
        accelerators.contains(Accelerator.gpu)) {
      model.close();
      onGpuFallback?.call(
        StateError(
          'CompiledModel GPU verification failed; retrying on CPU: '
          '$verification',
        ),
      );
      model = create(
        const {Accelerator.cpu},
        requestedForceCpu: true,
      );
      verification = verifyCompiledModel(bytes, model);
    }
    if (!verification.agrees) {
      model.close();
      throw StateError(
        'CompiledModel backend verification failed: $verification',
      );
    }
    _compiled = model;
    _compiledInput = Float32List(model.inputByteSizes[0] ~/ 4);
  }

  /// Runs the CompiledModel backend, returning its output tensors.
  ///
  /// Unlike [runInference], the backend allocates and returns outputs rather
  /// than filling caller-supplied buffers, so there is nothing to reuse on the
  /// output side here.
  Future<List<Float32List>> runCompiled() async {
    final model = _compiled;
    if (model == null) {
      throw StateError('runCompiled called without a CompiledModel backend.');
    }
    return model.runAsync([_compiledInput!]);
  }

  /// Runs inference with multiple inputs and named output buffers.
  Future<void> runInference(
    List<Object> inputs,
    Map<int, Object> outputs,
  ) async {
    if (_isolateInterpreter != null) {
      await _isolateInterpreter!.runForMultipleInputs(inputs, outputs);
    } else {
      _interpreter!.runForMultipleInputs(inputs, outputs);
    }
  }

  /// Runs inference with a single input and output tensor.
  Future<void> runInferenceSingle(Object input, Object output) async {
    if (_isolateInterpreter != null) {
      await _isolateInterpreter!.run(input, output);
    } else {
      _interpreter!.run(input, output);
    }
  }

  /// Releases the interpreter and associated native resources.
  void dispose() {
    _isolateInterpreter?.close();
    _interpreter?.close();
    _delegate?.delete();
    _compiled?.close();
    _compiled = null;
    _compiledInput = null;
  }
}
