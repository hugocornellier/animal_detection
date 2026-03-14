import 'dart:typed_data';
import 'package:flutter_litert/flutter_litert.dart';

/// Base class for TFLite model classes that use a single interpreter.
///
/// Provides shared interpreter initialization, inference dispatch, and disposal
/// for the body detector, species classifier, pose estimator, and face localizer.
abstract class SingleInterpreterModel {
  Interpreter? _interpreter;
  IsolateInterpreter? _isolateInterpreter;
  Delegate? _delegate;

  /// Exposes the underlying interpreter for subclasses that need to query
  /// tensor metadata (e.g. shapes) after initialization.
  Interpreter? get interpreter => _interpreter;

  /// Initializes the interpreter from a Flutter asset path.
  Future<void> initInterpreterFromAsset(
    String assetPath,
    PerformanceConfig config,
  ) async {
    final (options, delegate) = InterpreterFactory.create(config);
    _delegate = delegate;
    _interpreter = await Interpreter.fromAsset(assetPath, options: options);
    _isolateInterpreter = await InterpreterFactory.createIsolateIfNeeded(
      _interpreter!,
      _delegate,
    );
  }

  /// Initializes the interpreter from pre-loaded model bytes.
  Future<void> initInterpreterFromBuffer(
    Uint8List bytes,
    PerformanceConfig config,
  ) async {
    final (options, delegate) = InterpreterFactory.create(config);
    _delegate = delegate;
    _interpreter = Interpreter.fromBuffer(bytes, options: options);
    _isolateInterpreter = await InterpreterFactory.createIsolateIfNeeded(
      _interpreter!,
      _delegate,
    );
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
  }
}
