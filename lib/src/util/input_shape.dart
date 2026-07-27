import 'package:flutter_litert/native.dart';

/// Reads the square spatial dimension of [interpreter]'s input tensor 0.
///
/// Returns null when the shape is not the `[1, N, N, 3]` form the model classes
/// in this package assume.
int? probeSquareInputSize(Interpreter interpreter) {
  final List<int> shape = interpreter.getInputTensor(0).shape;
  if (shape.length != 4) return null;
  if (shape[1] != shape[2]) return null;
  if (shape[3] != 3) return null;
  return shape[1];
}

/// Throws if [interpreter]'s input tensor does not match `[1, expected,
/// expected, 3]`.
///
/// A configured size that disagrees with the bundled model is otherwise
/// invisible: `Interpreter.resizeInputTensor` accepts a shape the model was not
/// trained for without reporting an error, and inference then returns finite
/// but meaningless values. Feeding a 256px landmark model at 384px measured
/// NME_IOD 67.3 against a correct 3.5, while every output stayed in range and
/// no exception was raised.
///
/// [label] identifies the model in the message, since the failure is otherwise
/// hard to attribute across a six-stage pipeline.
void assertSquareInputSize(
  Interpreter interpreter,
  int expected,
  String label,
) {
  final int? actual = probeSquareInputSize(interpreter);
  if (actual == expected) return;
  final List<int> shape = interpreter.getInputTensor(0).shape;
  throw StateError(
    '$label: the bundled model declares input shape $shape, but this runner is '
    'configured for [1, $expected, $expected, 3]. Resizing to a shape the '
    'model was not trained for produces meaningless coordinates rather than an '
    'error, so it is rejected here instead. Either bundle a ${expected}px '
    'model or configure the runner for ${actual ?? "the declared"}px.',
  );
}
