// Does the static-shape re-export speed up the DEFAULT interpreter path?
//
// The dynamic tensor left by SoftArgmax2D's tf.shape() calls is why TFLite
// warns that static-shape-only delegates cannot cover the graph. If XNNPACK
// was being partially excluded, removing it should show up here. Bit-identical
// output is already established, so this measures speed only.
//
//   flutter drive --profile --driver=test_driver/integration_test.dart \
//     --target=integration_test/static_shape_perf_test.dart -d macos
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_litert/native.dart';

const _shipped =
    '/Users/hugocornellier/IdeaProjects/cat_detection/assets/models/'
    'cat_face_landmarks_full.tflite';
const _static = '/private/tmp/claude-501/'
    '-Users-hugocornellier-IdeaProjects-cat-detection-example/'
    '4c9115c2-76bf-490c-a607-ad32b6bb5b64/scratchpad/static/'
    'cat_static_384_float16.tflite';
const _size = 384;
const _warmup = 5;
const _iters = 30;

({double median, double p25, double p75}) _stats(List<double> xs) {
  final s = [...xs]..sort();
  double at(double q) => s[(q * (s.length - 1)).round()];
  return (median: at(0.5), p25: at(0.25), p75: at(0.75));
}

/// Times invoke() only, so preprocessing is out of the picture.
List<double> _bench(String path, PerformanceConfig cfg, Float32List input) {
  final bytes = File(path).readAsBytesSync();
  final (options, delegate) = InterpreterFactory.create(cfg);
  final itp = Interpreter.fromBuffer(bytes, options: options);
  itp.allocateTensors();
  final out = Float32List(96);
  final outs = <int, Object>{0: out.buffer};
  for (int i = 0; i < _warmup; i++) {
    itp.runForMultipleInputs([input.buffer], outs);
  }
  final samples = <double>[];
  for (int i = 0; i < _iters; i++) {
    final sw = Stopwatch()..start();
    itp.runForMultipleInputs([input.buffer], outs);
    sw.stop();
    samples.add(sw.elapsedMicroseconds / 1000.0);
  }
  itp.close();
  delegate?.delete();
  return samples;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('static re-export vs shipped, interpreter invoke',
      (tester) async {
    for (final f in [_shipped, _static]) {
      if (!File(f).existsSync()) {
        debugPrint('SSP FILE MISSING: $f');
        return;
      }
    }
    final input = Float32List(_size * _size * 3);
    for (int i = 0; i < input.length; i++) {
      input[i] = ((i * 13) % 255) / 255.0;
    }

    for (final cfg in <String, PerformanceConfig>{
      'xnnpack (auto on macOS)': const PerformanceConfig.xnnpack(),
      'disabled (plain CPU)': PerformanceConfig.disabled,
    }.entries) {
      // Interleaved so thermal drift hits both files equally.
      final a = <double>[];
      final b = <double>[];
      for (int round = 0; round < 3; round++) {
        a.addAll(_bench(_shipped, cfg.value, input));
        b.addAll(_bench(_static, cfg.value, input));
      }
      final sa = _stats(a);
      final sb = _stats(b);
      debugPrint('SSP ${cfg.key}');
      debugPrint('SSP   shipped (dynamic) ${sa.median.toStringAsFixed(2)} ms '
          '[${sa.p25.toStringAsFixed(2)}-${sa.p75.toStringAsFixed(2)}]');
      debugPrint('SSP   static  (fixed)   ${sb.median.toStringAsFixed(2)} ms '
          '[${sb.p25.toStringAsFixed(2)}-${sb.p75.toStringAsFixed(2)}]  '
          'delta ${(sb.median - sa.median).toStringAsFixed(2)} ms '
          '(${(sa.median / sb.median).toStringAsFixed(2)}x)');
    }
  }, timeout: const Timeout(Duration(minutes: 10)));
}
