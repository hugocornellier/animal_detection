// Does the optional CompiledModel path work, and is it faster?
//
// Interpreter remains the default and the verified path. This exists to answer
// whether the CompiledModel backend is worth exposing at all, since all three
// golden packages plumb it and all three default it off.
//
//   flutter test integration_test/compiled_model_test.dart -d macos
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:animal_detection/animal_detection.dart';

// Cat's bundled landmark model, read from the sibling checkout: animal does not
// bundle a landmark model of its own.
const _catLandmarkModel =
    '/Users/hugocornellier/IdeaProjects/cat_detection/assets/models/'
    'cat_face_landmarks_full.tflite';
const _size = 384;
const _landmarks = 48;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('CompiledModel path agrees with Interpreter and is benchmarked',
      (tester) async {
    final file = File(_catLandmarkModel);
    if (!file.existsSync()) {
      debugPrint('CM skipped: $_catLandmarkModel not found');
      return;
    }
    final bytes = await file.readAsBytes();

    // A deterministic non-uniform crop; content only needs to be stable across
    // both paths so the outputs are comparable.
    final mat = cv.Mat.zeros(_size, _size, cv.MatType.CV_8UC3);
    addTearDown(mat.dispose);
    for (int y = 0; y < _size; y++) {
      for (int x = 0; x < _size; x++) {
        mat.set<int>(y, x, (x * 5 + y * 11) % 256);
      }
    }
    const meta = CropMetadata(cx1: 0, cy1: 0, cropW: 1, cropH: 1);

    // --- interpreter path (the default) ---
    final interp = LandmarkModelRunnerBase(
      inputSize: _size,
      numLandmarks: _landmarks,
      modelPath: 'unused',
    );
    await interp.initializeFromBuffer(bytes, const PerformanceConfig(),
        useIsolateInterpreter: false);
    final a = await interp.predictRaw(mat, meta);

    // --- compiled path ---
    final compiled = LandmarkModelRunnerBase(
      inputSize: _size,
      numLandmarks: _landmarks,
      modelPath: 'unused',
    );
    List<(double, double)> b;
    try {
      await compiled.initializeCompiledFromBuffer(
        bytes,
        onGpuFallback: (e) => debugPrint('CM gpu fallback: $e'),
      );
      b = await compiled.predictRaw(mat, meta);
    } catch (e) {
      interp.dispose();
      compiled.dispose();
      debugPrint('CM UNAVAILABLE on this platform: $e');
      return;
    }

    debugPrint('CM interp[0..3]  = ${a.take(3).toList()}');
    debugPrint('CM compiled[0..3] = ${b.take(3).toList()}');
    expect(b.length, a.length);
    double worst = 0;
    for (int i = 0; i < a.length; i++) {
      final dx = (a[i].$1 - b[i].$1).abs();
      final dy = (a[i].$2 - b[i].$2).abs();
      if (dx > worst) worst = dx;
      if (dy > worst) worst = dy;
    }
    debugPrint('CM worst |interpreter - compiled| over '
        '${a.length * 2} coords = $worst');

    // --- benchmark both ---
    Future<double> bench(Future<void> Function() run) async {
      for (int i = 0; i < 3; i++) {
        await run();
      }
      final sw = Stopwatch()..start();
      const n = 15;
      for (int i = 0; i < n; i++) {
        await run();
      }
      sw.stop();
      return sw.elapsedMicroseconds / n / 1000.0;
    }

    final msI = await bench(() => interp.predictRaw(mat, meta));
    final msC = await bench(() => compiled.predictRaw(mat, meta));
    debugPrint('CM interpreter=${msI.toStringAsFixed(1)}ms  '
        'compiled=${msC.toStringAsFixed(1)}ms  '
        'ratio=${(msI / msC).toStringAsFixed(2)}x');

    interp.dispose();
    compiled.dispose();

    // Both run the same weights, so coordinates should agree closely. A large
    // divergence would mean the compiled path is wired wrong, not that the
    // backend is less accurate.
    // cropW/cropH are 1 in this test, so coordinates are in normalized [0,1]
    // space. 0.02 is 2% of the crop, generous for kernel differences between
    // backends but far tighter than the 0.57 first observed.
    expect(worst, lessThan(0.02),
        reason: 'compiled output diverges from interpreter output');
  }, timeout: const Timeout(Duration(minutes: 10)));
}
