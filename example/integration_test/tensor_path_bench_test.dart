// Attribution harness for the input-tensor path.
//
// Every model class in this package feeds TFLite the same way:
//
//     ImageUtils.matToFloat32(mat)          // per-pixel Dart loop, BGR->RGB /255
//     createNHWCTensor4D(size, size)        // List<List<List<List<double>>>>
//     fillNHWC4D(rgb, tensor, size, size)   // boxed-double fill
//
// The golden-standard packages (face_detection_tflite, pose_detection,
// hand_detection) instead keep a reused flat Float32List and hand TFLite its
// ByteBuffer, explicitly to avoid "the boxed-double allocation that
// Tensor.copyTo performs when handed a nested List<List<double>> dst".
//
// This measures the current path against candidate replacements at each
// resolution the pipeline actually uses, in isolation from any model, so a
// single change can be attributed rather than estimated.
//
// Run in profile mode (AOT) -- debug massively inflates boxed-list work:
//
//   flutter drive --profile \
//     --driver=test_driver/integration_test.dart \
//     --target=integration_test/tensor_path_bench_test.dart -d macos
import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:animal_detection/animal_detection.dart';
import 'package:flutter_litert/native.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

/// Resolutions the shipped models actually run at.
const _sizes = <int, String>{
  224: 'localizer / species',
  256: 'rtmpose',
  320: 'ssdlite',
  384: 'landmarks',
};

const _warmup = 5;
const _iters = 40;

/// Median plus interquartile spread. A bare mean hid a 2x discrepancy in an
/// earlier harness, so report the spread and interleave the variants.
({double median, double p25, double p75}) _stats(List<double> xs) {
  final s = [...xs]..sort();
  double at(double q) => s[(q * (s.length - 1)).round()];
  return (median: at(0.5), p25: at(0.25), p75: at(0.75));
}

double _msPer(Stopwatch sw, int iters) =>
    sw.elapsedMicroseconds / iters / 1000.0;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('input tensor path: current vs candidates', (tester) async {
    debugPrint(
        'TBENCH iters=$_iters warmup=$_warmup (interleaved, median [p25-p75])');

    for (final entry in _sizes.entries) {
      final size = entry.key;
      final label = entry.value;

      // Synthetic BGR frame at the model's input resolution. Content is
      // irrelevant to conversion cost; only the pixel count matters.
      final mat = cv.Mat.zeros(size, size, cv.MatType.CV_8UC3);
      addTearDown(mat.dispose);

      // Reusable destinations for the candidate paths.
      final tensor = createNHWCTensor4D(size, size);
      final flatBuf = Float32List(size * size * 3);

      final current = <double>[];
      final flatReuse = <double>[];
      final simd = <double>[];
      final zeroCopy = <double>[];
      final simdNoCvt = <double>[];

      for (int i = 0; i < _warmup; i++) {
        fillNHWC4D(ImageUtils.matToFloat32(mat), tensor, size, size);
      }

      // Interleaved so thermal drift hits all variants equally.
      for (int i = 0; i < _iters; i++) {
        // A: what ships today -- per-pixel loop, fresh boxed tensor, fill.
        var sw = Stopwatch()..start();
        final rgbA = ImageUtils.matToFloat32(mat);
        final tA = createNHWCTensor4D(size, size);
        fillNHWC4D(rgbA, tA, size, size);
        sw.stop();
        current.add(_msPer(sw, 1));

        // B: per-pixel loop, but reuse the boxed tensor (no realloc).
        sw = Stopwatch()..start();
        final rgbB = ImageUtils.matToFloat32(mat);
        fillNHWC4D(rgbB, tensor, size, size);
        sw.stop();
        flatReuse.add(_msPer(sw, 1));

        // C: OpenCV SIMD convert straight into a reused flat Float32List,
        // which is what the golden packages hand to TFLite as a ByteBuffer.
        sw = Stopwatch()..start();
        final rgb = cv.cvtColor(mat, cv.COLOR_BGR2RGB);
        final f32 = rgb.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);
        rgb.dispose();
        final n = size * size * 3;
        flatBuf.setRange(0, n, f32.data.buffer.asFloat32List(0, n));
        f32.dispose();
        sw.stop();
        simd.add(_msPer(sw, 1));

        // D: as C but skip the setRange copy -- view the converted Mat's own
        // buffer. Only valid if the Mat outlives the inference call.
        sw = Stopwatch()..start();
        final rgbD = cv.cvtColor(mat, cv.COLOR_BGR2RGB);
        final f32D = rgbD.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);
        rgbD.dispose();
        final viewD = f32D.data.buffer.asFloat32List(0, n);
        if (viewD.length != n) throw StateError('bad view');
        sw.stop();
        f32D.dispose();
        zeroCopy.add(_msPer(sw, 1));

        // E: how much of C is the BGR->RGB pass? convertTo only, no cvtColor.
        // Not a usable path (channel order wrong); measured to locate the cost.
        sw = Stopwatch()..start();
        final f32E = mat.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);
        flatBuf.setRange(0, n, f32E.data.buffer.asFloat32List(0, n));
        f32E.dispose();
        sw.stop();
        simdNoCvt.add(_msPer(sw, 1));
      }

      final a = _stats(current);
      final b = _stats(flatReuse);
      final c = _stats(simd);
      debugPrint('TBENCH ${size}px ($label)');
      debugPrint('TBENCH   A current (loop + fresh boxed tensor) '
          '${a.median.toStringAsFixed(3)} ms [${a.p25.toStringAsFixed(3)}-'
          '${a.p75.toStringAsFixed(3)}]');
      debugPrint('TBENCH   B loop + reused boxed tensor          '
          '${b.median.toStringAsFixed(3)} ms [${b.p25.toStringAsFixed(3)}-'
          '${b.p75.toStringAsFixed(3)}]  '
          'saves ${(a.median - b.median).toStringAsFixed(3)} ms');
      debugPrint('TBENCH   C SIMD -> reused flat Float32List     '
          '${c.median.toStringAsFixed(3)} ms [${c.p25.toStringAsFixed(3)}-'
          '${c.p75.toStringAsFixed(3)}]  '
          'saves ${(a.median - c.median).toStringAsFixed(3)} ms '
          '(${(a.median / c.median).toStringAsFixed(1)}x)');
      final d = _stats(zeroCopy);
      final e = _stats(simdNoCvt);
      debugPrint('TBENCH   D SIMD, no copy (view Mat buffer)    '
          '${d.median.toStringAsFixed(3)} ms [${d.p25.toStringAsFixed(3)}-'
          '${d.p75.toStringAsFixed(3)}]  '
          'vs C ${(c.median - d.median).toStringAsFixed(3)} ms');
      debugPrint('TBENCH   E convertTo only (no BGR->RGB pass)  '
          '${e.median.toStringAsFixed(3)} ms [${e.p25.toStringAsFixed(3)}-'
          '${e.p75.toStringAsFixed(3)}]  '
          'cvtColor costs ${(c.median - e.median).toStringAsFixed(3)} ms');
    }
  }, timeout: const Timeout(Duration(minutes: 10)));

  testWidgets('ImageUtils SIMD helpers match their per-pixel equivalents',
      (tester) async {
    // Guards the two helpers actually used by the model classes. The ImageNet
    // variant does a per-channel affine via scalar Mat ops, so it needs
    // proving independently of the plain /255 path.
    const size = 320;
    final mat = cv.Mat.zeros(size, size, cv.MatType.CV_8UC3);
    addTearDown(mat.dispose);
    for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++) {
        mat.set<int>(y, x, (x * 7 + y * 13) % 256);
      }
    }

    double worstOf(Float32List a, Float32List b) {
      expect(a.length, b.length);
      double w = 0;
      for (int i = 0; i < a.length; i++) {
        final d = (a[i] - b[i]).abs();
        if (d > w) w = d;
      }
      return w;
    }

    final plainWorst = worstOf(
      ImageUtils.matToFloat32Simd(mat),
      ImageUtils.matToFloat32(mat),
    );
    final imagenetWorst = worstOf(
      ImageUtils.matToFloat32ImageNetSimd(mat),
      ImageUtils.matToFloat32ImageNet(mat),
    );
    debugPrint(
        'TBENCH helper parity: plain=$plainWorst imagenet=$imagenetWorst');
    expect(plainWorst, lessThan(1e-6), reason: 'matToFloat32Simd mismatch');
    expect(imagenetWorst, lessThan(1e-5),
        reason: 'matToFloat32ImageNetSimd mismatch');

    // Buffer reuse must produce identical output to a fresh allocation, and
    // must actually reuse the instance handed in.
    final reusable = Float32List(size * size * 3);
    final returned = ImageUtils.matToFloat32Simd(mat, buffer: reusable);
    expect(identical(returned, reusable), isTrue,
        reason: 'supplied buffer of the right length should be reused');
    expect(worstOf(returned, ImageUtils.matToFloat32(mat)), lessThan(1e-6));

    // A wrong-length buffer must be rejected rather than corrupted.
    final tooSmall = Float32List(16);
    final grown = ImageUtils.matToFloat32Simd(mat, buffer: tooSmall);
    expect(identical(grown, tooSmall), isFalse);
    expect(grown.length, size * size * 3);
  });

  testWidgets('SIMD path produces the same values as the per-pixel loop',
      (tester) async {
    // A speed change must not move the numbers. Same convention as the
    // animal_detection 1.4.0 anchor-table change, which documented
    // bit-identical scores.
    const size = 384;
    final mat = cv.Mat.zeros(size, size, cv.MatType.CV_8UC3);
    addTearDown(mat.dispose);
    // Deterministic non-uniform content so the comparison is meaningful.
    for (int y = 0; y < size; y += 7) {
      for (int x = 0; x < size; x += 5) {
        mat.set<int>(y, x, (x * 3 + y) % 256);
      }
    }

    final loop = ImageUtils.matToFloat32(mat);

    final rgb = cv.cvtColor(mat, cv.COLOR_BGR2RGB);
    final f32 = rgb.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);
    rgb.dispose();
    final n = size * size * 3;
    final simd = Float32List(n)
      ..setRange(0, n, f32.data.buffer.asFloat32List(0, n));
    f32.dispose();

    expect(simd.length, loop.length);
    double worst = 0;
    for (int i = 0; i < n; i++) {
      final d = (simd[i] - loop[i]).abs();
      if (d > worst) worst = d;
    }
    debugPrint('TBENCH worst |SIMD - loop| over $n floats = $worst');
    expect(worst, lessThan(1e-6),
        reason: 'SIMD conversion must match the per-pixel loop');
  });
}
