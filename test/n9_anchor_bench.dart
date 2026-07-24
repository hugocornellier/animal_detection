// N9 before/after harness: real model, real images, detections + timing.
//
// Not a CI test (no `_test.dart` suffix, so `flutter test` will not collect it).
// Run explicitly, once per code revision, and diff the two JSON artifacts:
//
//   flutter test test/n9_anchor_bench.dart          # writes build/n9_bench.json
//   git stash && flutter test test/n9_anchor_bench.dart
//
// Detections are deterministic for a fixed model + image, so the accuracy
// comparison across the two runs is exact. Timing is reported as a
// distribution because it is not.

import 'dart:convert';
import 'dart:io';

import 'package:animal_detection/src/models/animal_body_detector.dart';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

/// Timed iterations per image. Override with N9_RUNS.
final int kRuns = int.tryParse(Platform.environment['N9_RUNS'] ?? '') ?? 100;

/// Untimed iterations before measuring, to settle JIT and allocator state.
const int kWarmup = 5;

const String kImageDir = 'example/integration_test/test_images';
const String kOutPath = 'build/n9_bench.json';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('N9 anchor/decode benchmark over real images', () async {
    final dir = Directory('${Directory.current.path}/$kImageDir');
    expect(dir.existsSync(), isTrue, reason: 'missing $kImageDir');

    final images = dir
        .listSync()
        .whereType<File>()
        .where((f) => f.path.endsWith('.jpg') || f.path.endsWith('.png'))
        .toList()
      ..sort((a, b) => a.path.compareTo(b.path));
    expect(images, isNotEmpty, reason: 'no test images found');

    final detector = AnimalBodyDetector();
    await detector.initialize(
      const PerformanceConfig(),
      useIsolateInterpreter: false,
    );

    final results = <String, dynamic>{};

    for (final file in images) {
      final name = file.uri.pathSegments.last;
      final mat = cv.imread(file.path);
      expect(mat.isEmpty, isFalse, reason: 'failed to decode $name');

      for (int i = 0; i < kWarmup; i++) {
        await detector.detect(mat);
      }

      // Detections are deterministic; capture once, at full precision.
      final dets = await detector.detect(mat);
      final detJson = <Map<String, dynamic>>[
        for (final (box, score) in dets)
          <String, dynamic>{
            'l': box.left,
            't': box.top,
            'r': box.right,
            'b': box.bottom,
            'score': score,
          },
      ];

      final timings = <double>[];
      for (int i = 0; i < kRuns; i++) {
        final sw = Stopwatch()..start();
        await detector.detect(mat);
        sw.stop();
        timings.add(sw.elapsedMicroseconds / 1000.0);
      }
      timings.sort();

      double pct(double p) =>
          timings[(timings.length * p).clamp(0, timings.length - 1).toInt()];

      results[name] = <String, dynamic>{
        'width': mat.cols,
        'height': mat.rows,
        'detectionCount': dets.length,
        'detections': detJson,
        'timingMs': <String, dynamic>{
          'runs': timings.length,
          'min': timings.first,
          'p50': pct(0.50),
          'p90': pct(0.90),
          'max': timings.last,
          'mean': timings.reduce((a, b) => a + b) / timings.length,
        },
      };

      // ignore: avoid_print
      print('$name  dets=${dets.length}  '
          'p50=${pct(0.50).toStringAsFixed(2)}ms  '
          'min=${timings.first.toStringAsFixed(2)}ms');

      mat.dispose();
    }

    detector.dispose();

    final out = File('${Directory.current.path}/$kOutPath');
    out.parent.createSync(recursive: true);
    out.writeAsStringSync(
      const JsonEncoder.withIndent('  ').convert(<String, dynamic>{
        'runsPerImage': kRuns,
        'images': results,
      }),
    );
    // ignore: avoid_print
    print('\nwrote ${out.path}');
  }, timeout: const Timeout(Duration(minutes: 20)));
}
