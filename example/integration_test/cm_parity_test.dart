// Backend parity: do the Interpreter and CompiledModel paths agree?
//
// This is the real check on AnimalBodyDetector's derived output shapes. The
// SSD decode groups heads by level, and CompiledModel reports byte sizes
// rather than shapes, so the shapes are derived from element counts. A wrong
// derivation mis-groups levels and produces different boxes, which shows up
// here rather than as an error.
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:animal_detection/animal_detection.dart';

const _image = 'integration_test/test_images/cat.jpg';
const _ssd =
    'packages/animal_detection/assets/models/superanimal_ssdlite_float16.tflite';
const _cls =
    'packages/animal_detection/assets/models/species_classifier_float16.tflite';
const _map = 'packages/animal_detection/assets/models/species_mapping.json';
const _pose =
    'packages/animal_detection/assets/models/superanimal_rtmpose_s_float16.tflite';

Future<List<Animal>> _run({
  required bool useCompiledModel,
  bool forceCpu = false,
}) async {
  final detector = AnimalDetector(enablePose: true);
  await detector.initializeFromBuffers(
    bodyDetectorBytes: (await rootBundle.load(_ssd)).buffer.asUint8List(),
    classifierBytes: (await rootBundle.load(_cls)).buffer.asUint8List(),
    speciesMappingJson: await rootBundle.loadString(_map),
    poseModelBytes: (await rootBundle.load(_pose)).buffer.asUint8List(),
    useIsolateInterpreter: false,
    useCompiledModel: useCompiledModel,
    compiledForceCpu: forceCpu,
  );
  try {
    final data = await rootBundle.load(_image);
    final mat = cv.imdecode(data.buffer.asUint8List(), cv.IMREAD_COLOR);
    try {
      return await detector.detectFromMat(mat,
          imageWidth: mat.cols, imageHeight: mat.rows);
    } finally {
      mat.dispose();
    }
  } finally {
    await detector.dispose();
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('both backends produce the same detections', (tester) async {
    final itp = await _run(useCompiledModel: false);
    final cm = await _run(useCompiledModel: true);
    final cmCpu = await _run(useCompiledModel: true, forceCpu: true);
    debugPrint('PARITY compiled cpu-only: '
        '${cmCpu.length} animal(s) species=${cmCpu.isEmpty ? "-" : cmCpu.first.species}');

    debugPrint('PARITY interpreter: ${itp.length} animal(s)');
    debugPrint('PARITY compiled   : ${cm.length} animal(s)');
    expect(itp, isNotEmpty, reason: 'interpreter found nothing');
    expect(cm.length, itp.length, reason: 'backends disagree on animal count');

    for (int i = 0; i < itp.length; i++) {
      final a = itp[i];
      final b = cm[i];
      final box = <double>[
        (a.boundingBox.left - b.boundingBox.left).abs(),
        (a.boundingBox.top - b.boundingBox.top).abs(),
        (a.boundingBox.right - b.boundingBox.right).abs(),
        (a.boundingBox.bottom - b.boundingBox.bottom).abs(),
      ].reduce((x, y) => x > y ? x : y);
      debugPrint('PARITY [$i] species itp=${a.species} cm=${b.species} | '
          'score ${a.score.toStringAsFixed(4)} vs ${b.score.toStringAsFixed(4)} | '
          'worst box delta ${box.toStringAsFixed(2)}px | '
          'pose ${a.pose?.landmarks.length} vs ${b.pose?.landmarks.length}');

      expect(b.species, a.species);

      // Compare keypoint VALUES, not just counts. An earlier version of this
      // test checked only length and would have passed while rtmpose diverged
      // by 0.238 under the mixed GPU/CPU partition.
      final ap = a.pose!.landmarks;
      final bp = b.pose!.landmarks;
      double worstKp = 0;
      for (int k = 0; k < ap.length; k++) {
        final dx = (ap[k].x - bp[k].x).abs();
        final dy = (ap[k].y - bp[k].y).abs();
        if (dx > worstKp) worstKp = dx;
        if (dy > worstKp) worstKp = dy;
      }
      debugPrint('PARITY [$i] worst pose keypoint delta '
          '${worstKp.toStringAsFixed(2)}px');
      expect(worstKp, lessThan(10.0), reason: 'pose keypoints diverge');
      // Metal and CPU kernels differ slightly; a mis-derived shape would move
      // boxes by hundreds of pixels, not a few.
      expect(box, lessThan(10.0), reason: 'bounding boxes diverge');
      expect(b.pose?.landmarks.length, a.pose?.landmarks.length);
    }
  }, timeout: const Timeout(Duration(minutes: 6)));
}
