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

double _boxDelta(BoundingBox a, BoundingBox b) => <double>[
      (a.left - b.left).abs(),
      (a.top - b.top).abs(),
      (a.right - b.right).abs(),
      (a.bottom - b.bottom).abs(),
    ].reduce((x, y) => x > y ? x : y);

void _expectParity(
  List<Animal> expected,
  List<Animal> actual,
  String label,
) {
  expect(actual.length, expected.length, reason: '$label animal count');
  for (var i = 0; i < expected.length; i++) {
    final a = expected[i];
    final b = actual[i];
    final box = _boxDelta(a.boundingBox, b.boundingBox);
    debugPrint(
      'PARITY $label [$i] species=${b.species} '
      'score=${b.score.toStringAsFixed(4)} '
      'worst box delta=${box.toStringAsFixed(2)}px '
      'pose=${b.pose?.landmarks.length}',
    );

    expect(b.species, a.species, reason: '$label species[$i]');
    expect(b.breed, a.breed, reason: '$label breed[$i]');
    expect(b.imageWidth, a.imageWidth, reason: '$label image width[$i]');
    expect(b.imageHeight, a.imageHeight, reason: '$label image height[$i]');
    expect(b.speciesConfidence, isNotNull);
    expect(a.speciesConfidence, isNotNull);
    expect(
      (b.speciesConfidence! - a.speciesConfidence!).abs(),
      lessThan(0.02),
      reason: '$label species confidence[$i]',
    );
    expect(
      (b.score - a.score).abs(),
      lessThan(0.02),
      reason: '$label body score[$i]',
    );
    expect(box, lessThan(10.0), reason: '$label body box[$i]');

    expect(b.pose, isNotNull, reason: '$label pose[$i]');
    final expectedPose = a.pose!.landmarks;
    final actualPose = b.pose!.landmarks;
    expect(actualPose.length, expectedPose.length, reason: '$label pose count');
    for (var j = 0; j < expectedPose.length; j++) {
      expect(actualPose[j].type, expectedPose[j].type);
      expect(
        (actualPose[j].confidence - expectedPose[j].confidence).abs(),
        lessThan(0.02),
        reason: '$label pose confidence[$i][$j]',
      );
      expect(
        (actualPose[j].x - expectedPose[j].x).abs(),
        lessThan(10.0),
        reason: '$label pose x[$i][$j]',
      );
      expect(
        (actualPose[j].y - expectedPose[j].y).abs(),
        lessThan(10.0),
        reason: '$label pose y[$i][$j]',
      );
    }
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('both backends produce the same detections', (tester) async {
    final itp = await _run(useCompiledModel: false);
    final cm = await _run(useCompiledModel: true);
    final cmCpu = await _run(useCompiledModel: true, forceCpu: true);
    expect(itp, isNotEmpty, reason: 'interpreter found nothing');
    _expectParity(itp, cmCpu, 'CompiledModel CPU');
    _expectParity(itp, cm, 'CompiledModel GPU+CPU');
  }, timeout: const Timeout(Duration(minutes: 6)));
}
