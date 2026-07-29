// Does useCompiledModel: true actually work end to end?
//
// The compiled path compiles and analyzes cleanly, which is exactly why it
// needs exercising: the question is whether every stage has the state it needs
// at detect() time, not whether it type-checks.
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

Future<List<Animal>> _run({required bool useCompiledModel}) async {
  final ssd = (await rootBundle.load(_ssd)).buffer.asUint8List();
  final cls = (await rootBundle.load(_cls)).buffer.asUint8List();
  final map = await rootBundle.loadString(_map);
  final pose = (await rootBundle.load(_pose)).buffer.asUint8List();

  final detector = AnimalDetector(enablePose: true);
  await detector.initializeFromBuffers(
    bodyDetectorBytes: ssd,
    classifierBytes: cls,
    speciesMappingJson: map,
    poseModelBytes: pose,
    useIsolateInterpreter: false,
    useCompiledModel: useCompiledModel,
  );
  try {
    final data = await rootBundle.load(_image);
    final mat = cv.imdecode(data.buffer.asUint8List(), cv.IMREAD_COLOR);
    try {
      return await detector.detectFromMat(
        mat,
        imageWidth: mat.cols,
        imageHeight: mat.rows,
      );
    } finally {
      mat.dispose();
    }
  } finally {
    await detector.dispose();
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('interpreter backend detects', (tester) async {
    final r = await _run(useCompiledModel: false);
    debugPrint('CMB interpreter: ${r.length} animal(s)'
        '${r.isEmpty ? '' : ' species=${r.first.species} '
            'score=${r.first.score.toStringAsFixed(4)} '
            'pose=${r.first.pose?.landmarks.length}'}');
    expect(r, isNotEmpty);
  }, timeout: const Timeout(Duration(minutes: 5)));

  testWidgets('compiled backend detects', (tester) async {
    List<Animal> r;
    try {
      r = await _run(useCompiledModel: true);
    } catch (e) {
      debugPrint('CMB compiled FAILED: $e');
      rethrow;
    }
    debugPrint('CMB compiled: ${r.length} animal(s)'
        '${r.isEmpty ? '' : ' species=${r.first.species} '
            'score=${r.first.score.toStringAsFixed(4)} '
            'pose=${r.first.pose?.landmarks.length}'}');
    expect(r, isNotEmpty);
  }, timeout: const Timeout(Duration(minutes: 5)));
}
