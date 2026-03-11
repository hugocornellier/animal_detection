import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:animal_detection/animal_detection.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late AnimalDetector detector;

  setUpAll(() async {
    detector = AnimalDetector(
      enablePose: true,
      poseModel: AnimalPoseModel.rtmpose,
      performanceConfig: PerformanceConfig.disabled,
    );
    await detector.initialize();
  });

  tearDownAll(() async {
    await detector.dispose();
  });

  final testImages = [
    'cat',
    'fox',
    'bear',
    'rabbit',
    'horse',
    'zebra',
    'cow',
    'sheep',
    'deer',
  ];

  for (final species in testImages) {
    testWidgets('Detect $species', (tester) async {
      final ByteData data = await rootBundle.load(
        'integration_test/test_images/$species.jpg',
      );
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Animal> results = await detector.detect(bytes);

      print('=== $species ===');
      print('  Detections: ${results.length}');
      for (int i = 0; i < results.length; i++) {
        final a = results[i];
        print('  [$i] species=${a.species}, breed=${a.breed}, '
            'score=${(a.score * 100).toStringAsFixed(1)}%, '
            'conf=${a.speciesConfidence != null ? (a.speciesConfidence! * 100).toStringAsFixed(1) : "n/a"}%, '
            'pose=${a.pose != null ? "${a.pose!.landmarks.length} kps" : "none"}, '
            'bbox=(${a.boundingBox.left.toInt()},${a.boundingBox.top.toInt()})-(${a.boundingBox.right.toInt()},${a.boundingBox.bottom.toInt()})');
      }
      print('');

      expect(results, isNotEmpty,
          reason: 'Expected at least one detection for $species');
    });
  }
}
