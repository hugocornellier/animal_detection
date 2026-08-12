import 'package:animal_detection/animal_detection.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('packed camera frame matches direct Mat detection',
      (tester) async {
    final detector = await AnimalDetector.create(
      performanceConfig: const PerformanceConfig.xnnpack(),
    );
    final data = await rootBundle.load('integration_test/test_images/cat.jpg');
    final mat = cv.imdecode(data.buffer.asUint8List(), cv.IMREAD_COLOR);
    final bgra = cv.cvtColor(mat, cv.COLOR_BGR2BGRA);
    try {
      final direct = await detector.detectFromMat(
        mat,
        imageWidth: mat.cols,
        imageHeight: mat.rows,
      );
      final camera = await detector.detectFromCameraFrame(
        CameraFrame(
          bytes: Uint8List.fromList(bgra.data),
          width: bgra.cols,
          height: bgra.rows,
          strideCols: bgra.cols,
          conversion: CameraFrameConversion.bgra2bgr,
        ),
      );

      expect(
        camera.map((value) => value.toMap()).toList(),
        direct.map((value) => value.toMap()).toList(),
      );
    } finally {
      bgra.dispose();
      mat.dispose();
      await detector.dispose();
    }
  }, timeout: const Timeout(Duration(minutes: 3)));
}
