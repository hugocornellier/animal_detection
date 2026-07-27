// Exercises the HRNet pose path, which no other suite covers.
//
// AnimalPoseModel.hrnet downloads a ~57 MB model on demand, so this is kept in
// its own file rather than slowing the default suite. It exists because the
// HRNet branch reads its heatmap with flat NHWC index arithmetic
// ((row * size + col) * keypoints + kp) after the move off nested output
// lists, and a wrong index there shifts keypoints silently rather than
// throwing.
//
//   flutter test integration_test/hrnet_pose_test.dart -d macos
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:animal_detection/animal_detection.dart';

const _image = 'integration_test/test_images/cat.jpg';

Future<List<Animal>> _run(AnimalPoseModel poseModel) async {
  final detector = AnimalDetector(enablePose: true, poseModel: poseModel);
  await detector.initialize();
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

  testWidgets('HRNet keypoints agree with RTMPose on the same image',
      (tester) async {
    final rtm = await _run(AnimalPoseModel.rtmpose);
    final hr = await _run(AnimalPoseModel.hrnet);

    expect(rtm, isNotEmpty, reason: 'rtmpose found no animal');
    expect(hr, isNotEmpty, reason: 'hrnet found no animal');

    final a = rtm.first.pose;
    final b = hr.first.pose;
    expect(a, isNotNull);
    expect(b, isNotNull);
    expect(b!.landmarks.length, a!.landmarks.length,
        reason: 'both models predict the same keypoint set');

    // Both estimate the same anatomy, so corresponding keypoints should land
    // near each other. A transposed or mis-strided heatmap index would scatter
    // them across the crop while still producing finite, in-bounds values.
    final w = rtm.first.imageWidth.toDouble();
    final h = rtm.first.imageHeight.toDouble();
    // Generous: the two models genuinely disagree somewhat. This is sized to
    // catch scattering, not to pin agreement.
    final tol = 0.15 * math.max(w, h);

    // Confidence is NOT comparable between the two models: rtmpose reports a
    // product of softmax maxima, hrnet reports a raw heatmap activation. So
    // compare geometry only, and use the median rather than the worst distance
    // so a few genuinely uncertain keypoints (occluded tail, far leg) cannot
    // fail the run while real scattering still does.
    final dists = <double>[];
    for (int i = 0; i < a.landmarks.length; i++) {
      final p = a.landmarks[i];
      final q = b.landmarks[i];
      expect(q.x.isFinite && q.y.isFinite, isTrue,
          reason: 'hrnet keypoint $i is not finite');
      expect(q.x, inInclusiveRange(-w, 2 * w));
      expect(q.y, inInclusiveRange(-h, 2 * h));
      dists.add(math.sqrt(
          (p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y)));
    }
    dists.sort();
    final median = dists[dists.length ~/ 2];

    double confMin(AnimalPose pose) => pose.landmarks
        .map((l) => l.confidence)
        .reduce((x, y) => x < y ? x : y);
    double confMax(AnimalPose pose) => pose.landmarks
        .map((l) => l.confidence)
        .reduce((x, y) => x > y ? x : y);

    debugPrint('HRNET n=${dists.length} median=${median.toStringAsFixed(1)}px '
        'worst=${dists.last.toStringAsFixed(1)}px '
        'tol=${tol.toStringAsFixed(1)}px image=${w.toInt()}x${h.toInt()}');
    debugPrint('HRNET conf rtmpose=[${confMin(a).toStringAsFixed(3)}, '
        '${confMax(a).toStringAsFixed(3)}] '
        'hrnet=[${confMin(b).toStringAsFixed(3)}, '
        '${confMax(b).toStringAsFixed(3)}]');

    expect(median, lessThan(tol),
        reason: 'hrnet keypoints diverge from rtmpose beyond plausible model '
            'disagreement, which is what a bad heatmap index looks like');
  }, timeout: const Timeout(Duration(minutes: 10)));
}
