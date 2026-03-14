import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:animal_detection/src/util/nms.dart';

// Helpers to build flattened Float64List boxes and Float64List scores.
Float64List makeBoxes(List<List<double>> boxes) {
  final flat = <double>[];
  for (final b in boxes) {
    flat.addAll(b); // [x1, y1, x2, y2]
  }
  return Float64List.fromList(flat);
}

Float64List makeScores(List<double> scores) => Float64List.fromList(scores);

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  // ---------------------------------------------------------------------------
  // Empty / trivial inputs
  // ---------------------------------------------------------------------------
  group('empty and trivial inputs', () {
    test('empty boxes and scores returns empty list', () {
      final result = nonMaxSuppression(
        boxes: makeBoxes([]),
        scores: makeScores([]),
        iouThreshold: 0.5,
        scoreThreshold: 0.3,
      );
      expect(result, isEmpty);
    });

    test('all scores below threshold returns empty list', () {
      final result = nonMaxSuppression(
        boxes: makeBoxes([
          [0.0, 0.0, 10.0, 10.0],
          [20.0, 20.0, 30.0, 30.0],
        ]),
        scores: makeScores([0.1, 0.2]),
        iouThreshold: 0.5,
        scoreThreshold: 0.3,
      );
      expect(result, isEmpty);
    });

    test('scores exactly at threshold are NOT kept (strictly greater than)',
        () {
      final result = nonMaxSuppression(
        boxes: makeBoxes([
          [0.0, 0.0, 10.0, 10.0]
        ]),
        scores: makeScores([0.5]),
        iouThreshold: 0.5,
        scoreThreshold: 0.5,
      );
      // score 0.5 is NOT > 0.5, so it should be filtered out
      expect(result, isEmpty);
    });
  });

  // ---------------------------------------------------------------------------
  // Single box
  // ---------------------------------------------------------------------------
  group('single box', () {
    test('single box above threshold is kept', () {
      final result = nonMaxSuppression(
        boxes: makeBoxes([
          [10.0, 10.0, 50.0, 50.0]
        ]),
        scores: makeScores([0.9]),
        iouThreshold: 0.5,
        scoreThreshold: 0.3,
      );
      expect(result, [0]);
    });

    test('single box below threshold is not kept', () {
      final result = nonMaxSuppression(
        boxes: makeBoxes([
          [10.0, 10.0, 50.0, 50.0]
        ]),
        scores: makeScores([0.1]),
        iouThreshold: 0.5,
        scoreThreshold: 0.3,
      );
      expect(result, isEmpty);
    });
  });

  // ---------------------------------------------------------------------------
  // Non-overlapping boxes
  // ---------------------------------------------------------------------------
  group('non-overlapping boxes', () {
    test('two non-overlapping boxes above threshold are both kept', () {
      final result = nonMaxSuppression(
        boxes: makeBoxes([
          [0.0, 0.0, 10.0, 10.0],
          [20.0, 20.0, 30.0, 30.0],
        ]),
        scores: makeScores([0.9, 0.8]),
        iouThreshold: 0.5,
        scoreThreshold: 0.3,
      );
      expect(result.length, 2);
      expect(result.contains(0), true);
      expect(result.contains(1), true);
    });

    test('three non-overlapping boxes all kept', () {
      final result = nonMaxSuppression(
        boxes: makeBoxes([
          [0.0, 0.0, 10.0, 10.0],
          [50.0, 0.0, 60.0, 10.0],
          [100.0, 0.0, 110.0, 10.0],
        ]),
        scores: makeScores([0.9, 0.85, 0.7]),
        iouThreshold: 0.5,
        scoreThreshold: 0.3,
      );
      expect(result.length, 3);
    });

    test('boxes touching at edge have IoU 0 and are both kept', () {
      // Box A: [0, 0, 10, 10], Box B: [10, 0, 20, 10] — share an edge, IoU = 0
      final result = nonMaxSuppression(
        boxes: makeBoxes([
          [0.0, 0.0, 10.0, 10.0],
          [10.0, 0.0, 20.0, 10.0],
        ]),
        scores: makeScores([0.9, 0.8]),
        iouThreshold: 0.5,
        scoreThreshold: 0.3,
      );
      expect(result.length, 2);
    });
  });

  // ---------------------------------------------------------------------------
  // Fully overlapping boxes (same box, different scores)
  // ---------------------------------------------------------------------------
  group('fully overlapping boxes', () {
    test('identical boxes: only the highest-score box is kept', () {
      final result = nonMaxSuppression(
        boxes: makeBoxes([
          [0.0, 0.0, 10.0, 10.0],
          [0.0, 0.0, 10.0, 10.0],
        ]),
        scores: makeScores([0.9, 0.7]),
        iouThreshold: 0.5,
        scoreThreshold: 0.3,
      );
      expect(result.length, 1);
      expect(result[0], 0); // index 0 has the higher score
    });

    test(
        'identical boxes: lower-score box is suppressed regardless of threshold',
        () {
      final result = nonMaxSuppression(
        boxes: makeBoxes([
          [5.0, 5.0, 15.0, 15.0],
          [5.0, 5.0, 15.0, 15.0],
          [5.0, 5.0, 15.0, 15.0],
        ]),
        scores: makeScores([0.6, 0.9, 0.75]),
        iouThreshold: 0.5,
        scoreThreshold: 0.3,
      );
      // index 1 has highest score (0.9), all others are suppressed
      expect(result.length, 1);
      expect(result[0], 1);
    });
  });

  // ---------------------------------------------------------------------------
  // Partial overlap — IoU threshold behavior
  // ---------------------------------------------------------------------------
  group('partial overlap IoU threshold', () {
    // Box A: [0,0,10,10] area=100
    // Box B: [5,0,15,10] area=100, intersection=[5,0,10,10]=50
    // IoU = 50 / (100 + 100 - 50) = 50/150 ≈ 0.333
    test('partially overlapping boxes with IoU below threshold are both kept',
        () {
      final result = nonMaxSuppression(
        boxes: makeBoxes([
          [0.0, 0.0, 10.0, 10.0],
          [5.0, 0.0, 15.0, 10.0],
        ]),
        scores: makeScores([0.9, 0.8]),
        iouThreshold: 0.5, // IoU ≈ 0.333 < 0.5
        scoreThreshold: 0.3,
      );
      expect(result.length, 2);
      expect(result.contains(0), true);
      expect(result.contains(1), true);
    });

    test(
        'partially overlapping boxes with IoU above threshold: lower score suppressed',
        () {
      final result = nonMaxSuppression(
        boxes: makeBoxes([
          [0.0, 0.0, 10.0, 10.0],
          [5.0, 0.0, 15.0, 10.0],
        ]),
        scores: makeScores([0.9, 0.8]),
        iouThreshold: 0.3, // IoU ≈ 0.333 > 0.3
        scoreThreshold: 0.3,
      );
      expect(result.length, 1);
      expect(result[0], 0); // index 0 has higher score
    });

    // Box A: [0,0,10,10] area=100
    // Box B: [2,2,8,8] area=36, intersection=[2,2,8,8]=36
    // IoU = 36 / (100 + 36 - 36) = 36/100 = 0.36
    test('contained box with IoU above threshold is suppressed', () {
      final result = nonMaxSuppression(
        boxes: makeBoxes([
          [0.0, 0.0, 10.0, 10.0],
          [2.0, 2.0, 8.0, 8.0],
        ]),
        scores: makeScores([0.9, 0.6]),
        iouThreshold: 0.3, // IoU = 0.36 > 0.3
        scoreThreshold: 0.3,
      );
      expect(result.length, 1);
      expect(result[0], 0);
    });

    test('contained box with IoU below threshold is kept', () {
      final result = nonMaxSuppression(
        boxes: makeBoxes([
          [0.0, 0.0, 10.0, 10.0],
          [2.0, 2.0, 8.0, 8.0],
        ]),
        scores: makeScores([0.9, 0.6]),
        iouThreshold: 0.5, // IoU = 0.36 < 0.5
        scoreThreshold: 0.3,
      );
      expect(result.length, 2);
    });
  });

  // ---------------------------------------------------------------------------
  // Score ordering
  // ---------------------------------------------------------------------------
  group('score ordering', () {
    test('result is ordered highest score first', () {
      final result = nonMaxSuppression(
        boxes: makeBoxes([
          [0.0, 0.0, 5.0, 5.0], // index 0, score 0.4
          [50.0, 50.0, 55.0, 55.0], // index 1, score 0.95
          [100.0, 0.0, 105.0, 5.0], // index 2, score 0.7
        ]),
        scores: makeScores([0.4, 0.95, 0.7]),
        iouThreshold: 0.5,
        scoreThreshold: 0.3,
      );
      expect(result.length, 3);
      expect(result[0], 1); // 0.95 — highest
      expect(result[1], 2); // 0.70
      expect(result[2], 0); // 0.40
    });

    test(
        'highest-score box is always the one kept when duplicates are suppressed',
        () {
      // Box at same location with 3 different scores
      final result = nonMaxSuppression(
        boxes: makeBoxes([
          [0.0, 0.0, 10.0, 10.0], // score 0.6
          [0.0, 0.0, 10.0, 10.0], // score 0.95
          [0.0, 0.0, 10.0, 10.0], // score 0.4
        ]),
        scores: makeScores([0.6, 0.95, 0.4]),
        iouThreshold: 0.5,
        scoreThreshold: 0.3,
      );
      expect(result.length, 1);
      expect(result[0], 1); // index 1 has score 0.95
    });
  });

  // ---------------------------------------------------------------------------
  // Multiple suppressions
  // ---------------------------------------------------------------------------
  group('multiple suppressions', () {
    test('one high-score box suppresses two nearby boxes', () {
      // Box 0 (score=0.95) overlaps both box 1 and box 2
      final result = nonMaxSuppression(
        boxes: makeBoxes([
          [0.0, 0.0, 10.0, 10.0], // score 0.95
          [1.0, 1.0, 11.0, 11.0], // score 0.8, high overlap with box 0
          [2.0, 2.0, 12.0, 12.0], // score 0.7, high overlap with box 0
          [100.0, 100.0, 110.0, 110.0], // score 0.85, no overlap
        ]),
        scores: makeScores([0.95, 0.8, 0.7, 0.85]),
        iouThreshold: 0.3,
        scoreThreshold: 0.3,
      );
      // Box 0 and box 3 are kept; boxes 1 and 2 are suppressed by box 0
      expect(result.contains(0), true);
      expect(result.contains(3), true);
      expect(result.contains(1), false);
      expect(result.contains(2), false);
      expect(result.length, 2);
    });

    test('mixed scenario: some below threshold, some suppressed, some kept',
        () {
      final result = nonMaxSuppression(
        boxes: makeBoxes([
          [0.0, 0.0, 10.0, 10.0], // score 0.1 — below threshold
          [50.0, 50.0, 60.0, 60.0], // score 0.9 — kept
          [51.0, 51.0, 61.0, 61.0], // score 0.8 — suppressed by box 1
          [200.0, 200.0, 210.0, 210.0], // score 0.7 — kept (no overlap)
        ]),
        scores: makeScores([0.1, 0.9, 0.8, 0.7]),
        iouThreshold: 0.3,
        scoreThreshold: 0.3,
      );
      expect(result.contains(0), false); // below threshold
      expect(result.contains(1), true);
      expect(result.contains(2), false); // suppressed
      expect(result.contains(3), true);
      expect(result.length, 2);
    });

    test(
        'chain suppression: winner from first cluster does not suppress distant cluster',
        () {
      // Cluster A: boxes 0 (0.9) and 1 (0.8) are overlapping
      // Cluster B: boxes 2 (0.85) and 3 (0.6) are overlapping, far from cluster A
      final result = nonMaxSuppression(
        boxes: makeBoxes([
          [0.0, 0.0, 10.0, 10.0], // cluster A, score 0.9
          [1.0, 0.0, 11.0, 10.0], // cluster A, score 0.8 — suppressed
          [500.0, 0.0, 510.0, 10.0], // cluster B, score 0.85
          [501.0, 0.0, 511.0, 10.0], // cluster B, score 0.6 — suppressed
        ]),
        scores: makeScores([0.9, 0.8, 0.85, 0.6]),
        iouThreshold: 0.3,
        scoreThreshold: 0.3,
      );
      expect(result.length, 2);
      expect(result.contains(0), true);
      expect(result.contains(2), true);
      expect(result.contains(1), false);
      expect(result.contains(3), false);
    });

    test('high iou threshold keeps more boxes', () {
      final result = nonMaxSuppression(
        boxes: makeBoxes([
          [0.0, 0.0, 10.0, 10.0],
          [1.0, 1.0, 11.0, 11.0],
          [2.0, 2.0, 12.0, 12.0],
        ]),
        scores: makeScores([0.9, 0.8, 0.7]),
        iouThreshold:
            0.99, // very high threshold — boxes must nearly perfectly overlap
        scoreThreshold: 0.3,
      );
      // IoU between adjacent shifted boxes is well below 0.99, so all kept
      expect(result.length, 3);
    });

    test('low iou threshold keeps fewer boxes', () {
      final result = nonMaxSuppression(
        boxes: makeBoxes([
          [0.0, 0.0, 10.0, 10.0],
          [1.0, 1.0, 11.0, 11.0],
          [2.0, 2.0, 12.0, 12.0],
        ]),
        scores: makeScores([0.9, 0.8, 0.7]),
        iouThreshold: 0.01, // very low — any overlap triggers suppression
        scoreThreshold: 0.3,
      );
      // All three overlap each other slightly, so only the highest-score is kept
      expect(result.length, 1);
      expect(result[0], 0); // score 0.9
    });
  });
}
