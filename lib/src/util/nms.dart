import 'dart:typed_data';

/// Non-maximum suppression on axis-aligned bounding boxes.
///
/// Returns indices of boxes to keep, sorted by score descending.
/// [boxes] is flattened as [N*4] with layout x1,y1,x2,y2 per box.
List<int> nonMaxSuppression({
  required Float64List boxes,
  required Float64List scores,
  required double iouThreshold,
  required double scoreThreshold,
}) {
  final int n = scores.length;

  final indices = <int>[];
  for (int i = 0; i < n; i++) {
    if (scores[i] > scoreThreshold) {
      indices.add(i);
    }
  }

  indices.sort((a, b) => scores[b].compareTo(scores[a]));

  final suppressed = List<bool>.filled(indices.length, false);
  final kept = <int>[];

  for (int i = 0; i < indices.length; i++) {
    if (suppressed[i]) continue;
    final int idx = indices[i];
    kept.add(idx);

    for (int j = i + 1; j < indices.length; j++) {
      if (suppressed[j]) continue;
      if (_iou(boxes, idx, indices[j]) > iouThreshold) {
        suppressed[j] = true;
      }
    }
  }

  return kept;
}

double _iou(Float64List boxes, int i, int j) {
  final double x1i = boxes[i * 4 + 0];
  final double y1i = boxes[i * 4 + 1];
  final double x2i = boxes[i * 4 + 2];
  final double y2i = boxes[i * 4 + 3];

  final double x1j = boxes[j * 4 + 0];
  final double y1j = boxes[j * 4 + 1];
  final double x2j = boxes[j * 4 + 2];
  final double y2j = boxes[j * 4 + 3];

  final double interX1 = x1i > x1j ? x1i : x1j;
  final double interY1 = y1i > y1j ? y1i : y1j;
  final double interX2 = x2i < x2j ? x2i : x2j;
  final double interY2 = y2i < y2j ? y2i : y2j;

  final double interW = interX2 - interX1;
  final double interH = interY2 - interY1;

  if (interW <= 0 || interH <= 0) return 0.0;

  final double interArea = interW * interH;
  final double areaI = (x2i - x1i) * (y2i - y1i);
  final double areaJ = (x2j - x1j) * (y2j - y1j);

  return interArea / (areaI + areaJ - interArea);
}
