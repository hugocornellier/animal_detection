import 'dart:math' as math;

/// Numerically stable softmax confidence at a specific index.
///
/// Returns the softmax probability at [peakIndex] from [logits].
/// Used by body pose estimator for keypoint confidence.
double softmaxConfidence(List<double> logits, int peakIndex) {
  double maxLogit = logits[0];
  for (int i = 1; i < logits.length; i++) {
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }
  double sumExp = 0.0;
  for (int i = 0; i < logits.length; i++) {
    sumExp += math.exp(logits[i] - maxLogit);
  }
  return math.exp(logits[peakIndex] - maxLogit) / sumExp;
}

/// Returns the argmax index and its softmax probability from [logits].
///
/// Used by species classifier for top-1 class prediction.
(int, double) argmaxSoftmax(List<double> logits) {
  // Find max logit for numerical stability
  double maxLogit = logits[0];
  for (int i = 1; i < logits.length; i++) {
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }

  // Compute softmax and find argmax simultaneously
  double expSum = 0.0;
  int bestIdx = 0;
  double bestExp = 0.0;

  for (int i = 0; i < logits.length; i++) {
    final e = math.exp(logits[i] - maxLogit);
    expSum += e;
    if (e > bestExp) {
      bestExp = e;
      bestIdx = i;
    }
  }

  return (bestIdx, bestExp / expSum);
}
