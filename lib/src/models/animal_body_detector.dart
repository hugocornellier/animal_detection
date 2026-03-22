import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';
import '../util/image_utils.dart';
import '../util/nms.dart';
import 'ssd_anchors.dart' show ssdAnchors;
import 'single_interpreter_model.dart';

/// SSDLite body detection model runner.
///
/// Detects animals in an image using the SuperAnimal SSDLite320 TFLite model.
/// Input: 320x320, ImageNet-normalized. Output: 12 tensors (6 reg + 6 cls),
/// one pair per SSD feature level. Returns bounding boxes in original image
/// pixel coordinates, sorted by score descending.
class AnimalBodyDetector extends SingleInterpreterModel {
  /// Input spatial dimension for the detector model (320x320).
  static const int inputSize = 320;

  static const String _modelPath =
      'packages/animal_detection/assets/models/superanimal_ssdlite_float16.tflite';

  // BoxCoder weights: (wx, wy, ww, wh)
  static const double _wx = 10.0;
  static const double _wy = 10.0;
  static const double _ww = 5.0;
  static const double _wh = 5.0;

  // Maximum exponent value for box size deltas to prevent overflow.
  static const double _bboxClip = 4.135166556742356;

  // NMS IoU threshold for detections.
  static const double _nmsIouThreshold = 0.55;

  // Total number of anchors across all feature levels.
  static const int _totalAnchors = 3234;

  // SSD feature level spatial sizes: (H, W) for each of the 6 levels.
  static const List<(int, int)> _levelSizes = [
    (20, 20),
    (10, 10),
    (5, 5),
    (3, 3),
    (2, 2),
    (1, 1),
  ];

  static const int _anchorsPerLoc = 6;

  late List<List<List<List<double>>>> _inputTensor;
  late Map<int, Object> _outputBuffers;
  late List<List<int>> _outputTensorShapes;
  Float32List? _rgbBuffer;

  /// Initializes the detector by loading the TFLite model from Flutter assets.
  Future<void> initialize(
    PerformanceConfig performanceConfig, {
    bool useIsolateInterpreter = true,
  }) async {
    await initInterpreterFromAsset(
      _modelPath,
      performanceConfig,
      useIsolateInterpreter: useIsolateInterpreter,
    );
    _inputTensor = createNHWCTensor4D(inputSize, inputSize);
    _outputBuffers = createOutputBuffers(
      interpreter!.getOutputTensors().map((t) => t.shape).toList(),
    );
    _outputTensorShapes =
        interpreter!.getOutputTensors().map((t) => t.shape).toList();
  }

  /// Initializes the detector from pre-loaded model bytes.
  Future<void> initializeFromBuffer(
    Uint8List bytes,
    PerformanceConfig performanceConfig, {
    bool useIsolateInterpreter = true,
  }) async {
    await initInterpreterFromBuffer(
      bytes,
      performanceConfig,
      useIsolateInterpreter: useIsolateInterpreter,
    );
    _inputTensor = createNHWCTensor4D(inputSize, inputSize);
    _outputBuffers = createOutputBuffers(
      interpreter!.getOutputTensors().map((t) => t.shape).toList(),
    );
    _outputTensorShapes =
        interpreter!.getOutputTensors().map((t) => t.shape).toList();
  }

  /// Detect animals in [image].
  ///
  /// Resizes the image to 320x320, applies ImageNet normalization, runs the
  /// SSDLite model, decodes anchor-based boxes, and applies NMS.
  ///
  /// Returns a list of (BoundingBox, score) pairs in original image coordinates,
  /// sorted by score descending. Empty if no detections pass [scoreThreshold].
  Future<List<(BoundingBox, double)>> detect(
    cv.Mat image, {
    double scoreThreshold = 0.5,
  }) async {
    final int origW = image.cols;
    final int origH = image.rows;

    // 1. Resize to 320x320 and apply ImageNet normalization (no letterbox).
    final resized = cv.resize(image, (inputSize, inputSize));
    _rgbBuffer = ImageUtils.matToFloat32ImageNet(resized);
    resized.dispose();

    fillNHWC4D(_rgbBuffer!, _inputTensor, inputSize, inputSize);

    // 2. Run inference.
    await runInference([_inputTensor], _outputBuffers);

    // 3. Group outputs by (H, W), identify reg (C=24) vs cls (C=12).
    final byHW = <(int, int), _LevelOutputs>{};

    for (int i = 0; i < _outputTensorShapes.length; i++) {
      final shape = _outputTensorShapes[i];
      // Shape is [1, H, W, C]
      final int h = shape[1];
      final int w = shape[2];
      final int c = shape[3];
      final key = (h, w);

      byHW[key] ??= _LevelOutputs();
      final buf = _outputBuffers[i]! as List<List<List<List<double>>>>;

      if (c == 24) {
        byHW[key]!.reg = buf;
        byHW[key]!.regH = h;
        byHW[key]!.regW = w;
      } else {
        byHW[key]!.cls = buf;
      }
    }

    // 4. Concatenate reg and cls across all levels into flat arrays.
    final regAll = Float64List(_totalAnchors * 4);
    final clsAll = Float64List(_totalAnchors * 2);

    int anchorOffset = 0;
    for (final (h, w) in _levelSizes) {
      final level = byHW[(h, w)];
      if (level == null || level.reg == null || level.cls == null) {
        throw StateError('Missing detector output for level ($h, $w)');
      }

      final reg = level.reg!;
      final cls = level.cls!;
      final int numAnchors = h * w * _anchorsPerLoc;

      int flatIdx = 0;
      for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
          for (int a = 0; a < _anchorsPerLoc; a++) {
            final int regBase = a * 4;
            final int clsBase = a * 2;
            final int outBase = (anchorOffset + flatIdx) * 4;
            final int outClsBase = (anchorOffset + flatIdx) * 2;

            regAll[outBase + 0] = reg[0][row][col][regBase + 0];
            regAll[outBase + 1] = reg[0][row][col][regBase + 1];
            regAll[outBase + 2] = reg[0][row][col][regBase + 2];
            regAll[outBase + 3] = reg[0][row][col][regBase + 3];

            clsAll[outClsBase + 0] = cls[0][row][col][clsBase + 0];
            clsAll[outClsBase + 1] = cls[0][row][col][clsBase + 1];

            flatIdx++;
          }
        }
      }

      assert(flatIdx == numAnchors);
      anchorOffset += numAnchors;
    }

    // 5. Decode boxes from regression deltas + anchors (in 320px space).
    final boxes320 = _decodeSsdBoxes(regAll);

    // 6. Compute softmax foreground scores (column 1).
    final scores = Float64List(_totalAnchors);
    for (int i = 0; i < _totalAnchors; i++) {
      final double logit0 = clsAll[i * 2 + 0];
      final double logit1 = clsAll[i * 2 + 1];
      final double maxLogit = logit0 > logit1 ? logit0 : logit1;
      final double exp0 = math.exp(logit0 - maxLogit);
      final double exp1 = math.exp(logit1 - maxLogit);
      scores[i] = exp1 / (exp0 + exp1);
    }

    // 7. Scale boxes to original image coordinates.
    final double scaleX = origW / inputSize;
    final double scaleY = origH / inputSize;
    final boxesOrig = Float64List(_totalAnchors * 4);
    for (int i = 0; i < _totalAnchors; i++) {
      final double x1 =
          (boxes320[i * 4 + 0] * scaleX).clamp(0.0, origW.toDouble());
      final double y1 =
          (boxes320[i * 4 + 1] * scaleY).clamp(0.0, origH.toDouble());
      final double x2 =
          (boxes320[i * 4 + 2] * scaleX).clamp(0.0, origW.toDouble());
      final double y2 =
          (boxes320[i * 4 + 3] * scaleY).clamp(0.0, origH.toDouble());
      boxesOrig[i * 4 + 0] = x1;
      boxesOrig[i * 4 + 1] = y1;
      boxesOrig[i * 4 + 2] = x2;
      boxesOrig[i * 4 + 3] = y2;
    }

    // 8. NMS.
    final keptIndices = nonMaxSuppression(
      boxes: boxesOrig,
      scores: scores,
      iouThreshold: _nmsIouThreshold,
      scoreThreshold: scoreThreshold,
    );

    // 9. Build results sorted by score descending (NMS already returns sorted).
    final results = <(BoundingBox, double)>[];
    for (final i in keptIndices) {
      results.add((
        BoundingBox.ltrb(
          boxesOrig[i * 4 + 0],
          boxesOrig[i * 4 + 1],
          boxesOrig[i * 4 + 2],
          boxesOrig[i * 4 + 3],
        ),
        scores[i],
      ));
    }

    return results;
  }

  /// Decodes SSD regression deltas into absolute boxes in 320px space.
  Float64List _decodeSsdBoxes(Float64List regFlat) {
    final decoded = Float64List(_totalAnchors * 4);

    for (int i = 0; i < _totalAnchors; i++) {
      final double ax1 = ssdAnchors[i * 4 + 0];
      final double ay1 = ssdAnchors[i * 4 + 1];
      final double ax2 = ssdAnchors[i * 4 + 2];
      final double ay2 = ssdAnchors[i * 4 + 3];

      final double aw = ax2 - ax1;
      final double ah = ay2 - ay1;
      final double acx = ax1 + 0.5 * aw;
      final double acy = ay1 + 0.5 * ah;

      final double dx = regFlat[i * 4 + 0] / _wx;
      final double dy = regFlat[i * 4 + 1] / _wy;
      final double dw =
          (regFlat[i * 4 + 2] / _ww).clamp(double.negativeInfinity, _bboxClip);
      final double dh =
          (regFlat[i * 4 + 3] / _wh).clamp(double.negativeInfinity, _bboxClip);

      final double predCx = dx * aw + acx;
      final double predCy = dy * ah + acy;
      final double predW = math.exp(dw) * aw;
      final double predH = math.exp(dh) * ah;

      decoded[i * 4 + 0] = predCx - 0.5 * predW;
      decoded[i * 4 + 1] = predCy - 0.5 * predH;
      decoded[i * 4 + 2] = predCx + 0.5 * predW;
      decoded[i * 4 + 3] = predCy + 0.5 * predH;
    }

    return decoded;
  }
}

/// Holds the reg and cls output buffers for one SSD feature level.
class _LevelOutputs {
  List<List<List<List<double>>>>? reg;
  List<List<List<List<double>>>>? cls;
  int regH = 0;
  int regW = 0;
}
