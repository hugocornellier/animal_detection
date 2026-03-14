import 'dart:async';
import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';
import '../util/image_utils.dart';
import 'single_interpreter_model.dart';

/// Generic face bounding box regression model using letterbox preprocessing.
///
/// Runs an EfficientNet-based regression model that takes a letterbox-resized
/// image and outputs a single [x1,y1,x2,y2] bounding box normalized to [0,1].
///
/// Used by both cat and dog face detection pipelines. The only difference
/// between species is the model weights (passed via asset path or buffer).
class FaceLocalizerModel extends SingleInterpreterModel {
  /// Input spatial dimension (width and height).
  final int inputSize;

  final String _modelPath;

  late List<List<List<List<double>>>> _inputTensor;
  late List<List<double>> _outputBuffer;
  Float32List? _rgbBuffer;

  /// Creates a face localizer with the given [inputSize] and [modelPath].
  FaceLocalizerModel({
    required this.inputSize,
    required String modelPath,
  }) : _modelPath = modelPath;

  /// Initializes the model from Flutter assets.
  Future<void> initialize(PerformanceConfig performanceConfig) async {
    await initInterpreterFromAsset(_modelPath, performanceConfig);
    _inputTensor = createNHWCTensor4D(inputSize, inputSize);
    _outputBuffer = List.generate(1, (_) => List.filled(4, 0.0));
  }

  /// Initializes the model from pre-loaded bytes (for isolate use).
  Future<void> initializeFromBuffer(
    Uint8List bytes,
    PerformanceConfig performanceConfig,
  ) async {
    await initInterpreterFromBuffer(bytes, performanceConfig);
    _inputTensor = createNHWCTensor4D(inputSize, inputSize);
    _outputBuffer = List.generate(1, (_) => List.filled(4, 0.0));
  }

  /// Detects a face bounding box in the given image crop.
  ///
  /// Returns null if the detected box is degenerate (width or height < 1px).
  Future<BoundingBox?> detect(cv.Mat image) async {
    final (padded, params) = ImageUtils.letterboxResize(image, inputSize);

    _rgbBuffer = ImageUtils.matToFloat32(padded);
    fillNHWC4D(_rgbBuffer!, _inputTensor, inputSize, inputSize);
    padded.dispose();

    await runInferenceSingle(_inputTensor, _outputBuffer);

    final raw = _outputBuffer[0];
    final xa = raw[0].clamp(0.0, 1.0) * inputSize;
    final ya = raw[1].clamp(0.0, 1.0) * inputSize;
    final xb = raw[2].clamp(0.0, 1.0) * inputSize;
    final yb = raw[3].clamp(0.0, 1.0) * inputSize;
    final x1 = xa < xb ? xa : xb;
    final x2 = xa < xb ? xb : xa;
    final y1 = ya < yb ? ya : yb;
    final y2 = ya < yb ? yb : ya;

    final scaled = scaleFromLetterbox(
      [x1, y1, x2, y2],
      params.scale,
      params.padLeft,
      params.padTop,
    );
    final iw = image.cols.toDouble();
    final ih = image.rows.toDouble();
    final origX1 = scaled[0].clamp(0.0, iw);
    final origY1 = scaled[1].clamp(0.0, ih);
    final origX2 = scaled[2].clamp(0.0, iw);
    final origY2 = scaled[3].clamp(0.0, ih);

    if (origX2 - origX1 < 1.0 || origY2 - origY1 < 1.0) return null;

    return BoundingBox.ltrb(origX1, origY1, origX2, origY2);
  }
}
