import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';
import '../util/image_utils.dart';
import '../util/math_utils.dart';
import 'single_interpreter_model.dart';

/// MobileNetV3-Small ImageNet classifier for species/breed identification.
///
/// Classifies a cropped animal image into one of the species categories defined
/// in species_mapping.json (dog, cat, fox, bear, etc.) using a 1000-class
/// ImageNet model. Returns the species category name and classifier confidence.
class SpeciesClassifier extends SingleInterpreterModel {
  /// Input spatial dimension for the classifier model (224x224).
  static const int inputSize = 224;

  static const String _modelPath =
      'packages/animal_detection/assets/models/species_classifier_float16.tflite';
  static const String _mappingPath =
      'packages/animal_detection/assets/models/species_mapping.json';

  final Map<int, String> _speciesLookup = {};
  List<String> _classNames = [];

  late List<List<List<List<double>>>> _inputTensor;
  late List<List<double>> _outputBuffer;
  Float32List? _rgbBuffer;

  /// Initializes the classifier by loading the TFLite model and species mapping
  /// from Flutter assets.
  Future<void> initialize(PerformanceConfig performanceConfig) async {
    await initInterpreterFromAsset(_modelPath, performanceConfig);
    _inputTensor = createNHWCTensor4D(inputSize, inputSize);
    _outputBuffer = List.generate(1, (_) => List.filled(1000, 0.0));

    final mappingJson = await rootBundle.loadString(_mappingPath);
    _buildLookups(mappingJson);
  }

  /// Initializes the classifier from pre-loaded model bytes and mapping JSON string.
  Future<void> initializeFromBuffer(
    Uint8List modelBytes,
    String mappingJson,
    PerformanceConfig performanceConfig,
  ) async {
    await initInterpreterFromBuffer(modelBytes, performanceConfig);
    _inputTensor = createNHWCTensor4D(inputSize, inputSize);
    _outputBuffer = List.generate(1, (_) => List.filled(1000, 0.0));

    _buildLookups(mappingJson);
  }

  void _buildLookups(String mappingJson) {
    final data = jsonDecode(mappingJson) as Map<String, dynamic>;
    final species = data['species'] as Map<String, dynamic>;
    for (final entry in species.entries) {
      final speciesName = entry.key;
      final info = entry.value as Map<String, dynamic>;
      final ids = info['imagenet_ids'] as List<dynamic>;
      for (final id in ids) {
        _speciesLookup[id as int] = speciesName;
      }
    }

    final names = data['imagenet_names'] as List<dynamic>?;
    if (names != null) {
      _classNames = names.cast<String>();
    }
  }

  /// Classify species from a cropped animal image.
  ///
  /// [crop] is an OpenCV Mat in BGR format at any size; it will be resized to
  /// 224x224 internally. Returns (species, breed, confidence).
  Future<(String, String?, double)> classify(cv.Mat crop) async {
    final resized = cv.resize(crop, (inputSize, inputSize));

    _rgbBuffer = ImageUtils.matToFloat32ImageNet(resized);
    fillNHWC4D(_rgbBuffer!, _inputTensor, inputSize, inputSize);
    resized.dispose();

    await runInferenceSingle(_inputTensor, _outputBuffer);

    final logits = _outputBuffer[0];
    final (top1Index, top1Prob) = argmaxSoftmax(logits);

    final species = _speciesLookup[top1Index] ?? 'unknown_animal';
    final breed = _classNames.isNotEmpty && top1Index < _classNames.length
        ? _classNames[top1Index]
        : null;

    return (species, breed, top1Prob);
  }
}
