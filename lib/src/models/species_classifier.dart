import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';
import '../util/image_utils.dart';

/// MobileNetV3-Small ImageNet classifier for species/breed identification.
///
/// Classifies a cropped animal image into one of the species categories defined
/// in species_mapping.json (dog, cat, fox, bear, etc.) using a 1000-class
/// ImageNet model. Returns the species category name and classifier confidence.
class SpeciesClassifier {
  /// Input spatial dimension for the classifier model (224x224).
  static const int inputSize = 224;

  static const String _modelPath =
      'packages/animal_detection/assets/models/species_classifier_float16.tflite';
  static const String _mappingPath =
      'packages/animal_detection/assets/models/species_mapping.json';

  Interpreter? _interpreter;
  IsolateInterpreter? _isolateInterpreter;
  Delegate? _delegate;

  final Map<int, String> _speciesLookup = {};
  List<String> _classNames = [];

  late List<List<List<List<double>>>> _inputTensor;
  late List<List<double>> _outputBuffer;
  Float32List? _rgbBuffer;

  /// Initializes the classifier by loading the TFLite model and species mapping
  /// from Flutter assets.
  Future<void> initialize(PerformanceConfig performanceConfig) async {
    final (options, delegate) = InterpreterFactory.create(performanceConfig);
    _delegate = delegate;
    _interpreter = await Interpreter.fromAsset(_modelPath, options: options);
    _isolateInterpreter = await InterpreterFactory.createIsolateIfNeeded(
      _interpreter!,
      _delegate,
    );
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
    final (options, delegate) = InterpreterFactory.create(performanceConfig);
    _delegate = delegate;
    _interpreter = Interpreter.fromBuffer(modelBytes, options: options);
    _isolateInterpreter = await InterpreterFactory.createIsolateIfNeeded(
      _interpreter!,
      _delegate,
    );
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

    if (_isolateInterpreter != null) {
      await _isolateInterpreter!.run(_inputTensor, _outputBuffer);
    } else {
      _interpreter!.run(_inputTensor, _outputBuffer);
    }

    final logits = _outputBuffer[0];
    double maxLogit = logits[0];
    for (int i = 1; i < 1000; i++) {
      if (logits[i] > maxLogit) maxLogit = logits[i];
    }

    double expSum = 0.0;
    final expValues = List<double>.filled(1000, 0.0);
    for (int i = 0; i < 1000; i++) {
      expValues[i] = exp(logits[i] - maxLogit);
      expSum += expValues[i];
    }

    int top1Index = 0;
    double top1Prob = expValues[0] / expSum;
    for (int i = 1; i < 1000; i++) {
      final prob = expValues[i] / expSum;
      if (prob > top1Prob) {
        top1Prob = prob;
        top1Index = i;
      }
    }

    final species = _speciesLookup[top1Index] ?? 'unknown_animal';
    final breed =
        _classNames.isNotEmpty && top1Index < _classNames.length
            ? _classNames[top1Index]
            : null;

    return (species, breed, top1Prob);
  }

  /// Disposes the interpreter and releases native resources.
  void dispose() {
    _isolateInterpreter?.close();
    _interpreter?.close();
    _delegate?.delete();
  }
}
