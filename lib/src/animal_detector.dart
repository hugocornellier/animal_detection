import 'dart:io';
import 'dart:isolate';

import 'package:flutter/services.dart';
import 'package:flutter_litert/native.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

import 'animal_detector_core.dart';
import 'types.dart';
import 'util/image_utils.dart';
import 'util/model_downloader.dart';

class _AnimalIsolateStartupData {
  const _AnimalIsolateStartupData({
    required this.sendPort,
    required this.bodyDetectorBytes,
    required this.classifierBytes,
    required this.speciesMappingJson,
    required this.poseModelBytes,
    required this.poseModelName,
    required this.enablePose,
    required this.cropMargin,
    required this.detThreshold,
    required this.performanceModeName,
    required this.numThreads,
    required this.posePerformanceModeName,
    required this.poseNumThreads,
    required this.useCompiledModel,
    required this.compiledForceCpu,
    required this.acceleratorIndices,
    required this.precisionIndex,
  });

  final SendPort sendPort;
  final TransferableTypedData bodyDetectorBytes;
  final TransferableTypedData classifierBytes;
  final String speciesMappingJson;
  final TransferableTypedData? poseModelBytes;
  final String poseModelName;
  final bool enablePose;
  final double cropMargin;
  final double detThreshold;
  final String performanceModeName;
  final int? numThreads;
  final String? posePerformanceModeName;
  final int? poseNumThreads;
  final bool useCompiledModel;
  final bool compiledForceCpu;
  final List<int> acceleratorIndices;
  final int precisionIndex;
}

/// On-device animal detection using a multi-stage LiteRT pipeline.
///
/// The detector owns one background isolate. Model loading and all inference
/// run there, matching the execution model used by the object, face, pose, and
/// hand detection packages.
class AnimalDetector {
  /// Creates an uninitialized detector with the requested pipeline settings.
  AnimalDetector({
    this.poseModel = AnimalPoseModel.rtmpose,
    this.enablePose = true,
    this.cropMargin = 0.20,
    this.detThreshold = 0.5,
    this.performanceConfig = const PerformanceConfig(),
    this.posePerformanceConfig,
  });

  /// Creates and initializes an animal detector in one step.
  static Future<AnimalDetector> create({
    AnimalPoseModel poseModel = AnimalPoseModel.rtmpose,
    bool enablePose = true,
    double cropMargin = 0.20,
    double detThreshold = 0.5,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
    PerformanceConfig? posePerformanceConfig,
    void Function(String model, int received, int total)? onDownloadProgress,
    bool useCompiledModel = false,
    Set<Accelerator> accelerators = const {
      Accelerator.gpu,
      Accelerator.cpu,
    },
    Precision precision = Precision.fp32,
  }) async {
    final detector = AnimalDetector(
      poseModel: poseModel,
      enablePose: enablePose,
      cropMargin: cropMargin,
      detThreshold: detThreshold,
      performanceConfig: performanceConfig,
      posePerformanceConfig: posePerformanceConfig,
    );
    await detector.initialize(
      onDownloadProgress: onDownloadProgress,
      useCompiledModel: useCompiledModel,
      accelerators: accelerators,
      precision: precision,
    );
    return detector;
  }

  /// Body-pose model variant used when [enablePose] is true.
  final AnimalPoseModel poseModel;

  /// Whether body-pose estimation runs after detection and classification.
  final bool enablePose;

  /// Fractional margin added around each detected body before pose inference.
  final double cropMargin;

  /// Minimum SSD score accepted as an animal detection.
  final double detThreshold;

  /// Delegate configuration for the classic Interpreter backend.
  final PerformanceConfig performanceConfig;

  /// Optional Interpreter configuration for the pose stage alone.
  final PerformanceConfig? posePerformanceConfig;

  _AnimalDetectorWorker? _worker;

  /// Whether the background worker has initialized all requested models.
  bool get isReady => _worker?.isReady ?? false;

  /// Alias retained for consistency with the previous API.
  bool get isInitialized => isReady;

  /// Returns true if the optional HRNet model is already cached locally.
  static Future<bool> isHrnetCached() => ModelDownloader.isHrnetCached();

  /// Loads the bundled models and starts the background detection isolate.
  ///
  /// [useCompiledModel] opts every stage into LiteRT Next CompiledModel. It is
  /// off by default. [accelerators] and [precision] configure that backend;
  /// [performanceConfig] remains specific to the classic Interpreter backend.
  Future<void> initialize({
    void Function(String model, int received, int total)? onDownloadProgress,
    bool useCompiledModel = false,
    Set<Accelerator> accelerators = const {
      Accelerator.gpu,
      Accelerator.cpu,
    },
    Precision precision = Precision.fp32,
  }) async {
    final modelData = await Future.wait<ByteData>([
      rootBundle.load(
        'packages/animal_detection/assets/models/'
        'superanimal_ssdlite_float16.tflite',
      ),
      rootBundle.load(
        'packages/animal_detection/assets/models/'
        'species_classifier_float16.tflite',
      ),
    ]);
    final mapping = await rootBundle.loadString(
      'packages/animal_detection/assets/models/species_mapping.json',
    );

    Uint8List? poseBytes;
    if (enablePose) {
      if (poseModel == AnimalPoseModel.hrnet) {
        poseBytes = await ModelDownloader.getHrnetModel(
          onProgress: onDownloadProgress == null
              ? null
              : (received, total) => onDownloadProgress(
                    ModelDownloader.modelHrnet,
                    received,
                    total,
                  ),
        );
      } else {
        final data = await rootBundle.load(
          'packages/animal_detection/assets/models/'
          'superanimal_rtmpose_s_float16.tflite',
        );
        poseBytes = data.buffer.asUint8List();
      }
    }

    await initializeFromBuffers(
      bodyDetectorBytes: modelData[0].buffer.asUint8List(),
      classifierBytes: modelData[1].buffer.asUint8List(),
      speciesMappingJson: mapping,
      poseModelBytes: poseBytes,
      useCompiledModel: useCompiledModel,
      accelerators: accelerators,
      precision: precision,
    );
  }

  /// Starts the detector from model bytes already loaded by the caller.
  ///
  /// [useIsolateInterpreter] is retained for source compatibility. The public
  /// detector now owns an outer worker isolate, so nested IsolateInterpreters
  /// are always disabled inside that worker.
  Future<void> initializeFromBuffers({
    required Uint8List bodyDetectorBytes,
    required Uint8List classifierBytes,
    required String speciesMappingJson,
    Uint8List? poseModelBytes,
    bool useIsolateInterpreter = true,
    bool useCompiledModel = false,
    bool compiledForceCpu = false,
    Set<Accelerator> accelerators = const {
      Accelerator.gpu,
      Accelerator.cpu,
    },
    Precision precision = Precision.fp32,
  }) async {
    if (_worker != null) await dispose();

    final worker = _AnimalDetectorWorker();
    try {
      await worker.initialize(
        startupData: (sendPort) => _AnimalIsolateStartupData(
          sendPort: sendPort,
          bodyDetectorBytes: TransferableTypedData.fromList([
            bodyDetectorBytes,
          ]),
          classifierBytes: TransferableTypedData.fromList([
            classifierBytes,
          ]),
          speciesMappingJson: speciesMappingJson,
          poseModelBytes: poseModelBytes == null
              ? null
              : TransferableTypedData.fromList([poseModelBytes]),
          poseModelName: poseModel.name,
          enablePose: enablePose,
          cropMargin: cropMargin,
          detThreshold: detThreshold,
          performanceModeName: performanceConfig.mode.name,
          numThreads: performanceConfig.numThreads,
          posePerformanceModeName: posePerformanceConfig?.mode.name,
          poseNumThreads: posePerformanceConfig?.numThreads,
          useCompiledModel: useCompiledModel,
          compiledForceCpu: compiledForceCpu,
          acceleratorIndices: accelerators.map((a) => a.index).toList(),
          precisionIndex: precision.index,
        ),
      );
    } catch (_) {
      await worker.dispose();
      rethrow;
    }
    _worker = worker;
  }

  /// Detects animals from encoded JPEG, PNG, or other OpenCV image bytes.
  Future<List<Animal>> detect(Uint8List imageBytes) async {
    final result = await _requireWorker().sendRequest<List<dynamic>>(
      'detect',
      {
        'bytes': TransferableTypedData.fromList([imageBytes])
      },
    );
    return _deserializeAnimals(result);
  }

  /// Detects animals from a pre-decoded OpenCV matrix.
  ///
  /// The supplied matrix remains owned by the caller.
  Future<List<Animal>> detectFromMat(
    cv.Mat image, {
    required int imageWidth,
    required int imageHeight,
  }) async {
    final result = await _requireWorker().sendRequest<List<dynamic>>(
      'detectMat',
      {
        'bytes': TransferableTypedData.fromList([image.data]),
        'width': imageWidth,
        'height': imageHeight,
        'matType': image.type.value,
      },
    );
    return _deserializeAnimals(result);
  }

  /// Detects animals directly from a camera frame prepared by flutter_litert.
  ///
  /// Colour conversion, optional rotation/downscaling, and inference all run
  /// in this detector's worker isolate.
  Future<List<Animal>> detectFromCameraFrame(
    CameraFrame frame, {
    int? maxDim,
  }) async {
    final result = await _requireWorker().sendRequest<List<dynamic>>(
      'detectCameraFrame',
      cameraFrameRpcFields(frame, {'maxDim': maxDim}),
    );
    return _deserializeAnimals(result);
  }

  /// Convenience wrapper accepting a package:camera `CameraImage`-shaped
  /// object without taking a hard dependency on package:camera.
  Future<List<Animal>> detectFromCameraImage(
    Object cameraImage, {
    CameraFrameRotation? rotation,
    bool? isBgra,
    int? maxDim,
  }) async {
    _requireWorker();
    final frame = prepareCameraFrameFromImage(
      cameraImage,
      rotation: rotation,
      isBgra: isBgra ?? Platform.isMacOS,
    );
    if (frame == null) return const <Animal>[];
    return detectFromCameraFrame(frame, maxDim: maxDim);
  }

  /// Releases the worker isolate and all native model resources.
  Future<void> dispose() async {
    final worker = _worker;
    _worker = null;
    if (worker != null) await worker.disposeGracefully();
  }

  _AnimalDetectorWorker _requireWorker() {
    final worker = _worker;
    if (worker == null || !worker.isReady) {
      throw StateError(
        'AnimalDetector not initialized. Call initialize() first.',
      );
    }
    return worker;
  }

  static List<Animal> _deserializeAnimals(List<dynamic> result) => result
      .map((item) => Animal.fromMap(Map<String, dynamic>.from(item as Map)))
      .toList();

  static cv.Mat _matFromBytes(
    int rows,
    int cols,
    cv.MatType type,
    Uint8List bytes,
  ) {
    final mat = cv.Mat.create(rows: rows, cols: cols, type: type);
    mat.data.setRange(0, bytes.length, bytes);
    return mat;
  }

  @pragma('vm:entry-point')
  static void _isolateEntry(_AnimalIsolateStartupData data) async {
    final mainSendPort = data.sendPort;
    final workerReceivePort = ReceivePort();
    AnimalDetectorCore? detector;

    try {
      final performanceMode = PerformanceMode.values.byName(
        data.performanceModeName,
      );
      final poseMode = data.posePerformanceModeName == null
          ? null
          : PerformanceMode.values.byName(data.posePerformanceModeName!);
      detector = AnimalDetectorCore(
        poseModel: AnimalPoseModel.values.byName(data.poseModelName),
        enablePose: data.enablePose,
        cropMargin: data.cropMargin,
        detThreshold: data.detThreshold,
        performanceConfig: PerformanceConfig(
          mode: performanceMode,
          numThreads: data.numThreads,
        ),
        posePerformanceConfig: poseMode == null
            ? null
            : PerformanceConfig(
                mode: poseMode,
                numThreads: data.poseNumThreads,
              ),
      );
      await detector.initializeFromBuffers(
        bodyDetectorBytes: data.bodyDetectorBytes.materialize().asUint8List(),
        classifierBytes: data.classifierBytes.materialize().asUint8List(),
        speciesMappingJson: data.speciesMappingJson,
        poseModelBytes: data.poseModelBytes?.materialize().asUint8List(),
        useIsolateInterpreter: false,
        useCompiledModel: data.useCompiledModel,
        compiledForceCpu: data.compiledForceCpu,
        accelerators: data.acceleratorIndices
            .map((index) => Accelerator.values[index])
            .toSet(),
        precision: Precision.values[data.precisionIndex],
      );
      mainSendPort.send(workerReceivePort.sendPort);
    } catch (error, stackTrace) {
      mainSendPort.send({
        'error': 'Animal detection isolate initialization failed: '
            '$error\n$stackTrace',
      });
      return;
    }

    workerReceivePort.listen((message) async {
      if (message is! Map) return;
      final id = message['id'] as int?;
      final op = message['op'] as String?;
      if (id == null || op == null) return;

      try {
        switch (op) {
          case 'detect':
            final bytes = (message['bytes'] as TransferableTypedData)
                .materialize()
                .asUint8List();
            final animals = await detector!.detect(bytes);
            mainSendPort.send({
              'id': id,
              'result': animals.map((animal) => animal.toMap()).toList(),
            });
          case 'detectMat':
            final bytes = (message['bytes'] as TransferableTypedData)
                .materialize()
                .asUint8List();
            final width = message['width'] as int;
            final height = message['height'] as int;
            final mat = _matFromBytes(
              height,
              width,
              cv.MatType(message['matType'] as int),
              bytes,
            );
            try {
              final animals = await detector!.detectFromMat(
                mat,
                imageWidth: width,
                imageHeight: height,
              );
              mainSendPort.send({
                'id': id,
                'result': animals.map((animal) => animal.toMap()).toList(),
              });
            } finally {
              mat.dispose();
            }
          case 'detectCameraFrame':
            final bytes = (message['bytes'] as TransferableTypedData)
                .materialize()
                .asUint8List();
            final frame = cameraFrameFromRpcMessage(message, bytes);
            final mat = ImageUtils.cameraFrameToBgrMat(
              frame,
              maxDim: message['maxDim'] as int?,
            );
            try {
              final animals = await detector!.detectFromMat(
                mat,
                imageWidth: mat.cols,
                imageHeight: mat.rows,
              );
              mainSendPort.send({
                'id': id,
                'result': animals.map((animal) => animal.toMap()).toList(),
              });
            } finally {
              mat.dispose();
            }
          case 'dispose':
            await detector?.dispose();
            detector = null;
            mainSendPort.send({'id': id, 'result': true});
            workerReceivePort.close();
        }
      } catch (error, stackTrace) {
        mainSendPort.send({'id': id, 'error': '$error\n$stackTrace'});
      }
    });
  }
}

class _AnimalDetectorWorker extends IsolateWorkerBase {
  @override
  String get workerDisposeOp => 'dispose';

  Future<void> initialize({
    required _AnimalIsolateStartupData Function(SendPort) startupData,
  }) async {
    await initWorker(
      (sendPort) => Isolate.spawn(
        AnimalDetector._isolateEntry,
        startupData(sendPort),
        debugName: 'AnimalDetector',
      ),
      timeout: const Duration(minutes: 2),
      timeoutMessage: 'Animal detection isolate initialization timed out',
    );
  }
}
