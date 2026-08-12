import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:animal_detection/animal_detection.dart';
import 'package:camera/camera.dart';
import 'package:file_selector/file_selector.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_litert/flutter_litert.dart' show CoverFitTransform;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:path_provider/path_provider.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:video_player/video_player.dart';

const Color _bodyColor = Color(0xFFFF9800);
const Color _poseColor = Color(0xFF00E676);
const int _cameraMaxDimension = 640;

String formatInferenceMilliseconds(num microseconds) {
  final milliseconds = microseconds / 1000;
  if (milliseconds < 10) return milliseconds.toStringAsFixed(3);
  if (milliseconds < 100) return milliseconds.toStringAsFixed(2);
  if (milliseconds < 1000) return milliseconds.toStringAsFixed(1);
  return milliseconds.toStringAsFixed(0);
}

class LiveInferenceStats {
  int? _latestUs;
  int _totalUs = 0;
  int _sampleCount = 0;
  int _generation = 0;

  int? get latestUs => _latestUs;
  int get sampleCount => _sampleCount;
  double? get averageUs => _sampleCount == 0 ? null : _totalUs / _sampleCount;

  int beginSample() => _generation;

  bool record(int sampleGeneration, int elapsedUs) {
    if (sampleGeneration != _generation) return false;
    _latestUs = elapsedUs;
    _totalUs += elapsedUs;
    _sampleCount++;
    return true;
  }

  void reset() {
    _latestUs = null;
    _totalUs = 0;
    _sampleCount = 0;
    _generation++;
  }
}

class LiveCameraMetrics extends StatelessWidget {
  final int fps;
  final int? latestInferenceUs;
  final double? averageInferenceUs;

  const LiveCameraMetrics({
    super.key,
    required this.fps,
    required this.latestInferenceUs,
    required this.averageInferenceUs,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        _FpsMetric(fps: fps),
        const _MetricDivider(),
        _InferenceMetric(label: 'LAST', microseconds: latestInferenceUs),
        const _MetricDivider(),
        _InferenceMetric(label: 'AVERAGE', microseconds: averageInferenceUs),
      ],
    );
  }
}

class _FpsMetric extends StatelessWidget {
  final int fps;

  const _FpsMetric({required this.fps});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 44,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text(
            'FPS',
            style: TextStyle(
              color: Colors.white60,
              fontSize: 9,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.7,
            ),
          ),
          Text(
            '$fps',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 14,
              fontWeight: FontWeight.w600,
              fontFeatures: [FontFeature.tabularFigures()],
            ),
          ),
        ],
      ),
    );
  }
}

class _InferenceMetric extends StatelessWidget {
  final String label;
  final num? microseconds;

  const _InferenceMetric({
    required this.label,
    required this.microseconds,
  });

  @override
  Widget build(BuildContext context) {
    final value =
        microseconds == null ? '—' : formatInferenceMilliseconds(microseconds!);
    return Semantics(
      label: microseconds == null
          ? '$label inference time unavailable'
          : '$label inference time $value milliseconds',
      child: SizedBox(
        width: 74,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              label,
              style: const TextStyle(
                color: Colors.white60,
                fontSize: 9,
                fontWeight: FontWeight.w600,
                letterSpacing: 0.7,
              ),
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                SizedBox(
                  width: 46,
                  child: FittedBox(
                    fit: BoxFit.scaleDown,
                    alignment: Alignment.centerRight,
                    child: Text(
                      value,
                      maxLines: 1,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                        fontFeatures: [FontFeature.tabularFigures()],
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 3),
                const SizedBox(
                  width: 18,
                  child: Text(
                    'ms',
                    style: TextStyle(color: Colors.white70, fontSize: 12),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _MetricDivider extends StatelessWidget {
  const _MetricDivider();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 1,
      height: 24,
      margin: const EdgeInsets.symmetric(horizontal: 5),
      color: Colors.white24,
    );
  }
}

class LiveCameraScreen extends StatefulWidget {
  const LiveCameraScreen({super.key});

  @override
  State<LiveCameraScreen> createState() => _LiveCameraScreenState();
}

class _LiveCameraScreenState extends State<LiveCameraScreen> {
  static const double _mobileTopBarExtent = 84;

  CameraController? _cameraController;
  List<CameraDescription> _availableCameras = const [];
  AnimalDetector? _detector;
  List<Animal> _animals = const [];
  Size? _imageSize;
  int? _sensorOrientation;
  bool _isFrontCamera = false;
  bool _isSwitchingCamera = false;
  bool _streamStarted = false;
  bool _isInitialized = false;
  bool _useCompiledModel = true;
  bool _enablePose = true;
  AnimalPoseModel _poseModel = AnimalPoseModel.rtmpose;
  DeviceOrientation _deviceOrientation = DeviceOrientation.portraitUp;
  StreamSubscription<AccelerometerEvent>? _accelerometerSub;
  final FrameThrottle _throttle = FrameThrottle();
  final FpsCounter _fpsCounter = FpsCounter();
  int _fps = 0;
  final LiveInferenceStats _inferenceStats = LiveInferenceStats();

  void _resetInferenceStats() {
    _inferenceStats.reset();
  }

  @override
  void initState() {
    super.initState();
    _initCamera();
    if (!kIsWeb && (Platform.isAndroid || Platform.isIOS)) {
      _accelerometerSub = accelerometerEventStream().listen((event) {
        final next = event.x.abs() > event.y.abs()
            ? (event.x > 0
                ? DeviceOrientation.landscapeLeft
                : DeviceOrientation.landscapeRight)
            : (event.y > 0
                ? DeviceOrientation.portraitUp
                : DeviceOrientation.portraitDown);
        if (next == DeviceOrientation.portraitDown &&
            (_deviceOrientation == DeviceOrientation.landscapeLeft ||
                _deviceOrientation == DeviceOrientation.landscapeRight)) {
          return;
        }
        if (next != _deviceOrientation && mounted) {
          setState(() => _deviceOrientation = next);
        }
      });
    }
  }

  Future<AnimalDetector> _createDetector(bool useCompiledModel) {
    return AnimalDetector.create(
      poseModel: _poseModel,
      enablePose: _enablePose,
      performanceConfig: const PerformanceConfig.xnnpack(),
      useCompiledModel: useCompiledModel,
    );
  }

  Future<void> _reinitDetector() async {
    final old = _detector;
    _detector = null;
    await old?.dispose();
    try {
      _detector = await _createDetector(_useCompiledModel);
    } catch (error) {
      if (!_useCompiledModel) rethrow;
      debugPrint(
        'Live camera CompiledModel init failed; falling back to XNNPACK: '
        '$error',
      );
      _useCompiledModel = false;
      _detector = await _createDetector(false);
    }
  }

  Future<void> _toggleBackend() async {
    setState(() {
      _isInitialized = false;
      _useCompiledModel = !_useCompiledModel;
      _resetInferenceStats();
    });
    try {
      await _reinitDetector();
    } catch (error) {
      if (mounted) _showError('Could not initialize detector: $error');
    }
    if (mounted) setState(() => _isInitialized = _detector?.isReady ?? false);
  }

  Future<void> _updateDetector(VoidCallback update) async {
    setState(() {
      _isInitialized = false;
      update();
      _animals = const [];
      _resetInferenceStats();
    });
    try {
      await _reinitDetector();
    } catch (error) {
      if (mounted) _showError('Could not update detector: $error');
    }
    if (mounted) setState(() => _isInitialized = _detector?.isReady ?? false);
  }

  Future<void> _initCamera() async {
    try {
      await _reinitDetector();
      if (mounted) setState(() => _isInitialized = true);
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        if (mounted) _showError('No cameras available');
        return;
      }
      _availableCameras = cameras;
      final camera = cameras.firstWhere(
        (value) => value.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );
      await _startController(camera);
    } catch (error, stack) {
      debugPrint('Camera init failed: $error\n$stack');
      if (mounted) _showError('Error initializing camera: $error');
    }
  }

  Future<void> _startController(CameraDescription camera) async {
    final controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    await controller.initialize();
    if (!mounted) {
      await controller.dispose();
      return;
    }
    setState(() {
      _cameraController = controller;
      _sensorOrientation = camera.sensorOrientation;
      _isFrontCamera = camera.lensDirection == CameraLensDirection.front;
    });
    await controller.startImageStream(_processCameraImage);
    _streamStarted = true;
  }

  bool get _canSwitchCamera {
    if (kIsWeb || !(Platform.isAndroid || Platform.isIOS)) return false;
    final front = _availableCameras.any(
      (value) => value.lensDirection == CameraLensDirection.front,
    );
    final back = _availableCameras.any(
      (value) => value.lensDirection == CameraLensDirection.back,
    );
    return front && back;
  }

  Future<void> _switchCamera() async {
    if (_isSwitchingCamera || !_canSwitchCamera) return;
    final target =
        _isFrontCamera ? CameraLensDirection.back : CameraLensDirection.front;
    final next = _availableCameras.firstWhere(
      (value) => value.lensDirection == target,
    );
    final previous = _cameraController;
    setState(() {
      _isSwitchingCamera = true;
      _cameraController = null;
      _animals = const [];
      _imageSize = null;
      _resetInferenceStats();
    });
    try {
      if (_streamStarted) {
        await previous?.stopImageStream();
        _streamStarted = false;
      }
      await previous?.dispose();
      await _startController(next);
    } catch (error) {
      if (mounted) _showError('Error switching camera: $error');
    } finally {
      if (mounted) setState(() => _isSwitchingCamera = false);
    }
  }

  DeviceOrientation _effectiveOrientation(BuildContext context) {
    return _cameraController?.value.deviceOrientation ??
        (MediaQuery.orientationOf(context) == Orientation.portrait
            ? DeviceOrientation.portraitUp
            : DeviceOrientation.landscapeLeft);
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (_fpsCounter.tick() && mounted) setState(() => _fps = _fpsCounter.fps);
    await _throttle.run(() async {
      final detector = _detector;
      if (detector == null || !_isInitialized || !mounted) return;
      try {
        final stopwatch = Stopwatch()..start();
        final statsGeneration = _inferenceStats.beginSample();
        final sensor = _sensorOrientation;
        final rotation = sensor == null
            ? null
            : rotationForFrame(
                width: image.width,
                height: image.height,
                sensorOrientation: sensor,
                isFrontCamera: _isFrontCamera,
                deviceOrientation: _effectiveOrientation(context),
              );
        final size = detectionSize(
          width: image.width,
          height: image.height,
          rotation: rotation,
          maxDim: _cameraMaxDimension,
        );
        final animals = await detector.detectFromCameraImage(
          image,
          rotation: rotation,
          isBgra: Platform.isMacOS,
          maxDim: _cameraMaxDimension,
        );
        stopwatch.stop();
        final detectionTimeUs = stopwatch.elapsedMicroseconds;
        _inferenceStats.record(statsGeneration, detectionTimeUs);
        if (mounted) {
          setState(() {
            _animals = animals;
            _imageSize = size;
          });
        }
      } catch (error) {
        debugPrint('Camera frame failed: $error');
      }
    });
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  @override
  void dispose() {
    _accelerometerSub?.cancel();
    if (_streamStarted) _cameraController?.stopImageStream();
    _cameraController?.dispose();
    _detector?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _cameraController;
    if (!_isInitialized || controller == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Live Animal Detection')),
        body: const Center(child: CircularProgressIndicator()),
      );
    }
    final effective = _effectiveOrientation(context);
    final portrait = effective == DeviceOrientation.portraitUp ||
        effective == DeviceOrientation.portraitDown;
    final aspect = portrait
        ? 1.0 / controller.value.aspectRatio
        : controller.value.aspectRatio;
    final turns = barQuarterTurns(_deviceOrientation);
    final mirror = (Platform.isAndroid && _isFrontCamera) || Platform.isWindows;
    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          Center(
            child: AspectRatio(
              aspectRatio: aspect,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  CameraPreview(controller),
                  if (_imageSize != null)
                    CustomPaint(
                      painter: _AnimalPainter(
                        animals: _animals,
                        imageSize: _imageSize!,
                        mirrorHorizontally: mirror,
                      ),
                    ),
                ],
              ),
            ),
          ),
          _positionedTopBar(turns),
        ],
      ),
    );
  }

  Widget _positionedTopBar(int turns) {
    final padding = MediaQuery.paddingOf(context);
    final bar = _cameraTopBar();
    final mobile = !kIsWeb && (Platform.isAndroid || Platform.isIOS);
    final barExtent = mobile ? _mobileTopBarExtent : kToolbarHeight;
    if (turns == 0) {
      return Positioned(
        top: padding.top,
        left: padding.left,
        right: padding.right,
        child: bar,
      );
    }
    return Positioned(
      top: padding.top,
      bottom: padding.bottom,
      left: turns == 3 ? padding.left : null,
      right: turns == 1 ? padding.right : null,
      width: barExtent,
      child: RotatedBox(quarterTurns: turns, child: bar),
    );
  }

  Widget _cameraTopBar() {
    final canPop = Navigator.of(context).canPop();
    final mobile = !kIsWeb && (Platform.isAndroid || Platform.isIOS);
    final metrics = LiveCameraMetrics(
      fps: _fps,
      latestInferenceUs: _inferenceStats.latestUs,
      averageInferenceUs: _inferenceStats.averageUs,
    );

    final controls = <Widget>[
      if (_canSwitchCamera)
        IconButton(
          tooltip: 'Switch camera',
          color: Colors.white,
          onPressed: _isSwitchingCamera ? null : _switchCamera,
          icon: Icon(
            Platform.isIOS ? Icons.flip_camera_ios : Icons.flip_camera_android,
          ),
        ),
      TextButton(
        onPressed: _isInitialized ? _toggleBackend : null,
        style: TextButton.styleFrom(
          minimumSize: const Size(92, 36),
          padding: const EdgeInsets.symmetric(horizontal: 4),
        ),
        child: Text(
          _useCompiledModel ? 'CM' : 'Interpreter',
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(
            color: Colors.amberAccent,
            fontWeight: FontWeight.bold,
            fontSize: 14,
          ),
        ),
      ),
      PopupMenuButton<void>(
        tooltip: 'Settings',
        icon: const Icon(Icons.settings, color: Colors.white),
        color: Colors.blueGrey,
        padding: EdgeInsets.zero,
        itemBuilder: (context) => [
          PopupMenuItem<void>(
            enabled: false,
            child: SizedBox(
              width: 250,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  SwitchListTile(
                    title: const Text(
                      'Body pose',
                      style: TextStyle(color: Colors.white),
                    ),
                    value: _enablePose,
                    onChanged: (value) =>
                        _updateDetector(() => _enablePose = value),
                  ),
                  DropdownButtonFormField<AnimalPoseModel>(
                    initialValue: _poseModel,
                    dropdownColor: Colors.blueGrey[800],
                    style: const TextStyle(color: Colors.white),
                    decoration: const InputDecoration(
                      labelText: 'Pose model',
                      labelStyle: TextStyle(color: Colors.white70),
                    ),
                    items: AnimalPoseModel.values
                        .map(
                          (value) => DropdownMenuItem(
                            value: value,
                            child: Text(value.name),
                          ),
                        )
                        .toList(),
                    onChanged: _enablePose
                        ? (value) {
                            if (value != null) {
                              _updateDetector(() => _poseModel = value);
                            }
                          }
                        : null,
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    ];

    return Material(
      color: Colors.black.withAlpha(179),
      elevation: 4,
      child: SizedBox(
        height: mobile ? _mobileTopBarExtent : kToolbarHeight,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 4),
          child: mobile
              ? Column(
                  children: [
                    SizedBox(
                      height: 48,
                      child: Row(
                        children: [
                          if (canPop)
                            IconButton(
                              tooltip: 'Back',
                              color: Colors.white,
                              onPressed: () => Navigator.maybePop(context),
                              icon: const Icon(Icons.arrow_back),
                            ),
                          const Spacer(),
                          ...controls,
                        ],
                      ),
                    ),
                    const Divider(height: 1, color: Colors.white12),
                    SizedBox(height: 35, child: Center(child: metrics)),
                  ],
                )
              : Row(
                  children: [
                    if (canPop)
                      IconButton(
                        tooltip: 'Back',
                        color: Colors.white,
                        onPressed: () => Navigator.maybePop(context),
                        icon: const Icon(Icons.arrow_back),
                      ),
                    const Expanded(
                      child: Padding(
                        padding: EdgeInsets.symmetric(horizontal: 8),
                        child: Text(
                          'Live Animal Detection',
                          style: TextStyle(color: Colors.white, fontSize: 18),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    ),
                    ...controls,
                    const SizedBox(width: 8),
                    metrics,
                  ],
                ),
        ),
      ),
    );
  }
}

class _AnimalPainter extends CustomPainter {
  const _AnimalPainter({
    required this.animals,
    required this.imageSize,
    this.mirrorHorizontally = false,
  });

  final List<Animal> animals;
  final Size imageSize;
  final bool mirrorHorizontally;

  @override
  void paint(Canvas canvas, Size size) {
    if (imageSize.isEmpty) return;
    final t = CoverFitTransform.cover(
      sourceWidth: imageSize.width,
      sourceHeight: imageSize.height,
      viewWidth: size.width,
      viewHeight: size.height,
      mirror: mirrorHorizontally,
    );
    final bodyPaint = Paint()
      ..color = _bodyColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    final posePaint = Paint()
      ..color = _poseColor
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;
    for (final animal in animals) {
      final box = animal.boundingBox;
      final p1 = t.map(box.left, box.top);
      final p2 = t.map(box.right, box.bottom);
      final rect = Rect.fromLTRB(
        math.min(p1.dx, p2.dx),
        math.min(p1.dy, p2.dy),
        math.max(p1.dx, p2.dx),
        math.max(p1.dy, p2.dy),
      );
      canvas.drawRect(
        rect,
        bodyPaint,
      );
      final pose = animal.pose;
      if (pose != null) {
        for (final connection in animalPoseConnections) {
          final a = pose.getLandmark(connection[0]);
          final b = pose.getLandmark(connection[1]);
          if (a == null ||
              b == null ||
              a.confidence < 0.3 ||
              b.confidence < 0.3) {
            continue;
          }
          canvas.drawLine(
            t.map(a.x, a.y),
            t.map(b.x, b.y),
            posePaint,
          );
        }
        for (final landmark in pose.landmarks) {
          if (landmark.confidence >= 0.3) {
            canvas.drawCircle(
              t.map(landmark.x, landmark.y),
              3,
              posePaint,
            );
          }
        }
      }
      _paintLabel(canvas, animal, rect);
    }
  }

  void _paintLabel(Canvas canvas, Animal animal, Rect boundingBox) {
    final species = animal.species ?? 'Animal';
    final confidence = (animal.score * 100).toStringAsFixed(0);
    final painter = TextPainter(
      text: TextSpan(
        text: '$species $confidence%',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 13,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    final left = boundingBox.left;
    final top = math.max(
      0.0,
      boundingBox.top - painter.height - 6,
    );
    canvas.drawRect(
      Rect.fromLTWH(left, top, painter.width + 8, painter.height + 6),
      Paint()..color = _bodyColor,
    );
    painter.paint(canvas, Offset(left + 4, top + 3));
  }

  @override
  bool shouldRepaint(covariant _AnimalPainter oldDelegate) =>
      oldDelegate.animals != animals ||
      oldDelegate.imageSize != imageSize ||
      oldDelegate.mirrorHorizontally != mirrorHorizontally;
}

class VideoFileScreen extends StatefulWidget {
  const VideoFileScreen({super.key});

  @override
  State<VideoFileScreen> createState() => _VideoFileScreenState();
}

class _VideoFileScreenState extends State<VideoFileScreen> {
  AnimalDetector? _detector;
  bool _isInitialized = false;
  bool _isProcessing = false;
  bool _cancelRequested = false;
  bool _useCompiledModel = true;
  bool _enablePose = true;
  AnimalPoseModel _poseModel = AnimalPoseModel.rtmpose;
  bool _smoothingEnabled = true;
  final _AnimalSmoother _smoother = _AnimalSmoother();
  String? _inputPath;
  String? _outputPath;
  String? _errorMessage;
  String? _statusMessage;
  int _totalFrames = 0;
  int _processedFrames = 0;
  double _videoFps = 0;
  int _videoWidth = 0;
  int _videoHeight = 0;
  Duration _elapsed = Duration.zero;
  final Stopwatch _wallClock = Stopwatch();
  VideoPlayerController? _playerController;
  bool _playerReady = false;
  String? _playerError;

  bool get _supportsInAppPlayer =>
      kIsWeb || Platform.isAndroid || Platform.isIOS || Platform.isMacOS;

  @override
  void initState() {
    super.initState();
    _initDetector();
  }

  Future<AnimalDetector> _createDetector(bool useCompiledModel) {
    return AnimalDetector.create(
      poseModel: _poseModel,
      enablePose: _enablePose,
      performanceConfig: const PerformanceConfig.xnnpack(),
      useCompiledModel: useCompiledModel,
    );
  }

  Future<void> _initDetector() async {
    try {
      _detector = await _createDetector(_useCompiledModel);
    } catch (error) {
      if (!_useCompiledModel) {
        if (mounted) setState(() => _errorMessage = '$error');
        return;
      }
      _useCompiledModel = false;
      try {
        _detector = await _createDetector(false);
      } catch (fallbackError) {
        if (mounted) setState(() => _errorMessage = '$fallbackError');
        return;
      }
    }
    if (mounted) setState(() => _isInitialized = true);
  }

  Future<void> _reinitDetector() async {
    setState(() => _isInitialized = false);
    final old = _detector;
    _detector = null;
    await old?.dispose();
    await _initDetector();
  }

  Future<void> _toggleBackend() async {
    if (_isProcessing) return;
    _useCompiledModel = !_useCompiledModel;
    await _reinitDetector();
  }

  @override
  void dispose() {
    _cancelRequested = true;
    _detector?.dispose();
    _playerController?.dispose();
    super.dispose();
  }

  Future<void> _pickVideo() async {
    const group = XTypeGroup(
      label: 'Videos',
      extensions: ['mp4', 'mov', 'm4v'],
    );
    final file = await openFile(acceptedTypeGroups: [group]);
    if (file != null) await _processVideo(file.path);
  }

  Future<void> _processVideo(String path) async {
    final detector = _detector;
    if (detector == null) return;
    final cap = cv.VideoCapture.fromFile(path);
    if (!cap.isOpened) {
      cap.release();
      setState(() => _errorMessage = 'Could not open video: $path');
      return;
    }
    final fps = cap.get(cv.CAP_PROP_FPS);
    final width = cap.get(cv.CAP_PROP_FRAME_WIDTH).toInt();
    final height = cap.get(cv.CAP_PROP_FRAME_HEIGHT).toInt();
    final total = cap.get(cv.CAP_PROP_FRAME_COUNT).toInt();
    final docs = await getApplicationDocumentsDirectory();
    final output =
        '${docs.path}/animal_${DateTime.now().millisecondsSinceEpoch}.mp4';
    final writer = cv.VideoWriter.fromFile(
      output,
      'avc1',
      fps > 0 ? fps : 30,
      (width, height),
    );
    if (!writer.isOpened) {
      cap.release();
      setState(() {
        _errorMessage = 'Could not create H.264 output video at $output';
      });
      return;
    }
    await _disposePlayer();
    if (!mounted) {
      cap.release();
      writer.release();
      return;
    }
    setState(() {
      _inputPath = path;
      _outputPath = output;
      _videoFps = fps;
      _videoWidth = width;
      _videoHeight = height;
      _totalFrames = total;
      _processedFrames = 0;
      _isProcessing = true;
      _cancelRequested = false;
      _errorMessage = null;
      _statusMessage = 'Processing...';
      _elapsed = Duration.zero;
    });
    _smoother.reset();
    _wallClock
      ..reset()
      ..start();
    cv.Mat? frame;
    var index = 0;
    try {
      while (mounted && !_cancelRequested) {
        final read = cap.read(m: frame);
        if (!read.$1 || read.$2.isEmpty) break;
        frame = read.$2;
        final raw = await detector.detectFromMat(
          frame,
          imageWidth: frame.cols,
          imageHeight: frame.rows,
        );
        final time = fps > 0 ? index / fps : index / 30;
        final animals = _smoother.apply(raw, time);
        _drawAnimals(frame, animals);
        writer.write(frame);
        index++;
        if (index % 4 == 0) {
          if (!mounted) break;
          setState(() {
            _processedFrames = index;
            _elapsed = _wallClock.elapsed;
          });
          await Future<void>.delayed(Duration.zero);
        }
      }
      if (mounted) {
        setState(() {
          _processedFrames = index;
          _elapsed = _wallClock.elapsed;
          _statusMessage = _cancelRequested
              ? 'Cancelled after $index frames.'
              : 'Done. Wrote $index frames to:\n$output';
        });
      }
    } catch (error) {
      if (mounted) setState(() => _errorMessage = 'Processing failed: $error');
    } finally {
      _wallClock.stop();
      cap.release();
      writer.release();
      frame?.dispose();
      if (mounted) setState(() => _isProcessing = false);
      if (mounted && !_cancelRequested) await _initPlayer(output);
    }
  }

  void _drawAnimals(cv.Mat frame, List<Animal> animals) {
    final body = cv.Scalar(0, 152, 255);
    final pose = cv.Scalar(118, 230, 0);
    final black = cv.Scalar(0, 0, 0);
    for (final animal in animals) {
      final box = animal.boundingBox;
      final left = box.left.round().clamp(0, frame.cols - 1);
      final top = box.top.round().clamp(0, frame.rows - 1);
      final right = box.right.round().clamp(0, frame.cols - 1);
      final bottom = box.bottom.round().clamp(0, frame.rows - 1);
      cv.rectangle(
        frame,
        cv.Rect(
          left,
          top,
          math.max(1, right - left),
          math.max(1, bottom - top),
        ),
        body,
        thickness: 2,
      );
      final animalPose = animal.pose;
      if (animalPose != null) {
        for (final connection in animalPoseConnections) {
          final a = animalPose.getLandmark(connection[0]);
          final b = animalPose.getLandmark(connection[1]);
          if (a == null ||
              b == null ||
              a.confidence < 0.3 ||
              b.confidence < 0.3) {
            continue;
          }
          cv.line(
            frame,
            cv.Point(a.x.round(), a.y.round()),
            cv.Point(b.x.round(), b.y.round()),
            pose,
            thickness: 3,
          );
        }
        for (final landmark in animalPose.landmarks) {
          if (landmark.confidence >= 0.3) {
            cv.circle(
              frame,
              cv.Point(landmark.x.round(), landmark.y.round()),
              3,
              pose,
              thickness: -1,
            );
          }
        }
      }
      final label =
          '${animal.species ?? 'Animal'} ${(animal.score * 100).round()}%';
      final (textSize, _) = cv.getTextSize(
        label,
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        2,
      );
      final labelTop = math.max(0, top - textSize.height - 8);
      cv.rectangle(
        frame,
        cv.Rect(
          left,
          labelTop,
          math.min(frame.cols - left, textSize.width + 8),
          textSize.height + 8,
        ),
        body,
        thickness: -1,
      );
      cv.putText(
        frame,
        label,
        cv.Point(left + 4, labelTop + textSize.height + 2),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        black,
        thickness: 2,
      );
    }
  }

  Future<void> _disposePlayer() async {
    final player = _playerController;
    _playerController = null;
    _playerReady = false;
    _playerError = null;
    await player?.dispose();
  }

  Future<void> _initPlayer(String path) async {
    await _disposePlayer();
    if (!_supportsInAppPlayer) return;
    final player = VideoPlayerController.file(File(path));
    _playerController = player;
    try {
      await player.initialize();
      await player.setLooping(true);
      if (!mounted) return;
      setState(() => _playerReady = true);
      await player.play();
    } catch (error) {
      if (mounted) {
        setState(() => _playerError = 'Could not load video: $error');
      }
    }
  }

  Future<void> _openOutput() async {
    final path = _outputPath;
    if (path == null) return;
    if (Platform.isMacOS) {
      await Process.run('open', [path]);
    } else if (Platform.isLinux) {
      await Process.run('xdg-open', [path]);
    } else if (Platform.isWindows) {
      await Process.run('cmd', ['/c', 'start', '', path]);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Video File - Animal Detection'),
        actions: [
          TextButton(
            onPressed: _isInitialized && !_isProcessing ? _toggleBackend : null,
            child: Text(
              _useCompiledModel ? 'CM' : 'Interpreter',
              style: const TextStyle(
                color: Colors.amber,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          PopupMenuButton<void>(
            tooltip: 'Settings',
            icon: const Icon(Icons.tune),
            itemBuilder: (context) => [
              PopupMenuItem<void>(
                enabled: false,
                child: SizedBox(
                  width: 260,
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      SwitchListTile(
                        title: const Text('Body pose'),
                        value: _enablePose,
                        onChanged: _isProcessing
                            ? null
                            : (value) {
                                _enablePose = value;
                                _reinitDetector();
                              },
                      ),
                      DropdownButtonFormField<AnimalPoseModel>(
                        initialValue: _poseModel,
                        decoration: const InputDecoration(
                          labelText: 'Pose model',
                        ),
                        items: AnimalPoseModel.values
                            .map(
                              (value) => DropdownMenuItem(
                                value: value,
                                child: Text(value.name),
                              ),
                            )
                            .toList(),
                        onChanged: _isProcessing || !_enablePose
                            ? null
                            : (value) {
                                if (value != null) {
                                  _poseModel = value;
                                  _reinitDetector();
                                }
                              },
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
      body: _buildBody(),
      floatingActionButton: _isInitialized && !_isProcessing
          ? FloatingActionButton.extended(
              onPressed: _pickVideo,
              icon: const Icon(Icons.video_file),
              label: const Text('Pick Video'),
            )
          : (_isProcessing
              ? FloatingActionButton.extended(
                  onPressed: () => setState(() => _cancelRequested = true),
                  icon: const Icon(Icons.cancel),
                  label: const Text('Cancel'),
                )
              : null),
    );
  }

  Widget _buildBody() {
    if (!_isInitialized && _errorMessage == null) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Initializing detector...'),
          ],
        ),
      );
    }
    final progress = _totalFrames > 0
        ? (_processedFrames / _totalFrames).clamp(0.0, 1.0)
        : 0.0;
    final processedFps = _elapsed.inMilliseconds > 0
        ? _processedFrames * 1000 / _elapsed.inMilliseconds
        : 0.0;
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (_errorMessage != null)
            Card(
              color: Colors.red[50],
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Text(_errorMessage!),
              ),
            ),
          if (_inputPath != null) ...[
            _infoRow('Input', _inputPath!),
            _infoRow(
              'Source',
              '$_videoWidth×$_videoHeight @ '
                  '${_videoFps.toStringAsFixed(2)} fps · $_totalFrames frames',
            ),
          ],
          if (!_isProcessing)
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('Smoothing (One-Euro filter)'),
              subtitle: Text(
                _smoothingEnabled
                    ? 'On: pose landmarks filtered across frames'
                    : 'Off: raw per-frame detections',
              ),
              value: _smoothingEnabled,
              onChanged: (value) {
                setState(() {
                  _smoothingEnabled = value;
                  _smoother.enabled = value;
                  _smoother.reset();
                });
              },
            ),
          if (!_isProcessing && _inputPath != null)
            Align(
              alignment: Alignment.centerLeft,
              child: OutlinedButton.icon(
                onPressed: () => _processVideo(_inputPath!),
                icon: const Icon(Icons.refresh),
                label: const Text('Re-run with current settings'),
              ),
            ),
          const SizedBox(height: 16),
          if (_isProcessing) ...[
            LinearProgressIndicator(value: _totalFrames > 0 ? progress : null),
            const SizedBox(height: 8),
            Text(
              'Frame $_processedFrames / $_totalFrames · '
              '${(progress * 100).toStringAsFixed(1)}% · '
              '${processedFps.toStringAsFixed(1)} fps · '
              'elapsed ${_formatDuration(_elapsed)}',
              style: const TextStyle(
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
          ] else if (_outputPath != null && _statusMessage != null)
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Icon(Icons.check_circle, color: Colors.green),
                        const SizedBox(width: 8),
                        Expanded(child: Text(_statusMessage!)),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Total time: ${_formatDuration(_elapsed)} '
                      '(${processedFps.toStringAsFixed(1)} fps avg)',
                    ),
                    const SizedBox(height: 12),
                    _buildPreview(),
                    if (Platform.isMacOS ||
                        Platform.isLinux ||
                        Platform.isWindows) ...[
                      const SizedBox(height: 12),
                      ElevatedButton.icon(
                        onPressed: _openOutput,
                        icon: const Icon(Icons.play_circle_outline),
                        label: const Text('Open output video'),
                      ),
                    ],
                  ],
                ),
              ),
            )
          else
            Center(
              child: Column(
                children: [
                  const SizedBox(height: 32),
                  Icon(
                    Icons.movie_creation_outlined,
                    size: 96,
                    color: Colors.grey[400],
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    'Pick an MP4 to run animal detection on every frame.\n'
                    'Output is written to the app documents directory.',
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Widget _infoRow(String label, String value) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 2),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              width: 70,
              child: Text(
                '$label:',
                style: const TextStyle(fontWeight: FontWeight.w600),
              ),
            ),
            Expanded(child: SelectableText(value)),
          ],
        ),
      );

  String _formatDuration(Duration duration) {
    final minutes = duration.inMinutes.remainder(60).toString().padLeft(2, '0');
    final seconds = duration.inSeconds.remainder(60).toString().padLeft(2, '0');
    return duration.inHours > 0
        ? '${duration.inHours}:$minutes:$seconds'
        : '$minutes:$seconds';
  }

  Widget _buildPreview() {
    if (!_supportsInAppPlayer) return const SizedBox.shrink();
    if (_playerError != null) {
      return Text(_playerError!, style: const TextStyle(color: Colors.red));
    }
    final player = _playerController;
    if (player == null || !_playerReady) {
      return const SizedBox(
        height: 64,
        child: Center(child: CircularProgressIndicator()),
      );
    }
    return _OutputVideoPlayer(controller: player);
  }
}

class _OutputVideoPlayer extends StatefulWidget {
  const _OutputVideoPlayer({required this.controller});

  final VideoPlayerController controller;

  @override
  State<_OutputVideoPlayer> createState() => _OutputVideoPlayerState();
}

class _OutputVideoPlayerState extends State<_OutputVideoPlayer> {
  void _tick() {
    if (mounted) setState(() {});
  }

  @override
  void initState() {
    super.initState();
    widget.controller.addListener(_tick);
  }

  @override
  void dispose() {
    widget.controller.removeListener(_tick);
    super.dispose();
  }

  String _format(Duration duration) {
    final minutes = duration.inMinutes.remainder(60).toString().padLeft(2, '0');
    final seconds = duration.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$minutes:$seconds';
  }

  @override
  Widget build(BuildContext context) {
    final value = widget.controller.value;
    final maxHeight = math.max(120.0, MediaQuery.sizeOf(context).height * 0.45);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Align(
          alignment: Alignment.centerLeft,
          child: ConstrainedBox(
            constraints: BoxConstraints(maxHeight: maxHeight),
            child: AspectRatio(
              aspectRatio: value.aspectRatio == 0 ? 16 / 9 : value.aspectRatio,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: VideoPlayer(widget.controller),
              ),
            ),
          ),
        ),
        Row(
          children: [
            IconButton(
              icon: Icon(value.isPlaying ? Icons.pause : Icons.play_arrow),
              onPressed: () => value.isPlaying
                  ? widget.controller.pause()
                  : widget.controller.play(),
            ),
            Expanded(
              child: VideoProgressIndicator(
                widget.controller,
                allowScrubbing: true,
                padding: const EdgeInsets.symmetric(vertical: 12),
              ),
            ),
            const SizedBox(width: 8),
            Text(
              '${_format(value.position)} / ${_format(value.duration)}',
              style: const TextStyle(
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
          ],
        ),
      ],
    );
  }
}

class _AnimalSmoother {
  bool enabled = true;
  final List<_AnimalTrack> _tracks = [];

  void reset() => _tracks.clear();

  List<Animal> apply(List<Animal> animals, double time) {
    if (!enabled) {
      _tracks.clear();
      return animals;
    }
    final unmatched = List<int>.generate(_tracks.length, (index) => index);
    final output = <Animal>[];
    for (final animal in animals) {
      var bestTrack = -1;
      var bestIou = 0.2;
      for (final index in unmatched) {
        final iou = _tracks[index].iou(animal.boundingBox);
        if (iou > bestIou) {
          bestIou = iou;
          bestTrack = index;
        }
      }
      final track = bestTrack < 0 ? _AnimalTrack() : _tracks[bestTrack];
      if (bestTrack < 0) {
        _tracks.add(track);
      } else {
        unmatched.remove(bestTrack);
      }
      track.update(animal.boundingBox);
      output.add(track.smooth(animal, time));
    }
    for (final index in unmatched) {
      _tracks[index].missed++;
    }
    _tracks.removeWhere((track) => track.missed > 5);
    return output;
  }
}

class _AnimalTrack {
  final Map<AnimalPoseLandmarkType, List<OneEuroFilter>> filters = {};
  double left = 0;
  double top = 0;
  double right = 0;
  double bottom = 0;
  bool hasBox = false;
  int missed = 0;

  double iou(BoundingBox box) => hasBox
      ? iouLTRB(
          box.left,
          box.top,
          box.right,
          box.bottom,
          left,
          top,
          right,
          bottom,
        )
      : 0;

  void update(BoundingBox box) {
    left = box.left;
    top = box.top;
    right = box.right;
    bottom = box.bottom;
    hasBox = true;
    missed = 0;
  }

  Animal smooth(Animal animal, double time) {
    final pose = animal.pose;
    if (pose == null) return animal;
    final landmarks = pose.landmarks.map((landmark) {
      final pair = filters.putIfAbsent(
        landmark.type,
        () => [
          OneEuroFilter(minCutoff: 1, beta: 0.1, dCutoff: 1),
          OneEuroFilter(minCutoff: 1, beta: 0.1, dCutoff: 1),
        ],
      );
      return AnimalPoseLandmark(
        type: landmark.type,
        x: pair[0].filter(landmark.x, time),
        y: pair[1].filter(landmark.y, time),
        confidence: landmark.confidence,
      );
    }).toList();
    return Animal(
      boundingBox: animal.boundingBox,
      score: animal.score,
      species: animal.species,
      breed: animal.breed,
      speciesConfidence: animal.speciesConfidence,
      pose: AnimalPose(landmarks: landmarks),
      imageWidth: animal.imageWidth,
      imageHeight: animal.imageHeight,
    );
  }
}
