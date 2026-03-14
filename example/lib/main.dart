import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:file_selector/file_selector.dart';
import 'package:animal_detection/animal_detection.dart';

void main() {
  runApp(const AnimalDetectionApp());
}

class AnimalDetectionApp extends StatelessWidget {
  const AnimalDetectionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Animal Detection Demo',
      theme: ThemeData(
        colorSchemeSeed: Colors.brown,
        useMaterial3: true,
      ),
      home: const AnimalDetectionHome(),
    );
  }
}

class AnimalDetectionHome extends StatelessWidget {
  const AnimalDetectionHome({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Animal Detection Demo'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.cruelty_free, size: 100, color: Colors.brown[300]),
            const SizedBox(height: 48),
            Text(
              'Animal Detection',
              style: Theme.of(context).textTheme.headlineMedium,
            ),
            const SizedBox(height: 16),
            Text(
              'Detect animals with species classification and body pose',
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Colors.grey[600],
                  ),
            ),
            const SizedBox(height: 48),
            SizedBox(
              width: 400,
              child: Card(
                elevation: 4,
                child: InkWell(
                  onTap: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => const StillImageScreen(),
                      ),
                    );
                  },
                  borderRadius: BorderRadius.circular(12),
                  child: Padding(
                    padding: const EdgeInsets.all(24),
                    child: Row(
                      children: [
                        const Icon(Icons.image, size: 64, color: Colors.brown),
                        const SizedBox(width: 24),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Still Image',
                                style: Theme.of(context).textTheme.titleLarge,
                              ),
                              const SizedBox(height: 8),
                              Text(
                                'Detect animals in photos from gallery, camera, or samples',
                                style: Theme.of(context)
                                    .textTheme
                                    .bodyMedium
                                    ?.copyWith(color: Colors.grey[600]),
                              ),
                            ],
                          ),
                        ),
                        const Icon(Icons.arrow_forward_ios),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class StillImageScreen extends StatefulWidget {
  const StillImageScreen({super.key});

  @override
  State<StillImageScreen> createState() => _StillImageScreenState();
}

class _StillImageScreenState extends State<StillImageScreen> {
  AnimalDetector? _detector;
  final ImagePicker _picker = ImagePicker();

  AnimalPoseModel _poseModel = AnimalPoseModel.rtmpose;
  bool _enablePose = true;
  bool _isInitialized = false;
  bool _isProcessing = false;
  bool _isDownloading = false;
  String _downloadStatus = '';
  Uint8List? _imageBytes;
  int _imageWidth = 0;
  int _imageHeight = 0;
  List<Animal> _results = [];
  String? _errorMessage;

  static const List<String> _samplePaths = [
    'packages/animal_detection/assets/samples/sample_animal_1.png',
    'packages/animal_detection/assets/samples/sample_animal_2.png',
    'packages/animal_detection/assets/samples/sample_animal_3.png',
    'packages/animal_detection/assets/samples/sample_animal_4.png',
    'packages/animal_detection/assets/samples/sample_animal_5.png',
  ];
  int _currentSampleIndex = 0;

  @override
  void initState() {
    super.initState();
    _initializeDetector();
  }

  Future<void> _initializeDetector() async {
    setState(() {
      _isProcessing = true;
      _isInitialized = false;
      _errorMessage = null;
    });

    try {
      await _detector?.dispose();
      _detector = AnimalDetector(
        poseModel: _poseModel,
        enablePose: _enablePose,
        performanceConfig: PerformanceConfig.disabled,
      );

      if (_poseModel == AnimalPoseModel.hrnet) {
        final cached = await AnimalDetector.isHrnetCached();
        if (!cached) {
          setState(() {
            _isDownloading = true;
            _downloadStatus = 'Downloading HRNet model...';
          });
        }
      }

      await _detector!.initialize(
        onDownloadProgress: (model, received, total) {
          if (!mounted) return;
          final mb = (received / 1024 / 1024).toStringAsFixed(1);
          final totalMb =
              total > 0 ? (total / 1024 / 1024).toStringAsFixed(1) : '?';
          setState(() {
            _downloadStatus = 'Downloading HRNet: $mb / $totalMb MB';
          });
        },
      );

      if (!mounted) return;
      setState(() {
        _isInitialized = true;
        _isProcessing = false;
        _isDownloading = false;
        _downloadStatus = '';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isProcessing = false;
        _isDownloading = false;
        _downloadStatus = '';
        _errorMessage = 'Failed to initialize: $e';
      });
    }
  }

  Future<void> _changePoseModel(AnimalPoseModel model) async {
    if (model == _poseModel) return;
    setState(() {
      _poseModel = model;
      _results = [];
    });
    await _initializeDetector();
    if (_imageBytes != null && _isInitialized) {
      await _runDetection(_imageBytes!);
    }
  }

  Future<void> _togglePose(bool value) async {
    if (value == _enablePose) return;
    setState(() {
      _enablePose = value;
      _results = [];
    });
    await _initializeDetector();
    if (_imageBytes != null && _isInitialized) {
      await _runDetection(_imageBytes!);
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(source: source);
      if (pickedFile == null) return;

      final Uint8List bytes = await pickedFile.readAsBytes();
      await _runDetection(bytes);
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _errorMessage = 'Error: $e';
      });
    }
  }

  Future<void> _pickFileFromSystem() async {
    try {
      const XTypeGroup typeGroup = XTypeGroup(
        label: 'images',
        extensions: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'],
      );
      final XFile? file = await openFile(acceptedTypeGroups: [typeGroup]);
      if (file == null) return;

      final Uint8List bytes = await File(file.path).readAsBytes();
      await _runDetection(bytes);
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _errorMessage = 'Error: $e';
      });
    }
  }

  bool get _isDesktop =>
      !kIsWeb && (Platform.isMacOS || Platform.isWindows || Platform.isLinux);

  Future<void> _loadSample() async {
    try {
      final String path = _samplePaths[_currentSampleIndex];
      _currentSampleIndex = (_currentSampleIndex + 1) % _samplePaths.length;

      final ByteData data = await rootBundle.load(path);
      final Uint8List bytes = data.buffer.asUint8List();
      await _runDetection(bytes);
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _errorMessage = 'Error loading sample: $e';
      });
    }
  }

  Future<void> _runDetection(Uint8List bytes) async {
    setState(() {
      _isProcessing = true;
      _errorMessage = null;
      _results = [];
    });

    try {
      final List<Animal> results = await _detector!.detect(bytes);

      int imgW = 0;
      int imgH = 0;
      if (results.isNotEmpty) {
        imgW = results.first.imageWidth;
        imgH = results.first.imageHeight;
      } else {
        final decoded = await decodeImageFromList(bytes);
        imgW = decoded.width;
        imgH = decoded.height;
      }

      setState(() {
        _imageBytes = bytes;
        _imageWidth = imgW;
        _imageHeight = imgH;
        _results = results;
        _isProcessing = false;
        if (results.isEmpty) {
          _errorMessage = 'No animals detected in image';
        }
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _errorMessage = 'Error: $e';
      });
    }
  }

  @override
  void dispose() {
    _detector?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Animal Detection'),
        actions: [
          if (_isInitialized && _results.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.info_outline),
              onPressed: _showDetectionInfo,
            ),
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: _showSettings,
          ),
        ],
      ),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_isDownloading) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 48),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const CircularProgressIndicator(),
              const SizedBox(height: 24),
              Text(
                _downloadStatus,
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodyLarge,
              ),
              const SizedBox(height: 8),
              Text(
                'This is a one-time download. The model will be cached for future use.',
                textAlign: TextAlign.center,
                style: Theme.of(context)
                    .textTheme
                    .bodySmall
                    ?.copyWith(color: Colors.grey[600]),
              ),
            ],
          ),
        ),
      );
    }

    if (!_isInitialized && _isProcessing) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Initializing animal detector...'),
          ],
        ),
      );
    }

    if (_errorMessage != null && _imageBytes == null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, size: 64, color: Colors.red),
            const SizedBox(height: 16),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 32),
              child: Text(
                _errorMessage!,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.red),
              ),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _initializeDetector,
              child: const Text('Retry'),
            ),
          ],
        ),
      );
    }

    if (_imageBytes == null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.cruelty_free, size: 100, color: Colors.grey[400]),
            const SizedBox(height: 24),
            Text(
              'Select an image to detect animals',
              style: TextStyle(fontSize: 18, color: Colors.grey[600]),
            ),
            const SizedBox(height: 24),
            _buildActionButtons(),
          ],
        ),
      );
    }

    return SingleChildScrollView(
      child: Column(
        children: [
          AnimalVisualizerWidget(
            imageBytes: _imageBytes!,
            imageWidth: _imageWidth,
            imageHeight: _imageHeight,
            results: _results,
          ),
          if (_isProcessing)
            const Padding(
              padding: EdgeInsets.all(16),
              child: Column(
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 8),
                  Text('Detecting animals...'),
                ],
              ),
            ),
          if (_errorMessage != null && !_isProcessing)
            Padding(
              padding: const EdgeInsets.all(16),
              child: Card(
                color: Colors.orange[50],
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(
                    children: [
                      const Icon(Icons.info_outline, color: Colors.orange),
                      const SizedBox(width: 8),
                      Expanded(child: Text(_errorMessage!)),
                    ],
                  ),
                ),
              ),
            ),
          if (_results.isNotEmpty)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Detected: ${_results.length} animal${_results.length > 1 ? 's' : ''}',
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(
                              color: Colors.green,
                              fontWeight: FontWeight.bold,
                            ),
                      ),
                      const SizedBox(height: 8),
                      for (final animal in _results) ...[
                        Text(
                          'Score: ${(animal.score * 100).toStringAsFixed(1)}%',
                          style: Theme.of(context).textTheme.bodyMedium,
                        ),
                        if (animal.species != null)
                          Text(
                            'Species: ${animal.species}',
                            style: Theme.of(context).textTheme.bodyMedium,
                          ),
                        if (animal.breed != null)
                          Text(
                            'Breed: ${animal.breed}',
                            style: Theme.of(context).textTheme.bodyMedium,
                          ),
                        if (animal.pose != null)
                          Text(
                            'Pose landmarks: ${animal.pose!.landmarks.length}',
                            style: Theme.of(context).textTheme.bodyMedium,
                          ),
                        if (_results.length > 1 && animal != _results.last)
                          const Divider(),
                      ],
                    ],
                  ),
                ),
              ),
            ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: _buildActionButtons(),
          ),
        ],
      ),
    );
  }

  Widget _buildActionButtons() {
    return Wrap(
      spacing: 12,
      runSpacing: 12,
      alignment: WrapAlignment.center,
      children: [
        ElevatedButton.icon(
          onPressed: _isInitialized && !_isProcessing
              ? () => _isDesktop
                  ? _pickFileFromSystem()
                  : _pickImage(ImageSource.gallery)
              : null,
          icon: const Icon(Icons.photo_library),
          label: Text(_isDesktop ? 'Open File' : 'Gallery'),
        ),
        if (!_isDesktop)
          ElevatedButton.icon(
            onPressed: _isInitialized && !_isProcessing
                ? () => _pickImage(ImageSource.camera)
                : null,
            icon: const Icon(Icons.camera_alt),
            label: const Text('Camera'),
          ),
        ElevatedButton.icon(
          onPressed: _isInitialized && !_isProcessing ? _loadSample : null,
          icon: const Icon(Icons.auto_awesome),
          label: const Text('Load Sample'),
        ),
      ],
    );
  }

  void _showSettings() {
    showModalBottomSheet(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setSheetState) => Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Settings',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              const SizedBox(height: 24),
              SwitchListTile(
                title: const Text('Body Pose Estimation'),
                subtitle: Text(
                  _enablePose
                      ? '24 SuperAnimal body keypoints enabled'
                      : 'Detection and species only',
                ),
                secondary: const Icon(Icons.accessibility_new),
                value: _enablePose,
                onChanged: _isDownloading
                    ? null
                    : (value) {
                        setSheetState(() {});
                        Navigator.pop(context);
                        _togglePose(value);
                      },
              ),
              const Divider(),
              Text(
                'Pose Model',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 8),
              RadioGroup<AnimalPoseModel>(
                groupValue: _poseModel,
                onChanged: (value) {
                  if (!_enablePose || _isDownloading || value == null) return;
                  setSheetState(() {});
                  Navigator.pop(context);
                  _changePoseModel(value);
                },
                child: Column(
                  children: [
                    RadioListTile<AnimalPoseModel>(
                      title: const Text('RTMPose-S'),
                      subtitle:
                          const Text('11.6 MB, bundled. Fast SimCC decoder.'),
                      value: AnimalPoseModel.rtmpose,
                    ),
                    RadioListTile<AnimalPoseModel>(
                      title: const Text('HRNet-w32'),
                      subtitle: const Text(
                          '54.6 MB, downloaded on demand. Most accurate.'),
                      value: AnimalPoseModel.hrnet,
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }

  void _showDetectionInfo() {
    if (_results.isEmpty) return;

    showModalBottomSheet(
      context: context,
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.6,
        minChildSize: 0.4,
        maxChildSize: 0.95,
        expand: false,
        builder: (context, scrollController) => ListView(
          controller: scrollController,
          padding: const EdgeInsets.all(16),
          children: [
            Text(
              'Detection Details',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 16),
            for (final animal in _results) ...[
              Text(
                '${animal.species ?? "Animal"} (score: ${(animal.score * 100).toStringAsFixed(1)}%)',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
              if (animal.breed != null)
                Padding(
                  padding: const EdgeInsets.only(top: 4, bottom: 4),
                  child: Text(
                      'Breed: ${animal.breed} (${(animal.speciesConfidence! * 100).toStringAsFixed(0)}%)'),
                ),
              if (animal.pose != null && animal.pose!.hasLandmarks) ...[
                const SizedBox(height: 8),
                Text(
                  'Body Pose (${animal.pose!.landmarks.length} keypoints)',
                  style: Theme.of(context).textTheme.titleSmall,
                ),
                ...animal.pose!.landmarks.map((lm) => Card(
                      margin: const EdgeInsets.only(bottom: 4),
                      child: ListTile(
                        dense: true,
                        leading: CircleAvatar(
                          radius: 14,
                          backgroundColor: Colors.orange,
                          child: Text(
                            lm.type.index.toString(),
                            style: const TextStyle(
                                fontSize: 9, color: Colors.white),
                          ),
                        ),
                        title: Text(
                          lm.type.name,
                          style: const TextStyle(fontWeight: FontWeight.w500),
                        ),
                        subtitle: Text(
                          'Position: (${lm.x.toStringAsFixed(1)}, ${lm.y.toStringAsFixed(1)})  conf: ${(lm.confidence * 100).toStringAsFixed(0)}%',
                        ),
                      ),
                    )),
              ],
              if (animal != _results.last) const Divider(height: 24),
            ],
          ],
        ),
      ),
    );
  }
}

class AnimalVisualizerWidget extends StatelessWidget {
  final Uint8List imageBytes;
  final int imageWidth;
  final int imageHeight;
  final List<Animal> results;

  const AnimalVisualizerWidget({
    super.key,
    required this.imageBytes,
    required this.imageWidth,
    required this.imageHeight,
    required this.results,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, constraints) {
      return Stack(
        children: [
          Image.memory(imageBytes, fit: BoxFit.contain),
          Positioned.fill(
            child: CustomPaint(
              painter: AnimalOverlayPainter(
                results: results,
                imageWidth: imageWidth,
                imageHeight: imageHeight,
              ),
            ),
          ),
        ],
      );
    });
  }
}

class AnimalOverlayPainter extends CustomPainter {
  final List<Animal> results;
  final int imageWidth;
  final int imageHeight;

  AnimalOverlayPainter({
    required this.results,
    required this.imageWidth,
    required this.imageHeight,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (results.isEmpty || imageWidth == 0 || imageHeight == 0) return;

    final double imageAspect = imageWidth / imageHeight;
    final double canvasAspect = size.width / size.height;
    double scaleX, scaleY;
    double offsetX = 0, offsetY = 0;

    if (canvasAspect > imageAspect) {
      scaleY = size.height / imageHeight;
      scaleX = scaleY;
      offsetX = (size.width - imageWidth * scaleX) / 2;
    } else {
      scaleX = size.width / imageWidth;
      scaleY = scaleX;
      offsetY = (size.height - imageHeight * scaleY) / 2;
    }

    for (final animal in results) {
      _drawBoundingBox(canvas, animal, scaleX, scaleY, offsetX, offsetY);
      _drawSpeciesLabel(canvas, animal, scaleX, scaleY, offsetX, offsetY);

      if (animal.pose != null && animal.pose!.hasLandmarks) {
        _drawBodySkeleton(canvas, animal, scaleX, scaleY, offsetX, offsetY);
        _drawBodyKeypoints(canvas, animal, scaleX, scaleY, offsetX, offsetY);
      }
    }
  }

  void _drawBoundingBox(Canvas canvas, Animal animal, double scaleX,
      double scaleY, double offsetX, double offsetY) {
    final Paint strokePaint = Paint()
      ..color = Colors.orange.withValues(alpha: 0.9)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final Paint fillPaint = Paint()
      ..color = Colors.orange.withValues(alpha: 0.08)
      ..style = PaintingStyle.fill;

    final double x1 = animal.boundingBox.left * scaleX + offsetX;
    final double y1 = animal.boundingBox.top * scaleY + offsetY;
    final double x2 = animal.boundingBox.right * scaleX + offsetX;
    final double y2 = animal.boundingBox.bottom * scaleY + offsetY;
    final Rect rect = Rect.fromLTRB(x1, y1, x2, y2);
    canvas.drawRect(rect, fillPaint);
    canvas.drawRect(rect, strokePaint);
  }

  void _drawSpeciesLabel(Canvas canvas, Animal animal, double scaleX,
      double scaleY, double offsetX, double offsetY) {
    if (animal.species == null) return;

    final double x1 = animal.boundingBox.left * scaleX + offsetX;
    final double y1 = animal.boundingBox.top * scaleY + offsetY;

    final String breedInfo = animal.breed != null &&
            animal.speciesConfidence != null
        ? ' (${animal.breed}, ${(animal.speciesConfidence! * 100).toStringAsFixed(0)}%)'
        : '';
    final String label = '${animal.species}$breedInfo';
    final TextPainter textPainter = TextPainter(
      text: TextSpan(
        text: label,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();

    final double padding = 4;
    final double labelY = y1 - textPainter.height - padding * 2;
    final Rect bgRect = Rect.fromLTWH(
      x1,
      labelY,
      textPainter.width + padding * 2,
      textPainter.height + padding * 2,
    );

    canvas.drawRect(
      bgRect,
      Paint()..color = Colors.orange.withValues(alpha: 0.85),
    );
    textPainter.paint(canvas, Offset(x1 + padding, labelY + padding));
  }

  void _drawBodySkeleton(Canvas canvas, Animal animal, double scaleX,
      double scaleY, double offsetX, double offsetY) {
    final Paint posePaint = Paint()
      ..color = Colors.red.withValues(alpha: 0.8)
      ..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round;

    for (final bone in animalPoseConnections) {
      final start = animal.pose!.getLandmark(bone[0]);
      final end = animal.pose!.getLandmark(bone[1]);
      if (start != null && end != null) {
        canvas.drawLine(
          Offset(start.x * scaleX + offsetX, start.y * scaleY + offsetY),
          Offset(end.x * scaleX + offsetX, end.y * scaleY + offsetY),
          posePaint,
        );
      }
    }
  }

  void _drawBodyKeypoints(Canvas canvas, Animal animal, double scaleX,
      double scaleY, double offsetX, double offsetY) {
    for (final lm in animal.pose!.landmarks) {
      final Offset center =
          Offset(lm.x * scaleX + offsetX, lm.y * scaleY + offsetY);
      canvas.drawCircle(center, 5, Paint()..color = Colors.red);
      canvas.drawCircle(center, 2, Paint()..color = Colors.white);
    }
  }

  @override
  bool shouldRepaint(AnimalOverlayPainter oldDelegate) => true;
}
