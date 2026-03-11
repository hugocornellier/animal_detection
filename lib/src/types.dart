export 'package:flutter_litert/flutter_litert.dart'
    show PerformanceMode, PerformanceConfig;

/// Body pose model variant for SuperAnimal keypoint extraction.
///
/// - [rtmpose]: RTMPose-S (11.6 MB, bundled). SimCC-based decoder, fast.
/// - [hrnet]: HRNet-w32 (54.6 MB, downloaded on demand). Heatmap-based,
///   most accurate.
enum AnimalPoseModel {
  /// RTMPose-S (11.6MB, bundled). SimCC-based, fast.
  rtmpose,

  /// HRNet-w32 (54.6MB, downloaded on demand). Heatmap-based, most accurate.
  hrnet,
}

/// SuperAnimal body keypoint types (indices 15–38 in the full SuperAnimal topology).
///
/// These 24 landmarks cover the spine, neck, tail, and all four limbs.
///
/// Example:
/// ```dart
/// final pose = animal.pose;
/// if (pose != null) {
///   final tail = pose.getLandmark(AnimalPoseLandmarkType.tailEnd);
///   if (tail != null) {
///     print('Tail tip at (${tail.x}, ${tail.y}) confidence=${tail.confidence}');
///   }
/// }
/// ```
enum AnimalPoseLandmarkType {
  /// Base of the neck (SuperAnimal index 15).
  neckBase,

  /// End of the neck (SuperAnimal index 16).
  neckEnd,

  /// Base of the throat (SuperAnimal index 17).
  throatBase,

  /// End of the throat (SuperAnimal index 18).
  throatEnd,

  /// Base of the back / withers (SuperAnimal index 19).
  backBase,

  /// End of the back (SuperAnimal index 20).
  backEnd,

  /// Middle of the back (SuperAnimal index 21).
  backMiddle,

  /// Base of the tail (SuperAnimal index 22).
  tailBase,

  /// Tip of the tail (SuperAnimal index 23).
  tailEnd,

  /// Front-left thigh (SuperAnimal index 24).
  frontLeftThigh,

  /// Front-left knee (SuperAnimal index 25).
  frontLeftKnee,

  /// Front-left paw (SuperAnimal index 26).
  frontLeftPaw,

  /// Front-right thigh (SuperAnimal index 27).
  frontRightThigh,

  /// Front-right knee (SuperAnimal index 28).
  frontRightKnee,

  /// Front-right paw (SuperAnimal index 29).
  frontRightPaw,

  /// Back-left paw (SuperAnimal index 30).
  backLeftPaw,

  /// Back-left thigh (SuperAnimal index 31).
  backLeftThigh,

  /// Back-right thigh (SuperAnimal index 32).
  backRightThigh,

  /// Back-left knee (SuperAnimal index 33).
  backLeftKnee,

  /// Back-right knee (SuperAnimal index 34).
  backRightKnee,

  /// Back-right paw (SuperAnimal index 35).
  backRightPaw,

  /// Bottom of the belly (SuperAnimal index 36).
  bellyBottom,

  /// Right side of the body middle (SuperAnimal index 37).
  bodyMiddleRight,

  /// Left side of the body middle (SuperAnimal index 38).
  bodyMiddleLeft,
}

/// 2D integer pixel coordinate.
class Point {
  /// X coordinate in pixels
  final int x;

  /// Y coordinate in pixels
  final int y;

  /// Creates a 2D pixel coordinate at position ([x], [y]).
  Point(this.x, this.y);
}

/// Axis-aligned bounding box in pixel coordinates.
///
/// Coordinates are in the original image space (not normalized).
class BoundingBox {
  /// Left edge x-coordinate in pixels
  final double left;

  /// Top edge y-coordinate in pixels
  final double top;

  /// Right edge x-coordinate in pixels
  final double right;

  /// Bottom edge y-coordinate in pixels
  final double bottom;

  /// Creates an axis-aligned bounding box with the specified edges.
  const BoundingBox({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });

  /// Serializes this bounding box to a map for cross-isolate transfer.
  Map<String, dynamic> toMap() => {
        'left': left,
        'top': top,
        'right': right,
        'bottom': bottom,
      };

  /// Deserializes a bounding box from a map.
  static BoundingBox fromMap(Map<String, dynamic> map) => BoundingBox(
        left: (map['left'] as num).toDouble(),
        top: (map['top'] as num).toDouble(),
        right: (map['right'] as num).toDouble(),
        bottom: (map['bottom'] as num).toDouble(),
      );
}

/// A single body pose keypoint with 2D coordinates and a confidence score.
///
/// Coordinates are in the original image space (pixels).
class AnimalPoseLandmark {
  /// The body keypoint type this represents
  final AnimalPoseLandmarkType type;

  /// X coordinate in pixels
  final double x;

  /// Y coordinate in pixels
  final double y;

  /// Model confidence for this keypoint (0.0 to 1.0)
  final double confidence;

  /// Creates a body pose landmark with 2D coordinates and confidence.
  AnimalPoseLandmark({
    required this.type,
    required this.x,
    required this.y,
    required this.confidence,
  });

  /// Serializes this pose landmark to a map for cross-isolate transfer.
  Map<String, dynamic> toMap() => {
        'type': type.name,
        'x': x,
        'y': y,
        'confidence': confidence,
      };

  /// Deserializes a pose landmark from a map.
  static AnimalPoseLandmark fromMap(Map<String, dynamic> map) =>
      AnimalPoseLandmark(
        type: AnimalPoseLandmarkType.values
            .firstWhere((e) => e.name == map['type']),
        x: (map['x'] as num).toDouble(),
        y: (map['y'] as num).toDouble(),
        confidence: (map['confidence'] as num).toDouble(),
      );
}

/// Full-body pose result for a single detected animal.
///
/// Contains up to 24 SuperAnimal body keypoints produced by the pose model.
class AnimalPose {
  /// List of body keypoints. May contain fewer than 24 entries if some
  /// keypoints were not detected with sufficient confidence.
  final List<AnimalPoseLandmark> landmarks;

  /// Creates an animal body pose with the given list of keypoints.
  const AnimalPose({required this.landmarks});

  /// Returns the landmark for [type], or null if not present.
  AnimalPoseLandmark? getLandmark(AnimalPoseLandmarkType type) {
    try {
      return landmarks.firstWhere((l) => l.type == type);
    } catch (_) {
      return null;
    }
  }

  /// Returns true if this pose has any landmarks.
  bool get hasLandmarks => landmarks.isNotEmpty;

  /// Serializes this pose to a map for cross-isolate transfer.
  Map<String, dynamic> toMap() => {
        'landmarks': landmarks.map((l) => l.toMap()).toList(),
      };

  /// Deserializes an animal pose from a map.
  static AnimalPose fromMap(Map<String, dynamic> map) => AnimalPose(
        landmarks: (map['landmarks'] as List)
            .map((l) => AnimalPoseLandmark.fromMap(l as Map<String, dynamic>))
            .toList(),
      );
}

/// Defines the standard skeleton connections between SuperAnimal body keypoints.
///
/// Each entry is a pair of [AnimalPoseLandmarkType] values representing the
/// start and end points of a bone segment.
const List<List<AnimalPoseLandmarkType>> animalPoseConnections = [
  // Neck/Throat
  [AnimalPoseLandmarkType.throatBase, AnimalPoseLandmarkType.throatEnd],
  // Spine
  [AnimalPoseLandmarkType.backBase, AnimalPoseLandmarkType.backMiddle],
  [AnimalPoseLandmarkType.backMiddle, AnimalPoseLandmarkType.backEnd],
  [AnimalPoseLandmarkType.backEnd, AnimalPoseLandmarkType.tailBase],
  [AnimalPoseLandmarkType.tailBase, AnimalPoseLandmarkType.tailEnd],
  // Front legs
  [AnimalPoseLandmarkType.frontLeftThigh, AnimalPoseLandmarkType.frontLeftKnee],
  [AnimalPoseLandmarkType.frontLeftKnee, AnimalPoseLandmarkType.frontLeftPaw],
  [
    AnimalPoseLandmarkType.frontRightThigh,
    AnimalPoseLandmarkType.frontRightKnee,
  ],
  [AnimalPoseLandmarkType.frontRightKnee, AnimalPoseLandmarkType.frontRightPaw],
  // Back legs
  [AnimalPoseLandmarkType.backLeftThigh, AnimalPoseLandmarkType.backLeftKnee],
  [AnimalPoseLandmarkType.backLeftKnee, AnimalPoseLandmarkType.backLeftPaw],
  [AnimalPoseLandmarkType.backRightThigh, AnimalPoseLandmarkType.backRightKnee],
  [AnimalPoseLandmarkType.backRightKnee, AnimalPoseLandmarkType.backRightPaw],
];

/// Top-level result for a single detected animal.
///
/// Produced by [AnimalDetector.detect()] and aggregates all pipeline stages:
/// SSD body detection, optional species classification, and optional body pose.
///
/// - [boundingBox]: Body bounding box in the original image
/// - [score]: SSD detector confidence (0.0 to 1.0)
/// - [species]: Predicted species label, or null if not run
/// - [breed]: Predicted breed label, or null if not run
/// - [speciesConfidence]: Classifier confidence, or null if not run
/// - [pose]: Body pose result, or null if not run
/// - [imageWidth] / [imageHeight]: Original image dimensions
class Animal {
  /// Body bounding box in pixel coordinates (original image space)
  final BoundingBox boundingBox;

  /// SSD detector confidence score (0.0 to 1.0)
  final double score;

  /// Predicted species label (e.g. "dog", "cat"), or null if classification was not run
  final String? species;

  /// Predicted breed label, or null if classification was not run
  final String? breed;

  /// Species classifier confidence (0.0 to 1.0), or null if not run
  final double? speciesConfidence;

  /// Body pose keypoints, or null if pose estimation was not run
  final AnimalPose? pose;

  /// Width of the original image in pixels
  final int imageWidth;

  /// Height of the original image in pixels
  final int imageHeight;

  /// Creates a top-level animal detection result.
  const Animal({
    required this.boundingBox,
    required this.score,
    this.species,
    this.breed,
    this.speciesConfidence,
    this.pose,
    required this.imageWidth,
    required this.imageHeight,
  });

  /// Serializes this result to a map for cross-isolate transfer.
  Map<String, dynamic> toMap() => {
        'boundingBox': boundingBox.toMap(),
        'score': score,
        'species': species,
        'breed': breed,
        'speciesConfidence': speciesConfidence,
        'pose': pose?.toMap(),
        'imageWidth': imageWidth,
        'imageHeight': imageHeight,
      };

  /// Deserializes an animal detection result from a map.
  static Animal fromMap(Map<String, dynamic> map) => Animal(
        boundingBox:
            BoundingBox.fromMap(map['boundingBox'] as Map<String, dynamic>),
        score: (map['score'] as num).toDouble(),
        species: map['species'] as String?,
        breed: map['breed'] as String?,
        speciesConfidence: (map['speciesConfidence'] as num?)?.toDouble(),
        pose: map['pose'] != null
            ? AnimalPose.fromMap(map['pose'] as Map<String, dynamic>)
            : null,
        imageWidth: map['imageWidth'] as int,
        imageHeight: map['imageHeight'] as int,
      );

  @override
  String toString() =>
      'Animal(score=${score.toStringAsFixed(3)}, species=$species, breed=$breed, pose=${pose != null})';
}

/// Metadata for mapping landmark coordinates from crop space back to original image space.
class CropMetadata {
  /// Left edge of the crop region in original image pixels.
  final double cx1;

  /// Top edge of the crop region in original image pixels.
  final double cy1;

  /// Width of the crop region in original image pixels.
  final double cropW;

  /// Height of the crop region in original image pixels.
  final double cropH;

  /// Creates crop metadata with origin and dimensions.
  const CropMetadata({
    required this.cx1,
    required this.cy1,
    required this.cropW,
    required this.cropH,
  });
}
