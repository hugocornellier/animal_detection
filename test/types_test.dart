import 'package:flutter_test/flutter_test.dart';
import 'package:animal_detection/animal_detection.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  // ---------------------------------------------------------------------------
  // AnimalPoseModel enum
  // ---------------------------------------------------------------------------
  group('AnimalPoseModel enum', () {
    test('has exactly 2 values', () {
      expect(AnimalPoseModel.values.length, 2);
    });

    test('values are rtmpose and hrnet', () {
      expect(AnimalPoseModel.values.contains(AnimalPoseModel.rtmpose), true);
      expect(AnimalPoseModel.values.contains(AnimalPoseModel.hrnet), true);
    });

    test('rtmpose has index 0', () {
      expect(AnimalPoseModel.rtmpose.index, 0);
    });

    test('hrnet has index 1', () {
      expect(AnimalPoseModel.hrnet.index, 1);
    });

    test('name property works', () {
      expect(AnimalPoseModel.rtmpose.name, 'rtmpose');
      expect(AnimalPoseModel.hrnet.name, 'hrnet');
    });
  });

  // ---------------------------------------------------------------------------
  // AnimalPoseLandmarkType enum, 24 values, SuperAnimal topology
  // ---------------------------------------------------------------------------
  group('AnimalPoseLandmarkType enum', () {
    test('has exactly 24 values', () {
      expect(AnimalPoseLandmarkType.values.length, 24);
    });

    test('neck indices', () {
      expect(AnimalPoseLandmarkType.neckBase.index, 0);
      expect(AnimalPoseLandmarkType.neckEnd.index, 1);
    });

    test('throat indices', () {
      expect(AnimalPoseLandmarkType.throatBase.index, 2);
      expect(AnimalPoseLandmarkType.throatEnd.index, 3);
    });

    test('back indices', () {
      expect(AnimalPoseLandmarkType.backBase.index, 4);
      expect(AnimalPoseLandmarkType.backEnd.index, 5);
      expect(AnimalPoseLandmarkType.backMiddle.index, 6);
    });

    test('tail indices', () {
      expect(AnimalPoseLandmarkType.tailBase.index, 7);
      expect(AnimalPoseLandmarkType.tailEnd.index, 8);
    });

    test('front leg indices', () {
      expect(AnimalPoseLandmarkType.frontLeftThigh.index, 9);
      expect(AnimalPoseLandmarkType.frontLeftKnee.index, 10);
      expect(AnimalPoseLandmarkType.frontLeftPaw.index, 11);
      expect(AnimalPoseLandmarkType.frontRightThigh.index, 12);
      expect(AnimalPoseLandmarkType.frontRightKnee.index, 13);
      expect(AnimalPoseLandmarkType.frontRightPaw.index, 14);
    });

    test('back leg indices', () {
      expect(AnimalPoseLandmarkType.backLeftPaw.index, 15);
      expect(AnimalPoseLandmarkType.backLeftThigh.index, 16);
      expect(AnimalPoseLandmarkType.backRightThigh.index, 17);
      expect(AnimalPoseLandmarkType.backLeftKnee.index, 18);
      expect(AnimalPoseLandmarkType.backRightKnee.index, 19);
      expect(AnimalPoseLandmarkType.backRightPaw.index, 20);
    });

    test('body midline indices', () {
      expect(AnimalPoseLandmarkType.bellyBottom.index, 21);
      expect(AnimalPoseLandmarkType.bodyMiddleRight.index, 22);
      expect(AnimalPoseLandmarkType.bodyMiddleLeft.index, 23);
    });

    test('verify specific landmark names by index', () {
      expect(AnimalPoseLandmarkType.values[0].name, 'neckBase');
      expect(AnimalPoseLandmarkType.values[1].name, 'neckEnd');
      expect(AnimalPoseLandmarkType.values[7].name, 'tailBase');
      expect(AnimalPoseLandmarkType.values[8].name, 'tailEnd');
      expect(AnimalPoseLandmarkType.values[11].name, 'frontLeftPaw');
      expect(AnimalPoseLandmarkType.values[14].name, 'frontRightPaw');
      expect(AnimalPoseLandmarkType.values[21].name, 'bellyBottom');
      expect(AnimalPoseLandmarkType.values[23].name, 'bodyMiddleLeft');
    });
  });

  // ---------------------------------------------------------------------------
  // AnimalPoseLandmark class
  // ---------------------------------------------------------------------------
  group('AnimalPoseLandmark', () {
    test('constructor stores all fields correctly', () {
      final landmark = AnimalPoseLandmark(
        type: AnimalPoseLandmarkType.tailEnd,
        x: 120.5,
        y: 340.7,
        confidence: 0.92,
      );
      expect(landmark.type, AnimalPoseLandmarkType.tailEnd);
      expect(landmark.x, 120.5);
      expect(landmark.y, 340.7);
      expect(landmark.confidence, 0.92);
    });

    test('toMap produces correct map with type, x, y, confidence keys', () {
      final landmark = AnimalPoseLandmark(
        type: AnimalPoseLandmarkType.neckBase,
        x: 10.0,
        y: 20.0,
        confidence: 0.75,
      );
      final map = landmark.toMap();
      expect(map['type'], 'neckBase');
      expect(map['x'], 10.0);
      expect(map['y'], 20.0);
      expect(map['confidence'], 0.75);
      expect(map.containsKey('type'), true);
      expect(map.containsKey('x'), true);
      expect(map.containsKey('y'), true);
      expect(map.containsKey('confidence'), true);
    });

    test('fromMap factory reconstructs correctly', () {
      final map = {
        'type': 'backLeftPaw',
        'x': 55.0,
        'y': 66.0,
        'confidence': 0.88,
      };
      final landmark = AnimalPoseLandmark.fromMap(map);
      expect(landmark.type, AnimalPoseLandmarkType.backLeftPaw);
      expect(landmark.x, 55.0);
      expect(landmark.y, 66.0);
      expect(landmark.confidence, 0.88);
    });

    test('fromMap handles integer coordinates and confidence', () {
      final map = {'type': 'tailBase', 'x': 100, 'y': 200, 'confidence': 1};
      final landmark = AnimalPoseLandmark.fromMap(map);
      expect(landmark.type, AnimalPoseLandmarkType.tailBase);
      expect(landmark.x, 100.0);
      expect(landmark.y, 200.0);
      expect(landmark.confidence, 1.0);
    });

    test('toMap/fromMap round-trip preserves all fields', () {
      final original = AnimalPoseLandmark(
        type: AnimalPoseLandmarkType.frontRightKnee,
        x: 123.45,
        y: 678.9,
        confidence: 0.613,
      );
      final restored = AnimalPoseLandmark.fromMap(original.toMap());
      expect(restored.type, AnimalPoseLandmarkType.frontRightKnee);
      expect(restored.x, 123.45);
      expect(restored.y, 678.9);
      expect(restored.confidence, 0.613);
    });

    test('round-trip all landmark types', () {
      for (final type in AnimalPoseLandmarkType.values) {
        final original =
            AnimalPoseLandmark(type: type, x: 50.0, y: 50.0, confidence: 0.5);
        final restored = AnimalPoseLandmark.fromMap(original.toMap());
        expect(restored.type, type);
      }
    });

    test('boundary values: zero coordinates and confidence', () {
      final landmark = AnimalPoseLandmark(
        type: AnimalPoseLandmarkType.bellyBottom,
        x: 0.0,
        y: 0.0,
        confidence: 0.0,
      );
      expect(landmark.x, 0.0);
      expect(landmark.y, 0.0);
      expect(landmark.confidence, 0.0);
      final restored = AnimalPoseLandmark.fromMap(landmark.toMap());
      expect(restored.x, 0.0);
      expect(restored.y, 0.0);
      expect(restored.confidence, 0.0);
    });

    test('boundary values: max confidence 1.0', () {
      final landmark = AnimalPoseLandmark(
        type: AnimalPoseLandmarkType.bodyMiddleLeft,
        x: 640.0,
        y: 480.0,
        confidence: 1.0,
      );
      expect(landmark.confidence, 1.0);
      final restored = AnimalPoseLandmark.fromMap(landmark.toMap());
      expect(restored.confidence, 1.0);
    });

    test('large coordinate values round-trip', () {
      final landmark = AnimalPoseLandmark(
        type: AnimalPoseLandmarkType.backRightPaw,
        x: 9999.99,
        y: 8888.88,
        confidence: 0.999,
      );
      final restored = AnimalPoseLandmark.fromMap(landmark.toMap());
      expect(restored.x, 9999.99);
      expect(restored.y, 8888.88);
    });
  });

  // ---------------------------------------------------------------------------
  // AnimalPose class
  // ---------------------------------------------------------------------------
  group('AnimalPose', () {
    AnimalPoseLandmark makeLandmark(
      AnimalPoseLandmarkType type, {
      double x = 0,
      double y = 0,
      double confidence = 0.9,
    }) {
      return AnimalPoseLandmark(type: type, x: x, y: y, confidence: confidence);
    }

    test('constructor stores landmarks list', () {
      final landmarks = [
        makeLandmark(AnimalPoseLandmarkType.neckBase, x: 10.0, y: 20.0),
        makeLandmark(AnimalPoseLandmarkType.tailEnd, x: 100.0, y: 200.0),
      ];
      final pose = AnimalPose(landmarks: landmarks);
      expect(pose.landmarks.length, 2);
    });

    test('hasLandmarks is true when landmarks are present', () {
      final pose = AnimalPose(
        landmarks: [
          makeLandmark(AnimalPoseLandmarkType.neckBase),
        ],
      );
      expect(pose.hasLandmarks, true);
    });

    test('hasLandmarks is false for empty landmarks list', () {
      final pose = AnimalPose(landmarks: []);
      expect(pose.hasLandmarks, false);
    });

    test('getLandmark returns correct landmark when present', () {
      final pose = AnimalPose(
        landmarks: [
          makeLandmark(AnimalPoseLandmarkType.tailEnd, x: 55.0, y: 66.0),
          makeLandmark(AnimalPoseLandmarkType.neckBase, x: 11.0, y: 22.0),
        ],
      );
      final found = pose.getLandmark(AnimalPoseLandmarkType.tailEnd);
      expect(found, isNotNull);
      expect(found!.x, 55.0);
      expect(found.y, 66.0);
    });

    test('getLandmark returns null when type not present', () {
      final pose = AnimalPose(
        landmarks: [
          makeLandmark(AnimalPoseLandmarkType.neckBase),
        ],
      );
      final notFound = pose.getLandmark(AnimalPoseLandmarkType.tailEnd);
      expect(notFound, isNull);
    });

    test('getLandmark returns null on empty pose', () {
      final pose = AnimalPose(landmarks: []);
      final notFound = pose.getLandmark(AnimalPoseLandmarkType.backLeftKnee);
      expect(notFound, isNull);
    });

    test('toMap produces map with landmarks key', () {
      final pose = AnimalPose(
        landmarks: [
          makeLandmark(AnimalPoseLandmarkType.backBase, x: 5.0, y: 10.0),
        ],
      );
      final map = pose.toMap();
      expect(map.containsKey('landmarks'), true);
      expect(map['landmarks'], isList);
      expect((map['landmarks'] as List).length, 1);
    });

    test('fromMap factory reconstructs pose correctly', () {
      final map = {
        'landmarks': [
          {'type': 'neckBase', 'x': 10.0, 'y': 20.0, 'confidence': 0.8},
          {'type': 'tailEnd', 'x': 100.0, 'y': 200.0, 'confidence': 0.95},
        ],
      };
      final pose = AnimalPose.fromMap(map);
      expect(pose.landmarks.length, 2);
      expect(pose.landmarks[0].type, AnimalPoseLandmarkType.neckBase);
      expect(pose.landmarks[1].type, AnimalPoseLandmarkType.tailEnd);
    });

    test('toMap/fromMap round-trip with multiple landmarks', () {
      final original = AnimalPose(
        landmarks: [
          makeLandmark(AnimalPoseLandmarkType.frontLeftPaw,
              x: 30.0, y: 40.0, confidence: 0.7),
          makeLandmark(AnimalPoseLandmarkType.backRightPaw,
              x: 80.0, y: 90.0, confidence: 0.85),
          makeLandmark(AnimalPoseLandmarkType.bellyBottom,
              x: 50.0, y: 60.0, confidence: 0.6),
        ],
      );
      final restored = AnimalPose.fromMap(original.toMap());
      expect(restored.landmarks.length, 3);
      expect(restored.landmarks[0].type, AnimalPoseLandmarkType.frontLeftPaw);
      expect(restored.landmarks[0].x, 30.0);
      expect(restored.landmarks[1].type, AnimalPoseLandmarkType.backRightPaw);
      expect(restored.landmarks[2].type, AnimalPoseLandmarkType.bellyBottom);
    });

    test('toMap/fromMap round-trip with empty landmarks', () {
      final original = AnimalPose(landmarks: []);
      final restored = AnimalPose.fromMap(original.toMap());
      expect(restored.landmarks.isEmpty, true);
      expect(restored.hasLandmarks, false);
    });

    test('round-trip preserves all 24 landmark types', () {
      final landmarks = AnimalPoseLandmarkType.values
          .map((t) => makeLandmark(t, x: t.index * 1.0, y: t.index * 2.0))
          .toList();
      final original = AnimalPose(landmarks: landmarks);
      final restored = AnimalPose.fromMap(original.toMap());
      expect(restored.landmarks.length, 24);
      for (int i = 0; i < 24; i++) {
        expect(restored.landmarks[i].type, AnimalPoseLandmarkType.values[i]);
        expect(restored.landmarks[i].x, i * 1.0);
        expect(restored.landmarks[i].y, i * 2.0);
      }
    });
  });

  // ---------------------------------------------------------------------------
  // animalPoseConnections constant
  // ---------------------------------------------------------------------------
  group('animalPoseConnections constant', () {
    test('has exactly 13 connections', () {
      expect(animalPoseConnections.length, 13);
    });

    test('each connection is a pair (length 2)', () {
      for (final connection in animalPoseConnections) {
        expect(connection.length, 2);
      }
    });

    test('all connection endpoints are valid AnimalPoseLandmarkType values',
        () {
      for (final connection in animalPoseConnections) {
        expect(AnimalPoseLandmarkType.values.contains(connection[0]), true);
        expect(AnimalPoseLandmarkType.values.contains(connection[1]), true);
      }
    });

    test('throat connection is present', () {
      expect(
        animalPoseConnections.any(
          (c) =>
              c[0] == AnimalPoseLandmarkType.throatBase &&
              c[1] == AnimalPoseLandmarkType.throatEnd,
        ),
        true,
      );
    });

    test('spine connections form a chain', () {
      expect(
        animalPoseConnections.any(
          (c) =>
              c[0] == AnimalPoseLandmarkType.backBase &&
              c[1] == AnimalPoseLandmarkType.backMiddle,
        ),
        true,
      );
      expect(
        animalPoseConnections.any(
          (c) =>
              c[0] == AnimalPoseLandmarkType.backMiddle &&
              c[1] == AnimalPoseLandmarkType.backEnd,
        ),
        true,
      );
      expect(
        animalPoseConnections.any(
          (c) =>
              c[0] == AnimalPoseLandmarkType.backEnd &&
              c[1] == AnimalPoseLandmarkType.tailBase,
        ),
        true,
      );
      expect(
        animalPoseConnections.any(
          (c) =>
              c[0] == AnimalPoseLandmarkType.tailBase &&
              c[1] == AnimalPoseLandmarkType.tailEnd,
        ),
        true,
      );
    });

    test('front left leg connections are present', () {
      expect(
        animalPoseConnections.any(
          (c) =>
              c[0] == AnimalPoseLandmarkType.frontLeftThigh &&
              c[1] == AnimalPoseLandmarkType.frontLeftKnee,
        ),
        true,
      );
      expect(
        animalPoseConnections.any(
          (c) =>
              c[0] == AnimalPoseLandmarkType.frontLeftKnee &&
              c[1] == AnimalPoseLandmarkType.frontLeftPaw,
        ),
        true,
      );
    });

    test('front right leg connections are present', () {
      expect(
        animalPoseConnections.any(
          (c) =>
              c[0] == AnimalPoseLandmarkType.frontRightThigh &&
              c[1] == AnimalPoseLandmarkType.frontRightKnee,
        ),
        true,
      );
      expect(
        animalPoseConnections.any(
          (c) =>
              c[0] == AnimalPoseLandmarkType.frontRightKnee &&
              c[1] == AnimalPoseLandmarkType.frontRightPaw,
        ),
        true,
      );
    });

    test('back left leg connections are present', () {
      expect(
        animalPoseConnections.any(
          (c) =>
              c[0] == AnimalPoseLandmarkType.backLeftThigh &&
              c[1] == AnimalPoseLandmarkType.backLeftKnee,
        ),
        true,
      );
      expect(
        animalPoseConnections.any(
          (c) =>
              c[0] == AnimalPoseLandmarkType.backLeftKnee &&
              c[1] == AnimalPoseLandmarkType.backLeftPaw,
        ),
        true,
      );
    });

    test('back right leg connections are present', () {
      expect(
        animalPoseConnections.any(
          (c) =>
              c[0] == AnimalPoseLandmarkType.backRightThigh &&
              c[1] == AnimalPoseLandmarkType.backRightKnee,
        ),
        true,
      );
      expect(
        animalPoseConnections.any(
          (c) =>
              c[0] == AnimalPoseLandmarkType.backRightKnee &&
              c[1] == AnimalPoseLandmarkType.backRightPaw,
        ),
        true,
      );
    });

    test('no connection has identical start and end', () {
      for (final connection in animalPoseConnections) {
        expect(connection[0] == connection[1], false);
      }
    });
  });

  // ---------------------------------------------------------------------------
  // BoundingBox class
  // ---------------------------------------------------------------------------
  group('BoundingBox', () {
    test('ltrb constructor stores left, top, right, bottom', () {
      final bbox = BoundingBox.ltrb(10.5, 20.3, 100.7, 200.1);
      expect(bbox.left, 10.5);
      expect(bbox.top, 20.3);
      expect(bbox.right, 100.7);
      expect(bbox.bottom, 200.1);
    });

    test('width and height computed correctly', () {
      final bbox = BoundingBox.ltrb(10.0, 20.0, 110.0, 220.0);
      expect(bbox.width, closeTo(100.0, 0.0001));
      expect(bbox.height, closeTo(200.0, 0.0001));
    });

    test('toMap/fromMap round-trip', () {
      final original = BoundingBox.ltrb(10.5, 20.3, 100.7, 200.1);
      final restored = BoundingBox.fromMap(original.toMap());
      expect(restored.left, 10.5);
      expect(restored.top, 20.3);
      expect(restored.right, 100.7);
      expect(restored.bottom, 200.1);
    });

    test('zero-size box is preserved in round-trip', () {
      final bbox = BoundingBox.ltrb(50.0, 50.0, 50.0, 50.0);
      final restored = BoundingBox.fromMap(bbox.toMap());
      expect(restored.left, restored.right);
      expect(restored.top, restored.bottom);
    });

    test('negative coordinates are stored as-is', () {
      final bbox = BoundingBox.ltrb(-50.0, -30.0, -10.0, -5.0);
      expect(bbox.left, -50.0);
      expect(bbox.top, -30.0);
      expect(bbox.right, -10.0);
      expect(bbox.bottom, -5.0);
    });

    test('negative coordinates round-trip via toMap/fromMap', () {
      final original = BoundingBox.ltrb(-100.0, -80.0, -20.0, -10.0);
      final restored = BoundingBox.fromMap(original.toMap());
      expect(restored.left, -100.0);
      expect(restored.top, -80.0);
      expect(restored.right, -20.0);
      expect(restored.bottom, -10.0);
    });

    test('large coordinate values round-trip', () {
      final original = BoundingBox.ltrb(0.0, 0.0, 4096.0, 3072.0);
      final restored = BoundingBox.fromMap(original.toMap());
      expect(restored.left, 0.0);
      expect(restored.top, 0.0);
      expect(restored.right, 4096.0);
      expect(restored.bottom, 3072.0);
    });
  });

  // ---------------------------------------------------------------------------
  // Animal class
  // ---------------------------------------------------------------------------
  group('Animal', () {
    BoundingBox makeBox() => BoundingBox.ltrb(10.0, 20.0, 200.0, 300.0);

    AnimalPose makePose() => AnimalPose(
          landmarks: [
            AnimalPoseLandmark(
              type: AnimalPoseLandmarkType.neckBase,
              x: 50.0,
              y: 60.0,
              confidence: 0.9,
            ),
          ],
        );

    test('constructor with all fields stores them correctly', () {
      final pose = makePose();
      final animal = Animal(
        boundingBox: makeBox(),
        score: 0.95,
        species: 'dog',
        breed: 'labrador',
        speciesConfidence: 0.87,
        pose: pose,
        imageWidth: 640,
        imageHeight: 480,
      );
      expect(animal.score, 0.95);
      expect(animal.species, 'dog');
      expect(animal.breed, 'labrador');
      expect(animal.speciesConfidence, 0.87);
      expect(animal.pose, isNotNull);
      expect(animal.imageWidth, 640);
      expect(animal.imageHeight, 480);
      expect(animal.boundingBox.left, 10.0);
      expect(animal.boundingBox.top, 20.0);
      expect(animal.boundingBox.right, 200.0);
      expect(animal.boundingBox.bottom, 300.0);
    });

    test('constructor with optional null fields', () {
      final animal = Animal(
        boundingBox: makeBox(),
        score: 0.80,
        imageWidth: 1280,
        imageHeight: 720,
      );
      expect(animal.species, isNull);
      expect(animal.breed, isNull);
      expect(animal.speciesConfidence, isNull);
      expect(animal.pose, isNull);
    });

    test('toMap produces correct keys', () {
      final animal = Animal(
        boundingBox: makeBox(),
        score: 0.75,
        species: 'cat',
        imageWidth: 640,
        imageHeight: 480,
      );
      final map = animal.toMap();
      expect(map.containsKey('boundingBox'), true);
      expect(map.containsKey('score'), true);
      expect(map.containsKey('species'), true);
      expect(map.containsKey('breed'), true);
      expect(map.containsKey('speciesConfidence'), true);
      expect(map.containsKey('pose'), true);
      expect(map.containsKey('imageWidth'), true);
      expect(map.containsKey('imageHeight'), true);
    });

    test('toMap encodes boundingBox as nested map with left/top/right/bottom',
        () {
      final animal = Animal(
        boundingBox: BoundingBox.ltrb(1.0, 2.0, 3.0, 4.0),
        score: 0.5,
        imageWidth: 100,
        imageHeight: 100,
      );
      final map = animal.toMap();
      final bbMap = map['boundingBox'] as Map<String, dynamic>;
      expect(bbMap['left'], 1.0);
      expect(bbMap['top'], 2.0);
      expect(bbMap['right'], 3.0);
      expect(bbMap['bottom'], 4.0);
    });

    test('toMap/fromMap round-trip with all fields', () {
      final original = Animal(
        boundingBox: BoundingBox.ltrb(5.0, 10.0, 300.0, 400.0),
        score: 0.93,
        species: 'cat',
        breed: 'siamese',
        speciesConfidence: 0.78,
        pose: makePose(),
        imageWidth: 1920,
        imageHeight: 1080,
      );
      final restored = Animal.fromMap(original.toMap());
      expect(restored.score, 0.93);
      expect(restored.species, 'cat');
      expect(restored.breed, 'siamese');
      expect(restored.speciesConfidence, 0.78);
      expect(restored.pose, isNotNull);
      expect(restored.imageWidth, 1920);
      expect(restored.imageHeight, 1080);
      expect(restored.boundingBox.left, 5.0);
      expect(restored.boundingBox.top, 10.0);
      expect(restored.boundingBox.right, 300.0);
      expect(restored.boundingBox.bottom, 400.0);
    });

    test('toMap/fromMap round-trip with null optional fields', () {
      final original = Animal(
        boundingBox: BoundingBox.ltrb(0.0, 0.0, 100.0, 100.0),
        score: 0.60,
        imageWidth: 640,
        imageHeight: 480,
      );
      final restored = Animal.fromMap(original.toMap());
      expect(restored.species, isNull);
      expect(restored.breed, isNull);
      expect(restored.speciesConfidence, isNull);
      expect(restored.pose, isNull);
      expect(restored.score, 0.60);
    });

    test('fromMap reconstructs pose landmarks correctly', () {
      final original = Animal(
        boundingBox: makeBox(),
        score: 0.88,
        pose: AnimalPose(
          landmarks: [
            AnimalPoseLandmark(
              type: AnimalPoseLandmarkType.backLeftPaw,
              x: 77.0,
              y: 88.0,
              confidence: 0.95,
            ),
          ],
        ),
        imageWidth: 800,
        imageHeight: 600,
      );
      final restored = Animal.fromMap(original.toMap());
      expect(restored.pose!.landmarks.length, 1);
      expect(
          restored.pose!.landmarks[0].type, AnimalPoseLandmarkType.backLeftPaw);
      expect(restored.pose!.landmarks[0].x, 77.0);
      expect(restored.pose!.landmarks[0].y, 88.0);
    });

    test('toString contains score, species, breed, and pose flag', () {
      final animal = Animal(
        boundingBox: makeBox(),
        score: 0.913,
        species: 'dog',
        breed: 'poodle',
        pose: makePose(),
        imageWidth: 640,
        imageHeight: 480,
      );
      final str = animal.toString();
      expect(str.contains('0.913'), true);
      expect(str.contains('dog'), true);
      expect(str.contains('poodle'), true);
      expect(str.contains('true'), true);
    });

    test('toString with null fields shows null for species and breed', () {
      final animal = Animal(
        boundingBox: makeBox(),
        score: 0.500,
        imageWidth: 640,
        imageHeight: 480,
      );
      final str = animal.toString();
      expect(str.contains('0.500'), true);
      expect(str.contains('null'), true);
      expect(str.contains('false'), true);
    });

    test('fromMap handles integer score and dimensions', () {
      final map = {
        'boundingBox': {'left': 0, 'top': 0, 'right': 100, 'bottom': 100},
        'score': 1,
        'species': null,
        'breed': null,
        'speciesConfidence': null,
        'pose': null,
        'imageWidth': 640,
        'imageHeight': 480,
      };
      final animal = Animal.fromMap(map);
      expect(animal.score, 1.0);
      expect(animal.imageWidth, 640);
      expect(animal.imageHeight, 480);
    });
  });

  // ---------------------------------------------------------------------------
  // CropMetadata class
  // ---------------------------------------------------------------------------
  group('CropMetadata', () {
    test('constructor stores all fields correctly', () {
      const meta =
          CropMetadata(cx1: 10.0, cy1: 20.0, cropW: 150.0, cropH: 200.0);
      expect(meta.cx1, 10.0);
      expect(meta.cy1, 20.0);
      expect(meta.cropW, 150.0);
      expect(meta.cropH, 200.0);
    });

    test('zero values are stored correctly', () {
      const meta = CropMetadata(cx1: 0.0, cy1: 0.0, cropW: 0.0, cropH: 0.0);
      expect(meta.cx1, 0.0);
      expect(meta.cy1, 0.0);
      expect(meta.cropW, 0.0);
      expect(meta.cropH, 0.0);
    });

    test('fractional values are stored without loss', () {
      const meta =
          CropMetadata(cx1: 1.5, cy1: 2.75, cropW: 300.25, cropH: 400.125);
      expect(meta.cx1, 1.5);
      expect(meta.cy1, 2.75);
      expect(meta.cropW, 300.25);
      expect(meta.cropH, 400.125);
    });

    test('large values are stored correctly', () {
      const meta =
          CropMetadata(cx1: 3840.0, cy1: 2160.0, cropW: 1920.0, cropH: 1080.0);
      expect(meta.cx1, 3840.0);
      expect(meta.cy1, 2160.0);
      expect(meta.cropW, 1920.0);
      expect(meta.cropH, 1080.0);
    });
  });
}
