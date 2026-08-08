/// Internal composition surface for first-party detector packages.
///
/// End-user applications should import `animal_detection.dart` and use
/// `AnimalDetector`. This library exists so composite packages such as
/// cat_detection and dog_detection can reuse the pipeline inside their own
/// worker isolate without nesting another isolate.
library;

export 'src/animal_detector_core.dart' show AnimalDetectorCore;
