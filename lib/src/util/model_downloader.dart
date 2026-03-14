import 'dart:typed_data';
import 'species_model_downloader.dart';

/// Downloads and caches the HRNet body pose model from GitHub Releases.
class ModelDownloader {
  static const _downloader = SpeciesModelDownloader(
    releaseBaseUrl:
        'https://github.com/hugocornellier/animal_detection/releases/download/v0.0.1-models',
    cacheSubdir: 'animal_detection/models',
    model256Name: '',
    model320Name: '',
  );

  /// File name of the HRNet pose model.
  static const String modelHrnet = 'superanimal_hrnet_w32_256_float16.tflite';

  static const String _hrnetFileName =
      'superanimal_hrnet_w32_256_float16.tflite';

  /// Downloads the HRNet model if not cached, returning the raw bytes.
  static Future<Uint8List> getHrnetModel({
    void Function(int received, int total)? onProgress,
  }) =>
      _downloader.getModel(_hrnetFileName, onProgress: onProgress);

  /// Returns true if the HRNet model is already cached locally.
  static Future<bool> isHrnetCached() =>
      _downloader.isModelCached(_hrnetFileName);

  /// Deletes all cached models.
  static Future<void> clearCache() => _downloader.clearCache();
}
