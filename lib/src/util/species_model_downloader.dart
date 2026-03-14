import 'dart:io';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

/// Downloads and caches species-specific ensemble landmark models from GitHub Releases.
///
/// Each species package (cat_detection, dog_detection) creates an instance
/// with its specific release URL, cache directory, and model file names.
class SpeciesModelDownloader {
  /// GitHub Releases base URL for downloadable models.
  final String releaseBaseUrl;

  /// Subdirectory under the app support directory for caching.
  final String cacheSubdir;

  /// File name of the 256px ensemble model.
  final String model256Name;

  /// File name of the 320px ensemble model.
  final String model320Name;

  /// Creates a downloader with the given configuration.
  const SpeciesModelDownloader({
    required this.releaseBaseUrl,
    required this.cacheSubdir,
    required this.model256Name,
    required this.model320Name,
  });

  Future<Directory> _cacheDir() async {
    final appDir = await getApplicationSupportDirectory();
    final dir = Directory('${appDir.path}/$cacheSubdir');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir;
  }

  /// Downloads a model file if not cached, returning the raw bytes.
  Future<Uint8List> getModel(
    String fileName, {
    void Function(int received, int total)? onProgress,
  }) async {
    final dir = await _cacheDir();
    final file = File('${dir.path}/$fileName');

    if (await file.exists()) {
      return await file.readAsBytes();
    }

    final url = '$releaseBaseUrl/$fileName';
    final request = http.Request('GET', Uri.parse(url));
    final response = await http.Client().send(request);

    if (response.statusCode != 200) {
      throw HttpException(
        'Failed to download $fileName: HTTP ${response.statusCode}',
        uri: Uri.parse(url),
      );
    }

    final totalBytes = response.contentLength ?? -1;
    final chunks = <List<int>>[];
    int received = 0;

    await for (final chunk in response.stream) {
      chunks.add(chunk);
      received += chunk.length;
      onProgress?.call(received, totalBytes);
    }

    final bytes = Uint8List(received);
    int offset = 0;
    for (final chunk in chunks) {
      bytes.setRange(offset, offset + chunk.length, chunk);
      offset += chunk.length;
    }

    await file.writeAsBytes(bytes, flush: true);
    return bytes;
  }

  /// Downloads both ensemble models (256px and 320px) in parallel.
  Future<(Uint8List, Uint8List)> getEnsembleModels({
    void Function(String model, int received, int total)? onProgress,
  }) async {
    final results = await Future.wait([
      getModel(
        model256Name,
        onProgress: onProgress != null
            ? (r, t) => onProgress(model256Name, r, t)
            : null,
      ),
      getModel(
        model320Name,
        onProgress: onProgress != null
            ? (r, t) => onProgress(model320Name, r, t)
            : null,
      ),
    ]);
    return (results[0], results[1]);
  }

  /// Returns true if both ensemble models are already cached locally.
  Future<bool> isEnsembleCached() async {
    final dir = await _cacheDir();
    final f256 = File('${dir.path}/$model256Name');
    final f320 = File('${dir.path}/$model320Name');
    return await f256.exists() && await f320.exists();
  }

  /// Returns true if a specific model file is cached locally.
  Future<bool> isModelCached(String fileName) async {
    final dir = await _cacheDir();
    return await File('${dir.path}/$fileName').exists();
  }

  /// Deletes all cached models.
  Future<void> clearCache() async {
    final dir = await _cacheDir();
    if (await dir.exists()) {
      await dir.delete(recursive: true);
    }
  }
}
