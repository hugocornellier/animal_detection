import 'dart:io';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

/// Downloads and caches models from GitHub Releases.
///
/// Models are cached in the application support directory under
/// `animal_detection/models/`. Subsequent calls return the cached bytes
/// without re-downloading.
class ModelDownloader {
  /// GitHub Releases base URL for downloadable models.
  static const String _releaseBaseUrl =
      'https://github.com/hugocornellier/dog_detection/releases/download/v0.0.1-models';

  /// File name of the HRNet pose model.
  static const String modelHrnet = 'superanimal_hrnet_w32_float16.tflite';

  /// Returns the local cache directory for downloaded models.
  static Future<Directory> _cacheDir() async {
    final appDir = await getApplicationSupportDirectory();
    final dir = Directory('${appDir.path}/animal_detection/models');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir;
  }

  /// Downloads a model file if not cached, returning the raw bytes.
  ///
  /// [onProgress] is called with (bytesReceived, totalBytes) during download.
  /// If totalBytes is unknown, -1 is passed.
  static Future<Uint8List> getModel(
    String fileName, {
    void Function(int received, int total)? onProgress,
  }) async {
    final dir = await _cacheDir();
    final file = File('${dir.path}/$fileName');

    if (await file.exists()) {
      return await file.readAsBytes();
    }

    final url = '$_releaseBaseUrl/$fileName';
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

  /// Downloads the HRNet pose model if not cached, returning the raw bytes.
  ///
  /// HRNet-w32 (~54.6MB) is the high-accuracy body pose model.
  /// [onProgress] is called with (bytesReceived, totalBytes) during download.
  static Future<Uint8List> getHrnetModel({
    void Function(int received, int total)? onProgress,
  }) async {
    return getModel(modelHrnet, onProgress: onProgress);
  }

  /// Returns true if the HRNet model is already cached locally.
  static Future<bool> isHrnetCached() async {
    final dir = await _cacheDir();
    final file = File('${dir.path}/$modelHrnet');
    return file.exists();
  }

  /// Deletes all cached models.
  static Future<void> clearCache() async {
    final dir = await _cacheDir();
    if (await dir.exists()) {
      await dir.delete(recursive: true);
    }
  }
}
