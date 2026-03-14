import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';
import '../types.dart';

/// Utility functions for image preprocessing using OpenCV.
class ImageUtils {
  /// Per-channel mean values (R, G, B) used for ImageNet normalization.
  static const List<double> imagenetMean = [0.485, 0.456, 0.406];

  /// Per-channel standard deviation values (R, G, B) used for ImageNet normalization.
  static const List<double> imagenetStd = [0.229, 0.224, 0.225];

  /// Letterbox-resize [src] to [targetSize]x[targetSize], preserving aspect ratio.
  /// Returns (resized Mat, LetterboxParams).
  static (cv.Mat, LetterboxParams) letterboxResize(cv.Mat src, int targetSize) {
    final params = computeLetterboxParams(
      srcWidth: src.cols,
      srcHeight: src.rows,
      targetWidth: targetSize,
      targetHeight: targetSize,
    );
    final resized = cv.resize(src, (params.newWidth, params.newHeight));
    final padded = cv.copyMakeBorder(
      resized,
      params.padTop,
      params.padBottom,
      params.padLeft,
      params.padRight,
      cv.BORDER_CONSTANT,
      value: cv.Scalar.black,
    );
    resized.dispose();
    return (padded, params);
  }

  /// Crop [bbox] from [src] with [margin] fraction on each side, then resize to [targetSize]x[targetSize].
  /// Returns (cropped+resized Mat, crop metadata for coordinate mapping).
  static (cv.Mat, CropMetadata) cropAndResize(
    cv.Mat src,
    BoundingBox bbox,
    double margin,
    int targetSize,
  ) {
    final bw = bbox.right - bbox.left;
    final bh = bbox.bottom - bbox.top;
    final cx1 = (bbox.left - bw * margin).clamp(0.0, src.cols.toDouble());
    final cy1 = (bbox.top - bh * margin).clamp(0.0, src.rows.toDouble());
    final cx2 = (bbox.right + bw * margin).clamp(0.0, src.cols.toDouble());
    final cy2 = (bbox.bottom + bh * margin).clamp(0.0, src.rows.toDouble());
    final cropW = cx2 - cx1;
    final cropH = cy2 - cy1;

    final cropped = src.region(
      cv.Rect(cx1.toInt(), cy1.toInt(), cropW.toInt(), cropH.toInt()),
    );
    final resized = cv.resize(cropped, (targetSize, targetSize));
    cropped.dispose();

    return (
      resized,
      CropMetadata(cx1: cx1, cy1: cy1, cropW: cropW, cropH: cropH),
    );
  }

  /// Convert BGR Mat to RGB Float32List normalized to [0, 1].
  static Float32List matToFloat32(cv.Mat mat) {
    return bgrBytesToRgbFloat32(
      bytes: mat.data,
      totalPixels: mat.rows * mat.cols,
    );
  }

  /// Convert BGR Mat to RGB Float32List with ImageNet normalization.
  ///
  /// Each channel is normalized: (pixel/255 - mean) / std
  /// where mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
  static Float32List matToFloat32ImageNet(cv.Mat mat) {
    final int totalPixels = mat.rows * mat.cols;
    final Uint8List bytes = mat.data;
    final result = Float32List(totalPixels * 3);

    for (int i = 0; i < totalPixels; i++) {
      final int bgr = i * 3;
      final int rgb = i * 3;
      // BGR -> RGB, normalize with ImageNet stats
      final double r = bytes[bgr + 2] / 255.0;
      final double g = bytes[bgr + 1] / 255.0;
      final double b = bytes[bgr + 0] / 255.0;
      result[rgb + 0] = (r - imagenetMean[0]) / imagenetStd[0];
      result[rgb + 1] = (g - imagenetMean[1]) / imagenetStd[1];
      result[rgb + 2] = (b - imagenetMean[2]) / imagenetStd[2];
    }

    return result;
  }

  /// Letterbox-resize and apply ImageNet normalization.
  static (Float32List, LetterboxParams) letterboxAndNormalizeImageNet(
    cv.Mat src,
    int targetSize,
  ) {
    final (padded, params) = letterboxResize(src, targetSize);
    final normalized = matToFloat32ImageNet(padded);
    padded.dispose();
    return (normalized, params);
  }

  /// Expands a bounding box by [margin] fraction on each side, clamped to image bounds.
  ///
  /// Returns (x1, y1, x2, y2) as integers.
  static (int, int, int, int) expandBox(
    double x1,
    double y1,
    double x2,
    double y2,
    double margin,
    int imgW,
    int imgH,
  ) {
    final bw = x2 - x1;
    final bh = y2 - y1;
    return (
      (x1 - bw * margin).clamp(0, imgW).toInt(),
      (y1 - bh * margin).clamp(0, imgH).toInt(),
      (x2 + bw * margin).clamp(0, imgW).toInt(),
      (y2 + bh * margin).clamp(0, imgH).toInt(),
    );
  }
}
