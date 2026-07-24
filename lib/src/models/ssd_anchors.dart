import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart'
    show SSDAnchorOptions, generateAnchors;

/// SSD anchor boxes for the SSDLite320 detector.
///
/// Shape: 3234 anchors x 4 values (cx, cy, w, h) in 320px input space.
/// Feature levels: 20x20, 10x10, 5x5, 3x3, 2x2, 1x1 with 6 anchors per location.
///
/// Generated once on first use rather than shipped as a literal table. The
/// generated values reproduce checkpoints/ssdlite_anchors.npy to within the
/// 6-decimal rounding of that export (max 4.1e-5 px); test/ssd_anchors_test.dart
/// pins the equivalence against the exported table.
///
/// Stored in centre form because that is what [generateAnchors] emits and what
/// the box decoder consumes, so neither side pays for a corner round-trip.
final Float64List ssdAnchors = _buildSsdAnchors();

/// TF OD API `create_ssd_anchors` configuration for SSDLite320.
///
/// minScale/maxScale 0.2-0.95 over 6 layers give per-level scales of 0.20,
/// 0.35, 0.50, 0.65, 0.80, 0.95, and each location carries the five aspect
/// ratios plus one interpolated-scale anchor at ratio 1.0.
const SSDAnchorOptions _kSsdLite320Options = SSDAnchorOptions(
  numLayers: 6,
  minScale: 0.2,
  maxScale: 0.95,
  inputSizeHeight: 320,
  inputSizeWidth: 320,
  anchorOffsetX: 0.5,
  anchorOffsetY: 0.5,
  // flutter_litert derives each feature map as ceil(inputSize / stride). These
  // are strides that reproduce the model's 20/10/5/3/2/1 grids at 320px. 107
  // and 160 are not real network strides, they are simply values that ceil to
  // 3 and 2; recompute all six if the input size ever changes.
  strides: [16, 32, 64, 107, 160, 320],
  aspectRatios: [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
  reduceBoxesInLowestLayer: false,
  interpolatedScaleAspectRatio: 1.0,
  fixedAnchorSize: false,
);

/// Maps a flutter_litert anchor slot to the model's slot within each location.
///
/// flutter_litert appends the interpolated-scale anchor last; the exported
/// table places it second. Everything else keeps its order.
const List<int> _kAnchorSlotOrder = [0, 5, 1, 2, 3, 4];

/// Number of anchors generated per feature-map location.
const int _kAnchorsPerLocation = 6;

Float64List _buildSsdAnchors() {
  final List<List<double>> raw = generateAnchors(_kSsdLite320Options);
  final double width = _kSsdLite320Options.inputSizeWidth.toDouble();
  final double height = _kSsdLite320Options.inputSizeHeight.toDouble();
  final out = Float64List(raw.length * 4);

  for (int block = 0; block < raw.length; block += _kAnchorsPerLocation) {
    for (int slot = 0; slot < _kAnchorsPerLocation; slot++) {
      final List<double> anchor = raw[block + _kAnchorSlotOrder[slot]];
      final int o = (block + slot) * 4;
      out[o + 0] = anchor[0] * width;
      out[o + 1] = anchor[1] * height;
      // The exported table caps anchor extent at the image size, keeping the
      // centre. Not cosmetic: w/h are the exponential decode scale, so an
      // unclamped anchor changes decoded box sizes on the coarsest levels.
      out[o + 2] = math.min(anchor[2], 1.0) * width;
      out[o + 3] = math.min(anchor[3], 1.0) * height;
    }
  }

  return out;
}
