import 'package:animal_detection/src/models/ssd_anchors.dart';
import 'package:flutter_test/flutter_test.dart';

import 'fixtures/ssd_anchors_reference.dart';

/// Pins the runtime anchor generator against the anchor table exported from
/// checkpoints/ssdlite_anchors.npy.
///
/// The generated values are full float64 while the exported table is rounded to
/// 6 decimals, so these are equal to within that rounding rather than bitwise.
/// The bound is asserted, not assumed: anything above it means the generator
/// configuration drifted, not that the export lost precision.
void main() {
  // 6-decimal rounding on values up to 320 gives at most 5e-7 per coordinate;
  // the observed worst case across all 3234 anchors is 4.1e-5 after the corner
  // to centre conversion. Anything past 1e-4 is a real divergence.
  const double tolerance = 1e-4;
  const int expectedAnchors = 3234;
  const int anchorsPerLocation = 6;

  group('ssdAnchors', () {
    test('reproduces the exported SSDLite320 anchor table', () {
      expect(kSsdAnchorsReference.length, expectedAnchors * 4);
      expect(ssdAnchors.length, kSsdAnchorsReference.length);

      double worst = 0.0;
      int worstIndex = -1;

      for (int i = 0; i < expectedAnchors; i++) {
        final int o = i * 4;
        // Reference is corner form (x1, y1, x2, y2); ssdAnchors is centre form.
        final double x1 = kSsdAnchorsReference[o + 0];
        final double y1 = kSsdAnchorsReference[o + 1];
        final double x2 = kSsdAnchorsReference[o + 2];
        final double y2 = kSsdAnchorsReference[o + 3];
        final List<double> expected = <double>[
          x1 + (x2 - x1) / 2,
          y1 + (y2 - y1) / 2,
          x2 - x1,
          y2 - y1,
        ];

        for (int k = 0; k < 4; k++) {
          final double error = (ssdAnchors[o + k] - expected[k]).abs();
          if (error > worst) {
            worst = error;
            worstIndex = i;
          }
        }
      }

      expect(
        worst,
        lessThan(tolerance),
        reason: 'anchor $worstIndex diverges from the exported table by $worst '
            'px, above the ${tolerance}px rounding budget',
      );
    });

    test('caps anchor extent at the image size, preserving the centre', () {
      // The coarsest levels generate anchors wider or taller than the input.
      // The export clamps them and the decoder uses w/h as the exponential
      // scale, so dropping the clamp silently changes decoded box sizes.
      int clamped = 0;
      for (int o = 0; o < ssdAnchors.length; o += 4) {
        expect(ssdAnchors[o + 2], lessThanOrEqualTo(320.0));
        expect(ssdAnchors[o + 3], lessThanOrEqualTo(320.0));
        if (ssdAnchors[o + 2] == 320.0 || ssdAnchors[o + 3] == 320.0) {
          clamped++;
        }
      }
      expect(clamped, greaterThan(0));
    });

    test('places the interpolated-scale anchor in the model slot', () {
      // Within each location the exported order is:
      //   [ratio 1.0 @ scale, ratio 1.0 @ interpolated scale, 2.0, 0.5, 3.0, 1/3]
      // so slot 1 is square and strictly larger than slot 0.
      for (int block = 0;
          block < expectedAnchors;
          block += anchorsPerLocation) {
        final int a = block * 4;
        final int b = (block + 1) * 4;
        expect(ssdAnchors[a + 2], closeTo(ssdAnchors[a + 3], tolerance));
        expect(ssdAnchors[b + 2], closeTo(ssdAnchors[b + 3], tolerance));
        expect(ssdAnchors[b + 2], greaterThan(ssdAnchors[a + 2]));
      }
    });

    test('is generated once and cached', () {
      expect(identical(ssdAnchors, ssdAnchors), isTrue);
    });
  });
}
