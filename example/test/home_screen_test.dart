import 'package:flutter_test/flutter_test.dart';

import 'package:animal_detection_example/main.dart';

void main() {
  testWidgets('home exposes live, still-image, and video demos',
      (tester) async {
    await tester.pumpWidget(const AnimalDetectionApp());

    expect(find.text('Choose a Demo'), findsOneWidget);
    expect(find.text('Live Camera'), findsOneWidget);
    expect(find.text('Still Image'), findsOneWidget);
    expect(find.text('Video File'), findsOneWidget);
  });
}
