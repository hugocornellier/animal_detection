// Host-side driver so integration tests can run in profile mode (AOT), which
// `flutter test` cannot do. Required for representative performance numbers,
// since debug-mode Dart heavily inflates boxed-list work.
//
//   flutter drive --profile \
//     --driver=test_driver/integration_test.dart \
//     --target=integration_test/tensor_path_bench_test.dart -d macos
import 'package:integration_test/integration_test_driver.dart';

Future<void> main() => integrationDriver();
