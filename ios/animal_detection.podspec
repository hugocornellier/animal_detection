#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint animal_detection.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'animal_detection'
  s.version          = '0.0.1'
  s.summary          = 'On-device animal detection, species classification, and body pose estimation.'
  s.description      = <<-DESC
On-device animal detection, species classification, and body pose estimation using TensorFlow Lite.
                       DESC
  s.homepage         = 'https://github.com/hugocornellier/animal_detection'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'email@example.com' }
  s.source           = { :path => '.' }
  s.source_files = 'animal_detection/Sources/animal_detection/**/*'
  s.dependency 'Flutter'
  s.platform = :ios, '13.0'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
  }
  s.swift_version = '5.0'

  s.resource_bundles = {'animal_detection_privacy' => ['animal_detection/Sources/animal_detection/PrivacyInfo.xcprivacy']}
end
