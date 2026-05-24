Pod::Spec.new do |s|
  s.name                  = 'animal_detection'
  s.version               = '1.0.0'
  s.summary               = 'Animal detection via TensorFlow Lite (macOS)'
  s.description           = 'Flutter plugin for on-device animal detection using TensorFlow Lite.'
  s.homepage              = 'https://github.com/hugocornellier/animal_detection'
  s.license               = { :type => 'MIT' }
  s.authors               = { 'Hugo Cornellier' => 'hugo@example.com' }
  s.source                = { :path => '.' }

  s.platform              = :osx, '11.0'

end
