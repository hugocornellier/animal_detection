# Model weights: mixed licensing

Not all files in this directory share the package's Apache 2.0 license.

| File | License | Commercial use |
|---|---|---|
| `superanimal_ssdlite_float16.tflite` | SuperAnimal "Modified MIT" | **No** |
| `superanimal_rtmpose_s_float16.tflite` | SuperAnimal "Modified MIT" | **No** |
| `species_classifier_float16.tflite` | BSD-3-Clause (torchvision) | Yes |
| `species_mapping.json` | Apache 2.0 (this package) | Yes |

The two SuperAnimal files are format conversions of checkpoints from the
[Mathis Laboratory](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped),
licensed for **academic, non-commercial purposes only** and explicitly
**non-transferable**. The HRNet pose model downloaded on demand
(`superanimal_hrnet_w32_float16.tflite`) is the same.

See [`NOTICE`](../../NOTICE) in the package root for the full statement, the
required citation, and the commercial-licensing contacts.
