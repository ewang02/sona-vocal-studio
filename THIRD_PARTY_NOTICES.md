# Third-party notices

Sona uses third-party components and redistributes only those identified as
bundled below. These notices apply only to those components, not to Sona as a
whole.

## SoundTouchJS

`public/soundtouch-processor.js` contains the pre-bundled processor from
`@soundtouchjs/audio-worklet` 2.1.0 with `@soundtouchjs/core` inlined.

- Copyright: the SoundTouchJS contributors
- License: Mozilla Public License 2.0 (MPL-2.0)
- Source: https://github.com/cutterbl/SoundTouchJS
- License text: https://www.mozilla.org/MPL/2.0/

The bundled JavaScript is source code and remains available in this repository
under the MPL-2.0. Modifications to that covered file must remain available
under the same license.

## Audio Separator and Ultimate Vocal Remover

The local song-processing pipeline uses `audio-separator`, which is licensed
under the MIT License and derives substantially from Ultimate Vocal Remover.
The separator project specifically asks integrations using UVR-trained models
to credit UVR and its developers.

- Audio Separator: https://github.com/nomadkaraoke/python-audio-separator
- Ultimate Vocal Remover: https://github.com/Anjok07/ultimatevocalremovergui
- UVR public model repository: https://github.com/TRvlvr/model_repo
- Audio Separator license: MIT

Credits named by Audio Separator include Anjok07, DilanBoskan, Kuielab,
Woosung Choi, KimberleyJSN, Hv, and zhzhongshi. Sona does not commit the model
weights to Git. `npm run pipeline:setup` downloads the selected model assets
from their release URLs and verifies their exact sizes and SHA-256 hashes
before use.

Sona selects one of two model sets for the machine it is installed on and
downloads only that set.

The `compact` set is `Kim_Inst.onnx`, `Kim_Vocal_2.onnx`, and
`5_HP-Karaoke-UVR.pth`, from the UVR public model repository above.

The `roformer` set is two Mel-Band RoFormer checkpoints, each with its
configuration file, downloaded from the Audio Separator project's
`model-configs` release:

- `vocals_mel_band_roformer.ckpt` by Kimberley Jensen. Her published
  repository (https://huggingface.co/KimberleyJSN/melbandroformer) declares the
  MIT License. Sona pins the release file by hash and has not independently
  confirmed that it is byte-identical to the file in that repository.
- `mel_band_roformer_karaoke_gabox.ckpt` by Gabox
  (https://huggingface.co/GaboxR67/MelBandRoformers). That repository declares
  no license, so no permission beyond what its author has stated publicly
  should be assumed.

The Mel-Band RoFormer architecture is described in Wang et al., "Mel-Band
RoFormer for Music Source Separation" (2023); Audio Separator's implementation
derives from the MIT-licensed BS-RoFormer project
(https://github.com/lucidrains/BS-RoFormer).

No model asset is embedded in hosted builds or served by Sona's local
companion. Installation is a separate, explicit local action.

The public availability of a model asset is not itself a grant of additional
rights, and the MIT license on inference code does not necessarily determine
the license of separately published weights. Distributors and users are
responsible for confirming that their intended use of each downloaded model
and processed recording is permitted.

## Other dependencies

JavaScript and Python dependencies installed from `package-lock.json` and
`requirements-pipeline.txt` remain subject to their respective licenses. Their
source packages include the applicable license texts and notices.
