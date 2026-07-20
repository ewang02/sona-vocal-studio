# Third-party notices

Sona uses and redistributes components from the following projects. These
notices apply only to the identified third-party components, not to Sona as a
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
weights to Git. `npm run pipeline:setup` downloads the selected public UVR
model assets from their canonical release URLs and verifies their exact sizes
and SHA-256 hashes before use.

The public availability of a model asset is not itself a grant of additional
rights. Users are responsible for confirming that their intended use of each
downloaded model and processed recording is permitted.

## Other dependencies

JavaScript and Python dependencies installed from `package-lock.json` and
`requirements-pipeline.txt` remain subject to their respective licenses. Their
source packages include the applicable license texts and notices.
