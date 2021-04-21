import os
from pathlib import Path

import sounddevice as sd
import soundfile as sf

from speechbrain.pretrained import TransformerASR


def audio_to_string(duration=5, sample_rate=48000, voice_path='data/voice_model/'):
    voice_path = Path(voice_path)
    print('Loading voice recognition model...')

    asr_model = TransformerASR.from_hparams(
        source="speechbrain/asr-transformer-transformerlm-librispeech",
        savedir=str(voice_path),
    )

    input(f'Ready to record {duration} seconds of audio. Press [ENTER] ')
    selection = 'n'

    transcription = ''

    while 'n' in selection:
        print('Recording...')
        recording = sd.rec(int(duration * sample_rate),
                           samplerate=sample_rate,
                           channels=2)
        sd.wait()

        print('Playing your recording...')
        sd.play(recording, sample_rate)
        sf.write(str(voice_path/'recording.wav'), recording, sample_rate, format='WAV')

        print('Performing transcription...')
        transcription = asr_model.transcribe_file(str(voice_path/'recording.wav'))
        print('The transcription: ', transcription)

        selection = input('Keep this result? [y/n]: ').lower()

    os.remove('./recording.wav')

    return [transcription]


if __name__ == '__main__':
    print(audio_to_string())
