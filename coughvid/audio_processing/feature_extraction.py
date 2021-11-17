import numpy as np
from scipy.stats import kurtosis


''' FUNCTIONS FOR FEATURE EXTRACTION '''


def spec_to_image(spec, eps=1e-6):
    mean = np.mean(spec)
    std = np.std(spec)
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = np.min(spec), np.max(spec)
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled


def normalize_audio(audio):
    '''Normalize the audio signal according to the formula in
    https://www.sciencedirect.com/science/article/pii/S0010482521003668?via%3Dihub
    '''
    return 0.9 * audio / np.max(audio)


def zcr(frame):
    '''Calculate the number of times the signal passes through zero,
    a.k.a. the zero-crossing rate.
    '''
    zero_crosses = len(np.nonzero(np.diff(frame > 0))[0])
    return zero_crosses


def log_energy(frame):
    '''Calculate the log energy of the audio signal.
    '''
    return np.log2(max(1, np.sum(np.power(frame, 2))))


def extract_frames(valid_samples, num_frames, frame_length, uuid=None):
    '''Extract frames number of frames of frame_length length.
    '''
    # if not len(valid_samples) >= frame_length * frames:
    #   print(f'WARNING: {len(valid_samples)} frames found,'
    #         f'need at least {frame_length * frames}' )

    # frame_skip = int(np.ceil(len(valid_samples)*1.0/frames))
    # assert frame_skip > 0

    frames = []

    for i in np.linspace(0, len(valid_samples)-1, num_frames, endpoint=False):
        frame = valid_samples[int(i):int(i)+frame_length]

        if len(frame) < frame_length:
            print(f'WARNING: Unexpected frame length encountered at {int(i)}'
                  f' of {len(valid_samples)}: {len(frame)}. '
                  f'Padding {frame_length-len(frame)} frames.')

            frame = np.pad(frame,
                           (0, max(0, frame_length-len(frame))), 'constant')
        elif len(frame) > frame_length:
            frame = frame[:frame_length-1]

        # assert len(frame) == frame_length,\
        #    f'Frame length of {len(frame)} detected in {uuid}.'

        frames += [frame]

    assert len(frames) == num_frames,\
        f'Only {len(frames)} frames extracted, need {frames}.'

    return np.array(frames, dtype=np.float32)


def extract_other_features(frame):
    '''Extract frame kurtosis, frame log energy and frame zero-crossing rate.
    '''
    features = np.array([kurtosis(frame), log_energy(frame), zcr(frame)])
    return features
