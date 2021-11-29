import auditok
import numpy as np
import random

#This function is used for spliting and randomly samples the coughs in each audio sample



def split_cough(filename, duration, max_silence, energy_threshold):
    dir_file = ''.join(filename)
    audio_regions = auditok.split(
        dir_file,
        min_dur=duration,        # minimum duration of a valid audio event in seconds
        max_dur=duration,        # maximum duration of an event
        max_silence=max_silence,     #maximum duration of tolerated continuous silence within an event
        energy_threshold=energy_threshold,  # threshold of detection
        strict_min_dur=True
    )
    for i, r in enumerate(audio_regions):
        audio = np.asarray(r)
        x = np.array([[audio]]).astype(np.float32)
        if i == 0:
            x_set = np.array([x])
        else:
            x_set = np.append(x_set,[x], axis = 0)
            
    sample = random.choice(x_set)
    return sample
    


    
 