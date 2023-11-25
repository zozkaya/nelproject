## file for classifier code 
import random 
outputs = ["no movement", "clamp close", "clamp open", "wrist left", "wrist right", "arm out", "arm in", "arm up", "arm_down"] 

def classify_gesture():
    i = random.randint(0, 8)

    return outputs[i]
