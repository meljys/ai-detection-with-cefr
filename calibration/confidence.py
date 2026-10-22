"""
Confidence of a flag/noflag decision, between 0 and 1

confidence = 0.5 * margin + 0.5 * certainty

margin = |calibrated score - 0.5| *2 (how far the score is from decision line, far = surer)

certainty = 1 - normalised entropy of the 8 class probs ( 1 = model puts all its weight on one class, 0 = spread evenly)
"""

import numpy as np

def confidence(calibrated_score, probs):
    """
    calibrated_score: array of shape (n,), from apply_thresholds (0.5= on decision line)
    probs: array of shape (n, 8), the model probs for the same texts
    returns an array of shape (n,) with values between 0 and 1
    """
    margin = np.abs(calibrated_score - 0.5) * 2.0 # far from decision line = surer (0..1)
    p = np.clip(probs, 1e-12, 1.0) # avoid log (0)
    entropy = -(p * np.log(p)).sum(axis=1) / np.log(probs.shape[1])
    # how spread out the 8 numbers are, scaled 0..1 (0 = all on one class)
    return 0.5 * margin + 0.5 * (1.0-entropy) #average of how far from line and how model is certain