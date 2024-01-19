import torch
import numpy as np
import numpy

def arrange(v1, v2, l1, l2):
    n = len(v1)
    m = n // 2
    video1_1, video1_2 = v1.split(m)    
    if l1.tolist() != l2.tolist():
        video2_1, video2_2 = v2.split(m)
    else:
        video2_1, video2_2 = v2[m:], v2[:m]
    return video1_1, video1_2, video2_1, video2_2


def make_pair(p1, p2):
    new_vid1 = torch.repeat(p1.unsqueeze(1), 1, len(p2)).view(-1, len(p1))
    new_vid2 = p2.repeat(len(p1))
    return new_vid1, new_vid2

def convert(inp):
    p = numpy.asarray(inp,dtype=object)
    p_t = torch.from_numpy(p)
    return p_t

def final(v1,v2,l1,l2):
    v1_1 , v1_2, v2_1, v2_2 = arrange(v1, v2, l1, l2)
    modal1 , modal2 = make_pair(v1_1, v2_1)
    mod1, mod2 = make_pair(v1_2, v2_2)
    video1 = torch.cat([modal1, mod1], dim=0)
    video2 = torch.cat([modal2, mod2], dim=0)
    return video1, video2
