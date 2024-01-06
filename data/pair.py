import torch
import numpy as np
import numpy

def arrange(v1, v2, l1,l2):
    n = len(v1)
    m = int(n/2)
    if l1.tolist() != l2.tolist():
        video1_1 = v1[0:m]
        video1_2 = v1[m:]
        video2_1 = v2[m:]
        video2_2 = v2[0:m]
        return video1_1,video1_2, video2_1, video2_2
    else:
        video1_1 = v1[0:m]
        video1_2 = v1[m:]
        video2_1 = v2[0:m]
        video2_2 = v2[m:]
        return video1_1, video1_2, video2_1, video2_2

def make_pair(p1,p2):
    new_vid1=[]
    new_vid2 =[]
    n =len(p1)
    for i in range(n):
        for j in range(n):
            new_vid1.append(p1[i])
            new_vid2.append(p2[j])
   
    return torch.stack(new_vid1, dim=0), torch.stack(new_vid2,dim=0)

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
