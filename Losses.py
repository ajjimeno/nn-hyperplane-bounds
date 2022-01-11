import torch
import torch.nn as nn

class MultiHuberLoss(nn.Module):
    def __init__(self):
        super(MultiHuberLoss, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, input, target):
        m = (torch.ones(input.shape)*-1).to(self.dummy_param.device)

        for i in range(len(target)):
            m[i,target[i]] = 1

        m = m*input

        return torch.sum(torch.where(m>=-1.0, torch.max(torch.tensor(0.0).to(self.dummy_param.device),1-m)**2, -4*m))/(len(target))

        #for i in range(len(input)):
        #    for j in range(len(input[i])):
        #        if j != target[i]:
        #            y = -1
        #        else:
        #            y = 1

        #        yh = y*input[i,j] 
 
        #        if yh >= -1 :
        #            output += torch.max(torch.tensor(0.0, requires_grad=True).to(self.dummy_param.device), (1 - yh))**2
        #        else: 
        #            output -= 4*yh
        #return (output/len(target))/len(input)