local BatchTableCriterion, Criterion = torch.class('nn.BatchTableCriterion', 'nn.Criterion')

function BatchTableCriterion:__init(criterion)
    Criterion.__init(self)
    self.criterion = criterion
end

function BatchTableCriterion:updateOutput(input, target)
    assert(torch.type(input) == 'table', "input should be table")
    assert(torch.type(target) == 'table', "target should be table")
    self.output = 0
    self.gradInput = {}
    for i = 1,#input do
        self.output = self.output + self.criterion:forward(input[i], target[i])
        self.gradInput[i] = self.criterion:backward(input[i], target[i])
    end
    self.output = self.output / #input
    return self.output
end

function BatchTableCriterion:updateGradInput(input, target)
    return self.gradInput
end

function BatchTableCriterion:cuda()
   self.criterion:type('torch.CudaTensor')
   return self
end
