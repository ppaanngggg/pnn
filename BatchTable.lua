local BatchTable, parent = torch.class('nn.BatchTable', 'nn.Container')

function BatchTable:__init(module)
    parent.__init(self)

    self.module = module
    self.modules[1] = self.module
end

function BatchTable:updateOutput(input)
    assert(torch.type(input) == 'table', "input should be table")
    self.output = {}
    self.sharedModules = {}
    param, gradParam = self.module:parameters()
    print(param[1][1][1])
    for i = 1,#input do
        self.sharedModules[i] = self.module:clone('weight', 'bias', 'gradWeight', 'gradBias')
        self.output[i] = self.sharedModules[i]:forward(input[i])
    end
    return self.output
end

function BatchTable:updateGradInput(input, gradOutput)
    assert(torch.type(input) == 'table', "input should be table")
    assert(torch.type(gradOutput) == 'table', "gradOutput should be table")
    self.gradInput = {}
    for i = 1,#input do
        self.gradInput[i] = self.sharedModules[i]:updateGradInput(input[i], gradOutput[i])
    end
    return self.gradInput
end

function BatchTable:accGradParameters(input, gradOutput, scale)
    assert(torch.type(input) == 'table', "input should be table")
    assert(torch.type(gradOutput) == 'table', "gradOutput should be table")
    for i = 1,#input do
        self.sharedModules[i]:accGradParameters(input[i], gradOutput[i], scale)
    end
end

function BatchTable:accUpdateGradParameters(input, gradOutput, lr)
    assert(torch.type(input) == 'table', "input should be table")
    assert(torch.type(gradOutput) == 'table', "gradOutput should be table")
    for i = 1,#input do
        self.sharedModules[i]:accUpdateGradParameters(input[i], gradOutput[i], lr)
    end
end
