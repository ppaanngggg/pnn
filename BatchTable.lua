local BatchTable, parent = torch.class('nn.BatchTable', 'nn.Container')

function BatchTable:__init(module)
    parent.__init(self)

    self.module = module
    self.modules[1] = self.module
    self.sharedModules = {}
end

function BatchTable:add(module)
    assert(true, "BatchTable can't add module")
end

function BatchTable:updateOutput(input)
    assert(torch.type(input) == 'table', "input should be table")
    self.output = {}
    param, gradParam = self.module:parameters()
    for i = 1,#input do
        self.sharedModules[i] = self.sharedModules[i] or self.module:clone('weight', 'bias', 'gradWeight', 'gradBias')
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

BatchTable.accUpdateGradParameters = BatchTable.sharedAccUpdateGradParameters
