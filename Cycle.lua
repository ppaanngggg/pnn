local Cycle, parent = torch.class('nn.Cycle', 'nn.Container')

function Cycle:__init(module)
    parent.__init(self)
    self.module = module
    self.modules[1] = self.module
    self.sharedModules = {}
end

function Cycle:updateOutput(input)
    -- input is {real_input, rho}
    assert(torch.type(input) == 'table', "expecting input table")
    local realInput = input[1]
    -- save rho
    local rho = input[2]
    assert(rho:size()[1] == 1, "expecting size()[1] == 1 for arg 2")
    assert(rho:dim() == 1, "expecting dim() == 1 for arg 2")
    --assert(torch.type(rho) == 'torch.LongTensor', "expecting torch.LongTensor value for arg 2")
    rho = rho[1]
    self.rho = rho

    self.output = {}
    for i = 1,self.rho do
        self.sharedModules[i] = self.sharedModules[i] or self.module:clone('weight', 'bias', 'gradWeight', 'gradBias')
        if i == 1 then
            self.output[1] = self.sharedModules[1]:updateOutput(realInput)
        else
            self.output[i] = self.sharedModules[i]:updateOutput(self.output[i - 1])
        end
    end

    return self.output
end

function 
