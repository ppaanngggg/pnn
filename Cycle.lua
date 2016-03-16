local Cycle, parent = torch.class('nn.Cycle', 'nn.Container')

function Cycle:__init(module)
    parent.__init(self)

    self.module = module
    self.modules[1] = self.module
    self.sharedModules = {}
end

function Cycle:add(module)
    assert(true, "Cycle can't add module")
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

function Cycle:updateGradInput(input, gradOutput)
    -- input is {real_input, rho}
    assert(torch.type(input) == 'table', "expecting input table")
    local realInput = input[1]
    assert(#gradOutput == self.rho, "#gradOutput must be equal with rho")

    self._combinedGradInput = {}
    if self.rho == 1 then
        self._combinedGradInput[1] =
            self.sharedModules[1]:updateGradInput(
                realInput,
                gradOutput[1]
            )
    else
        for i = self.rho,1,-1 do
            if i == self.rho then
                self._combinedGradInput[self.rho] =
                    self.sharedModules[self.rho]:updateGradInput(
                        self.output[self.rho - 1],
                        gradOutput[self.rho]
                    )
            else
                self._combinedGradInput[i + 1]:add(gradOutput[i])
                if i == 1 then
                    self._combinedGradInput[i] =
                        self.sharedModules[i]:updateGradInput(
                            realInput,
                            self._combinedGradInput[i + 1]
                        )
                else
                    self._combinedGradInput[i] =
                        self.sharedModules[i]:updateGradInput(
                            self.output[i - 1],
                            self._combinedGradInput[i + 1]
                        )
                end
            end
        end
    end
    self.gradInput = {self._combinedGradInput[1], torch.zeros(1):typeAs(input[2])}
    return self.gradInput
end

function Cycle:accGradParameters(input, gradOutput, scale)
    -- input is {real_input, rho}
    assert(torch.type(input) == 'table', "expecting input table")
    local realInput = input[1]
    assert(#gradOutput == self.rho, "#gradOutput must be equal with rho")
    if self.rho == 1 then
        self.sharedModules[1]:accGradParameters(
            realInput, gradOutput[1], scale
        )
    else
        for i = self.rho,1,-1 do
            if i == self.rho then
                self.sharedModules[self.rho]:accGradParameters(
                    self.output[self.rho - 1], gradOutput[self.rho], scale
                )
            elseif i == 1 then
                self.sharedModules[i]:accGradParameters(
                    realInput, self._combinedGradInput[i + 1], scale
                )
            else
                self.sharedModules[i]:accGradParameters(
                    self.output[i - 1], self._combinedGradInput[i + 1], scale
                )
            end
        end
    end
end

Cycle.accUpdateGradParameters = Cycle.sharedAccUpdateGradParameters
