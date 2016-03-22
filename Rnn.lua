local Rnn, parent = torch.class('nn.Rnn', 'nn.Container')

function Rnn:__init(module, initPrevOutput, isContinue)
    parent.__init(self)

    assert(initPrevOutput, "expecting initPrevOutput")

    self.module = module
    self.modules[1] = self.module
    self.sharedModules = {}

    self.initPrevOutput = initPrevOutput
    self.isContinue = isContinue
    self.output = nil
end

function Rnn:add(module)
    assert(true, "Rnn can't add module")
end

function Rnn:updateOutput(input)
    -- input is {x_1, x_2, ...}
    assert(torch.type(input) == 'table', "expecting input table")
    self.rho = #input

    if self.isContinue and self.output then
        self.prevOutput = self.output[#self.output]
    else
        self.prevOutput = self.initPrevOutput
    end
    self.output = {}
    for i = 1,self.rho do
        self.sharedModules[i] = self.sharedModules[i] or self.module:clone('weight', 'bias', 'gradWeight', 'gradBias')
        if i == 1 then
            self.output[i] = self.sharedModules[i]:updateOutput({input[i], self.prevOutput})
        else
            self.output[i] = self.sharedModules[i]:updateOutput({input[i], self.output[i - 1]})
        end
    end
    return self.output
end

function Rnn:updateGradInput(input, gradOutput)
    assert(torch.type(input) == 'table', "expecting input table")
    assert(#gradOutput == self.rho, "#gradOutput must be equal with rho")

    self.lastGradOutput = gradOutput[self.rho]:clone()
    if self.isContinue and self._combinedGrad then
        self.lastGradOutput:add(self._combinedGrad[1])
    end

    local gradInput
    self.gradInput = {}
    self._combinedGrad = {}
    if self.rho == 1 then
        gradInput = self.sharedModules[1]:updateGradInput(
            {input[1], self.prevOutput},
            self.lastGradOutput
        )
        self.gradInput[1] = gradInput[1]
        self._combinedGrad[1] = gradInput[2]
    else
        for i = self.rho,1,-1 do
            if i == self.rho then
                gradInput = self.sharedModules[self.rho]:updateGradInput(
                    {input[self.rho], self.output[self.rho - 1]},
                    self.lastGradOutput
                )
            else
                self._combinedGrad[i + 1]:add(gradOutput[i])
                if i == 1 then
                    gradInput = self.sharedModules[i]:updateGradInput(
                            {input[i], self.prevOutput},
                            self._combinedGrad[i + 1]
                        )
                else
                    gradInput = self.sharedModules[i]:updateGradInput(
                            {input[i], self.output[i - 1]},
                            self._combinedGrad[i + 1]
                        )
                end
            end
            self.gradInput[i] = gradInput[1]
            self._combinedGrad[i] = gradInput[2]
        end
    end
    return self.gradInput
end

function Rnn:accGradParameters(input, gradOutput, scale)
    assert(torch.type(input) == 'table', "expecting input table")
    assert(#gradOutput == self.rho, "#gradOutput must be equal with rho")
    if self.rho == 1 then
        self.sharedModules[1]:accGradParameters(
            {input[1], self.prevOutput}, self.lastGradOutput, scale
        )
    else
        for i = self.rho,1,-1 do
            if i == self.rho then
                self.sharedModules[self.rho]:accGradParameters(
                    {input[self.rho], self.output[self.rho - 1]}, self.lastGradOutput, scale
                )
            elseif i == 1 then
                self.sharedModules[i]:accGradParameters(
                    {input[i], self.prevOutput}, self._combinedGrad[i + 1], scale
                )
            else
                self.sharedModules[i]:accGradParameters(
                    {input[i], self.output[i - 1]}, self._combinedGrad[i + 1], scale
                )
            end
        end
    end
end

function Rnn:reset()
    self.output = nil
    self._combinedGrad = nil
    self.gradInput = nil
end

Rnn.accUpdateGradParameters = Rnn.sharedAccUpdateGradParameters
