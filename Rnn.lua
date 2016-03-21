local Rnn, parent = torch.class('nn.Rnn', 'nn.Container')

function Rnn:__init(module, initPrevOutput, isContinue)
    parent.__init(self)

    assert(initPrevOutput, "expecting initPrevOutput")

    self.module = module
    self.modules[1] = self.module
    self.sharedModules = {}

    self.initPrevOutput = initPrevOutput
    self.isContinue = isContinue
end

function Rnn:add(module)
    assert(true, "Rnn can't add module")
end

function Rnn:updateOutput(input)
    -- input is {x_1, x_2, ...}
    assert(torch.type(input) == 'table', "expecting input table")
    self.rho = #input

    self.prevOutput = self.isContinue and self.output[#self.output] or self.initPrevOutput
    self.output = {}
    for i = 1,self.rho do
        self.sharedModules[i] = self.sharedModules[i] or self.module:clone('weight', 'bias', 'gradWeight', 'gradBias')
        if i == 1 then
            self.output[i] = self.sharedModules[i]:updateOutput({input[i], prevOutput})
        else
            self.output[i] = self.sharedModules[i]:updateOutput({input[i], self.output[i - 1]})
        end
    end
    return self.output
end

function Rnn:updateGradInput(input, gradOutput)
end

function Rnn:accGradParameters(input, gradOutput, scale)
end

Rnn.accUpdateGradParameters = Rnn.sharedAccUpdateGradParameters
