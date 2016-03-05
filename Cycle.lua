------------------------------------------------------------------------
--[[ Cycle ]]--
-- Encapsulates an AbstractRecurrent instance (rnn) which is repeatedly
-- presented with the input(first time) or prev output(later)
-- for rho(variable) time steps. The output is a table of rho outputs of the rnn.
------------------------------------------------------------------------
assert(not nn.Cycle, "update nnx package : luarocks install nnx")
local Cycle, parent = torch.class('nn.Cycle', 'nn.AbstractSequencer')

function Cycle:__init(module)
    parent.__init(self)

    self.module = (not torch.isTypeOf(rnn, 'nn.AbstractRecurrent')) and nn.Recursor(module) or module

    self.modules[1] = self.module
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

    self.module:maxBPTTstep(rho) -- hijack rho (max number of time-steps for backprop)
    self.module:forget()

    -- clear output
    self.output = {}
    -- cycle rho times
    for step=1,self.rho do
        if step == 1 then
            self.output[1] = nn.rnn.recursiveCopy(
                self.output[1],
                self.module:updateOutput(realInput)
            )
        else
            self.output[step] = nn.rnn.recursiveCopy(
                self.output[step],
                self.module:updateOutput(self.output[step-1])
            )
        end
    end
    return self.output
end

function Cycle:updateGradInput(input, gradOutput)
    local realInput = input[1]
    assert(self.module.step - 1 == self.rho, "inconsistent rnn steps")
    assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
    assert(#gradOutput == self.rho, "gradOutput should have rho elements")


    -- back-propagate through time (BPTT)
    self._gradOutputs = {}
    for step=self.rho,1,-1 do
        self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], gradOutput[step])
        -- add prev gradInput if not last step
        if step ~= self.rho then
            self._gradOutputs[step] = nn.rnn.recursiveAdd(self._gradOutputs[step], self.gradInput)
        end

        if step == 1 then
            -- use input if first step
            self.gradInput = self.module:updateGradInput(realInput, self._gradOutputs[step])
        else
            -- else prev output as input
            self.gradInput = self.module:updateGradInput(self.output[step - 1], self._gradOutputs[step])
        end
    end
    self.gradInput = {self.gradInput, torch.Tensor{0}:typeAs(input[2])}
    return self.gradInput
end

function Cycle:accGradParameters(input, gradOutput, scale)
    local realInput = input[1]
    assert(self.module.step - 1 == self.rho, "inconsistent rnn steps")
    assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
    assert(#gradOutput == self.rho, "gradOutput should have rho elements")

    -- back-propagate through time (BPTT)
    for step=self.rho,1,-1 do
        if step == 1 then
            self.module:accGradParameters(realInput, self._gradOutputs[step], scale)
        else
            self.module:accGradParameters(self.output[step - 1], self._gradOutputs[step], scale)
        end
    end
end

function Cycle:maxBPTTstep(rho)
    self.module:maxBPTTstep(rho)
end

function Cycle:__tostring__()
    local tab = '  '
    local line = '\n'
    local str = torch.type(self) .. ' {' .. line
    str = str .. tab .. '[  input,  output(1),...,output(rho-1)  ]'.. line
    str = str .. tab .. '     V         V             V     '.. line
    str = str .. tab .. tostring(self.modules[1]):gsub(line, line .. tab) .. line
    str = str .. tab .. '     V         V             V     '.. line
    str = str .. tab .. '[ output(1),output(2),...,output(rho) ]' .. line
    str = str .. '}'
    return str
end
