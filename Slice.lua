local Slice, parent = torch.class('nn.Slice', 'nn.Module')

function Slice:__init(first, last)
    parent.__init(self)
    self.first = first
    self.last = last
end

function Slice:checkRange(input)
    -- check range of first and last
    local first, last, len_input
    len_input = input:size()[1]
    if self.first < 0 then
        first = len_input + self.first + 1
    else
        first = self.first
    end
    if self.last < 0 then
        last = len_input + self.last + 1
    else
        last = self.last
    end
    if first < 1 or first > len_input or last < 1 or last > len_input or last < first then
        error('nn.Slice:updateOutput(input) out of range')
    end
    return first, last
end

function Slice:updateOutput(input)
    local first, last = self:checkRange(input)
    local output = input[{{first, last},{}}]
    self.output = self.output:typeAs(output)
    self.output:resizeAs(output):copy(output)
    return self.output
end

function Slice:updateGradInput(input, gradOutput)
    local first, last = self:checkRange(input)
    self.gradInput:resizeAs(input):zero()
    self.gradInput[{{first, last},{}}]:copy(gradOutput)
    return self.gradInput
end
