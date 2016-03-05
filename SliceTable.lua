local SliceTable, parent = torch.class('nn.SliceTable', 'nn.Module')

function SliceTable:__init(first, last)
    parent.__init(self)
    self.first = first
    self.last = last

    self.output = {}
    self.gradInput = {}
end

function SliceTable:checkRange(input)
    -- check range of first and last
    local first, last
    if self.first < 0 then
        first = #input + self.first + 1
    else
        first = self.first
    end
    if self.last < 0 then
        last = #input + self.last + 1
    else
        last = self.last
    end
    if first < 1 or first > #input or last < 1 or last > #input or last < first then
        error('nn.SliceTable:updateOutput(input) out of range')
    end
    return first, last
end

function SliceTable:updateOutput(input)
    local first, last = self:checkRange(input)
    -- clear output
    for k,v in ipairs(self.output) do self.output[k] = nil end
    -- copy input from first to last
    for i = 1, last - first + 1 do
        self.output[i] = input[first + i - 1]
    end
    return self.output
end

function SliceTable:updateGradInput(input, gradOutput)
    local first, last = self:checkRange(input)
    -- copy grad from first to last
    for i = 1, last - first + 1 do
        self.gradInput[first + i - 1] = gradOutput[i]
    end
    -- fill 0 out of range
    for i = 1, #input do
        if (i < first) or (i > last) then
            self.gradInput[i] = nn.utils.recursiveResizeAs(self.gradInput[i], input[i])
            nn.utils.recursiveFill(self.gradInput[i], 0)
        end
    end
    -- clear out of #input
    for i = #input + 1, #self.gradInput do
        self.gradInput[i] = nil
    end
    return self.gradInput
end

function SliceTable:type(type, tensorCache)
    self.output = {}
    self.gradInput = {}
    return parent.type(self, type, tensorCache)
end
