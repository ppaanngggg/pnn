require 'nn'

pnn = {}

require('pnn.utils')

require('pnn.Slice')
require('pnn.SliceTable')

require('pnn.BatchTable')
require('pnn.BatchTableCriterion')
require('pnn.BatchTrainer')


require('pnn.Cycle')
require('pnn.Rnn')
require('pnn.GRU')

if pcall(function () require 'cutorch' end) then
    require('pnn.MultiGPUTrainer')
end

nn.pnn = pnn
