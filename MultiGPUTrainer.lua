--[[
	This file implements cuda parallelism for BatchTable.
	It split a bigger batch into smaller batch for each GPU.
]]--

local MultiGPUTrainer = torch.class('nn.MultiGPUTrainer')

require 'nn'
require 'pnn.init'
require 'optim'
require 'cutorch'
require 'cunn'

local threads = require 'threads'

function MultiGPUTrainer:__init(module, criterion, optim, dataset, gpuTable)
	self.module = module
	self.criterion = criterion
	self.optim = optim
	self.dataset = dataset
	self.gpuTable = gpuTable

	self.pool = threads.Threads(
		#self.gpuTable,
		function(threadid)
			require 'nn'
			require 'pnn.init'
			require 'optim'
			require 'cutorch'
			require 'cunn'
			cutorch.setDevice(gpuTable[threadid])
			print('thread',threadid,'on',cutorch.getDevice())
		end,
		function(threadid)
			-- tModule =
		end
	)

end


-- function
