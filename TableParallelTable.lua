--[[
	This file implements table parallelism for Torch modules.

	The same model is replicated on multiple GPUs. The input is split, typeically
	into smaller mini-batch.
]]--

local TableParallelTable, parent = torch.class('nn.TableParallelTable', 'nn.Container')

require 'pnn.BatchTable'
local threads = require 'threads'

function TableParallelTable:__init(module, gpuTable)
	parent.__init(self)

	self.module = (not torch.isTypeOf(rnn, 'nn.BatchTable')) and nn.BatchTable(module) or module
	self.module:float()
	self.modules[1] = self.module

	self.gpuTable = gpuTable

	local tmpModule = self.module
	self.pool = threads.Threads(
		#gpuTable,
		function(id)
			require 'nn'
			require 'rnn'
			-- require 'pnn.Cycle'
			-- require 'pnn.Slice'
			require 'pnn.init'
			-- require 'cutorch'
			-- require 'cunn'
			-- cutorch.setDevice(gpuTable[id])
		end,
		function(id)
			-- function toCuda(input)
			-- 	if torch.type(input) == 'table' then
			-- 		local ret = {}
			-- 		for k,v in ipairs(input) do
			-- 			ret[k] = toCuda(v)
			-- 		end
			-- 		return ret
			-- 	else
			-- 		return input:cuda()
			-- 	end
			-- end
			function toFloat(input)
				if torch.type(input) == 'table' then
					local ret = {}
					for k,v in ipairs(input) do
						ret[k] = toFloat(v)
					end
					return ret
				else
					return input:float()
				end
			end
			tModule = tmpModule:clone()--:cuda()
		end
	)
	self.pool:specific(true)
end

function TableParallelTable:split(input)
	local inputTable = {}
	local batchSize = #input / #self.gpuTable
	local nBigBatch = #input % #self.gpuTable
	local index = 1
	for i=1,#self.gpuTable do
		inputTable[i] = {}
		local curBatchSize
		if i <= nBigBatch then
			curBatchSize = batchSize + 1
		else
			curBatchSize = batchSize
		end
		for j=1,curBatchSize do
			inputTable[i][j] = input[index]
			index = index + 1
		end
	end
	return inputTable
end

function TableParallelTable:merge(outputTable)
	local output = {}
	index = 1
	for i =1,#outputTable do
		for j =1,#outputTable[i] do
			output[index] = outputTable[i][j]
			index = index + 1
		end
	end
	return output
end

function TableParallelTable:updateOutput(input)
	local inputTable = self:split(input)
	local outputTable = {}
	for i = 1,#self.gpuTable do
		self.pool:addjob(
			i,
			function(tInput)
				-- print(cutorch.getDevice())
				-- local _input = toCuda(input[i])
				tInput = toFloat(tInput)
				local tOutput = tModule:forward(tInput)
				return toFloat(tOutput)
			end,
			function(tOutput)
				outputTable[i] = tOutput
			end,
			inputTable[i]
		)
	end
	self.pool:synchronize()
	self.output = self:merge(outputTable)
	return self.output
end

function TableParallelTable:updateGradInput(input, gradOutput)
	local inputTable = self:split(input)
	local gradOutputTable = self:split(gradOutput)
	local gradInputTable = {}
	for i = 1,#self.gpuTable do
		self.pool:addjob(
			i,
			function(tInput, tGradOutput)
				tInput = toFloat(tInput)
				tGradOutput = toFloat(tGradOutput)
				local tGradInput = tModule:updateGradInput(tInput, tGradOutput)
				return toFloat(tGradInput)
			end,
			function(tGradInput)
				gradInputTable[i] = tGradInput
			end,
			inputTable[i],
			gradOutputTable[i]
		)
	end
	self.pool:synchronize()
	self.gradInput = self:merge(gradInputTable)
	return self.gradInput
end

function TableParallelTable:accGradParameters(input, gradOutput, scale)
	local inputTable = self:split(input)
	local gradOutputTable = self:split(gradOutput)
	local gradParametersTable = {}
	for i = 1,#self.gpuTable do
		self.pool:addjob(
			i,
			function(tInput, tGradOutput)
				tInput = toFloat(tInput)
				tGradOutput = toFloat(tGradOutput)
				tModule:accGradParameters(tInput, tGradOutput)
				tParameters, tGradParameters = tModule:parameters()
				return toFloat(tGradParameters)
			end,
			function(tGradParameters)
				gradParametersTable[i] = tGradParameters
			end,
			inputTable[i],
			gradOutputTable[i]
		)
	end
	self.pool:synchronize()
	local parameters, gradParameters = self.module:parameters()
	for i = 1,#gradParameters do
		for j = 1,#gradParametersTable do
			gradParameters[i]:add(gradParametersTable[j][i])
		end
	end
end

function TableParallelTable:updateParameters(learningRate)
	parent.updateParameters(self, learningRate)
	local parameters = self.module:parameters()
	for i = 1,#self.gpuTable do
		self.pool:addjob(
			i,
			function(mainParameters)
				tParameters = tModule:parameters()
				for i=1,#tParameters do
					tParameters[i]:copy(mainParameters[i])
				end
			end,
			function()	end,
			parameters
		)
	end
	self.pool:synchronize()
end

function TableParallelTable:zeroGradParameters()
	parent.zeroGradParameters(self)
	for i = 1,#self.gpuTable do
		self.pool:addjob(
			i,
			function()
				tModule:zeroGradParameters()
			end
		)
	end
	self.pool:synchronize()
end

function TableParallelTable:training()
	parent.training(self)
	for i = 1,#self.gpuTable do
		self.pool:addjob(
			i,
			function()
				tModule:training()
			end
		)
	end
	self.pool:synchronize()
end

function TableParallelTable:evaluate()
	parent.evaluate(self)
	for i = 1,#self.gpuTable do
		self.pool:addjob(
			i,
			function()
				tModule:evaluate()
			end
		)
	end
	self.pool:synchronize()
end

function TableParallelTable:reset()
	parent.reset(self)
	for i = 1,#self.gpuTable do
		self.pool:addjob(
			i,
			function()
				tModule:reset()
			end
		)
	end
	self.pool:synchronize()
end
