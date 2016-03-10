--[[
	This file implements cuda parallelism for BatchTable.
	It split a bigger batch into smaller batch for each GPU.
]]--

local MultiGPUTrainer = torch.class('nn.MultiGPUTrainer')

require 'nn'
require 'cutorch'
require 'cunn'

local threads = require 'threads'

function MultiGPUTrainer:__init(module, criterion, optim, optimParams, gpuTable)
	self.module = module
	self.x, self.dldx = self.module:getParameters()

	self.criterion = criterion
	self.optim = optim
	self.optimParams = optimParams
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
			cutorch.seed()
			print('thread',threadid,'on',cutorch.getDevice())
		end,
		function(threadid)
			tModule = module:clone():cuda()
			tCriterion = criterion:clone():cuda()
			t_x ,t_dldx = tModule:getParameters()
		end
	)
	self.pool:specific(true)
end


function MultiGPUTrainer:train(dataset, loopTime)
	self.dataset = dataset
	-- store the whole dataset into GPU
	for i = 1,#self.gpuTable do
		self.pool:addjob(
			i,
			function()
				tDataset = pnn.recursiveCuda(dataset)
			end
		)
	end
	self.pool:synchronize()

	-- feval function for optim
	local feval = function(x_new)
		local x = self.x
		local dldx = self.dldx
		-- zero module's gradParameters
		dldx:zero()
		-- operate on each GPU
		local totalFx = 0
		for i = 1,#self.gpuDataIdx do
			self.pool:addjob(
				i,
				function(dataIdx)
					-- 1. update tModule's parameters to current module's parameters
					t_x:copy(x)
					-- 2. zero all gradParameters
					t_dldx:zero()
					-- 3. get this GPU's input and target
					local input = tDataset[dataIdx][1]
					local target = tDataset[dataIdx][2]
					local output = tModule:forward(input)
					local fx = tCriterion:forward(output, target)
					tModule:backward(input, tCriterion:backward(output, target))
					return fx, pnn.recursiveDouble(t_dldx)
				end,
				function(fx, t_dldx)
					totalFx = totalFx + fx
					dldx:add(t_dldx)
				end,
				self.gpuDataIdx[i]
			)
		end
		self.pool:synchronize()
		return totalFx / #self.gpuDataIdx, dldx
	end

	-- split the whole dataset to each GPU
	local subLoopTime = #dataset / #self.gpuTable
	local lastSubLoopSize = #dataset % #self.gpuTable

	print('begin training')
	print('loop', '|', 'err', '|', 'time')
	local minFx = nil
	for i = 1,loopTime do
		local begin_time = torch.tic()
		local totalFx = {}
		-- loop complete loop
		for j = 1,subLoopTime do
			self.gpuDataIdx = {}
			local baseIdx = (j - 1) * #self.gpuTable
			for t = 1,#self.gpuTable do
				self.gpuDataIdx[t] = baseIdx + t
			end
			local _, fx = self.optim(feval, self.x, self.optimParams)
			for fx_idx = 1,#fx do
				if totalFx[fx_idx] == nil then
					totalFx[fx_idx] = fx[fx_idx]
				else
					totalFx[fx_idx] = totalFx[fx_idx] + fx[fx_idx]
				end
			end
		end
		-- last uncomplete loop
		if lastSubLoopSize > 0 then
			self.gpuDataIdx = {}
			for t = 1,lastSubLoopSize do
				self.gpuDataIdx[t] = #dataset - lastSubLoopSize + t
			end
			local _, fx = self.optim(feval, self.x, self.optimParams)
			for fx_idx = 1,#fx do
				if totalFx[fx_idx] == nil then
					totalFx[fx_idx] = fx[fx_idx]
				else
					totalFx[fx_idx] = totalFx[fx_idx] + fx[fx_idx]
				end
			end
		end
		-- avg fx
		local totalLoopTime = subLoopTime
		if lastSubLoopSize > 0 then
			totalLoopTime = totalLoopTime + 1
		end
		for fx_idx = 1,#totalFx do
			totalFx[fx_idx] = totalFx[fx_idx] / totalLoopTime
		end

		print(i, totalFx[1], torch.tic() - begin_time)
		if minFx == nil then
			minFx = totalFx
		end
		if totalFx[1] <= minFx[1] then
			torch.saveobj('model', self.module)
			minFx = totalFx
		end

	end
end
