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

function MultiGPUTrainer:train(dataset, batch_size, loopTime)
	self.dataset = dataset
	self.batch_size = batch_size

	local per_gpu_batch_size = {}
	local remained = batch_size % #self.gpuTable
	for i = 1,#self.gpuTable do
		per_gpu_batch_size[i] = math.floor(batch_size / #self.gpuTable)
	end
	for i = 1,remained do
		per_gpu_batch_size[i] = per_gpu_batch_size[i] + 1
	end

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

	print('loop', '|', 'err', '|', 'time')
	local best_f = nil
	for loop = 1,loopTime do
		local begin_time = torch.tic()
        local total_f = 0
        local num_f = 0
        -- rand sample
        local rand_index = torch.randperm(#dataset)

		for batch_num = 1,#dataset / batch_size do
			-- loop over the whole batch and acc the grad
			local func = function(x)
				-- get per gpu sample indexes
				local batch_base = (batch_num - 1) * batch_size
				local per_gpu_batch_index = {}
				local batch_index = batch_base + 1
				for gpu_index = 1,#self.gpuTable do
					per_gpu_batch_index[gpu_index] = {}
					for per_gpu_index = 1,per_gpu_batch_size[gpu_index] do
						per_gpu_batch_index[gpu_index][per_gpu_index] = rand_index[batch_index]
						batch_index = batch_index + 1
					end
				end
				-- localize for thread
				local x = self.x
				local dldx = self.dldx
				-- zero module's gradParameters
				dldx:zero()
				-- operate on each GPU
				local batch_total_f = 0
				for gpu_index = 1,#self.gpuTable do
					self.pool:addjob(
						gpu_index,
						function(dataIdx)
							-- 1. update tModule's parameters to current module's parameters
							t_x:copy(x)
							-- 2. zero all gradParameters
							t_dldx:zero()
							-- 3. get this GPU's input and target
							local gpu_batch_total_f = 0
							for _,index in ipairs(dataIdx) do
			                    local sample = tDataset[index]
			                    local f = tCriterion:forward(tModule:forward(sample[1]), sample[2])
			                    tModule:backward(sample[1], tCriterion:backward(tModule.output, sample[2]))
			                    gpu_batch_total_f = gpu_batch_total_f + f
			                end
							return gpu_batch_total_f, pnn.recursiveDouble(t_dldx)
						end,
						function(fx, t_dldx)
							total_f = total_f + fx
							batch_total_f = batch_total_f + fx
							dldx:add(t_dldx)
						end,
						per_gpu_batch_index[gpu_index]
					)
				end
				self.pool:synchronize()

                return batch_total_f, self.dldx
            end
            -- update params
            self.optim(func, self.x, self.optimParams)
			num_f = num_f + batch_size
        end

		-- save module and update best_f
        local avg_f = total_f / num_f
        if best_f == nil or avg_f < best_f then
            best_f = avg_f
            torch.saveobj('nobatch_model', self.module)
        end
        print(loop, avg_f, torch.tic() - begin_time)
	end
end
