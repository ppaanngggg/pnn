local BatchTrainer = torch.class('nn.BatchTrainer')

function BatchTrainer:__init(module, criterion, optim, optimParams)
    self.module = module
    self.x, self.dldx = self.module:getParameters()

    self.criterion = criterion
    self.optim = optim
    self.optimParams = optimParams
end

function BatchTrainer:train(dataset, batch_size, loopTime)
    self.dataset = dataset
    self.batch_size = batch_size

    print('loop', '|', 'err', '|', 'time')
    local best_f = nil

    for loop = 1,loopTime do
        local begin_time = torch.tic()
        local total_f = 0
        local num_f = 0
        -- rand sample
        rand_index = torch.randperm(#dataset)
        for batch_num = 1,#dataset / batch_size do
            -- loop over the whole batch and acc the grad
            local func = function(x)
                local batch_base = (batch_num - 1) * batch_size
                self.module:zeroGradParameters()
                local batch_total_f = 0
                for offset = 1,batch_size do
                    local sample = dataset[rand_index[batch_base + offset]]
                    local f = self.criterion:forward(self.module:forward(sample[1]), sample[2])
                    self.module:backward(sample[1], self.criterion:backward(self.module.output, sample[2]))
                    batch_total_f = batch_total_f + f
                    total_f = total_f + f
                    num_f = num_f + 1
                end
                return batch_total_f, self.dldx
            end
            -- update params
            self.optim(func, self.x, self.optimParams)
        end
        -- save module and update best_f
        local avg_f = total_f / num_f
        if best_f == nil or avg_f < best_f then
            best_f = avg_f
            torch.saveobj('model', self.module)
        end
        print(loop, avg_f, torch.tic() - begin_time)
    end

end
