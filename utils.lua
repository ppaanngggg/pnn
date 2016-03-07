function pnn.recursiveCuda(input)
    if torch.type(input) == 'table' then
        local ret = {}
        for k,v in ipairs(input) do
            ret[k] = pnn.recursiveCuda(v)
        end
        return ret
    elseif torch.isTensor(input) then
        return input:cuda()
    else
        return input
    end
end

function pnn.recursiveDouble(input)
    if torch.type(input) == 'table' then
        local ret = {}
        for k,v in ipairs(input) do
            ret[k] = pnn.recursiveDouble(v)
        end
        return ret
    elseif torch.isTensor(input) then
        return input:double()
    else
        return input
    end
end

function pnn.datasetBatch(dataset, batchSize)
    local nBatch = #dataset / batchSize
    local lastBatchSize = #dataset % batchSize
    local newDataset = {}
    local index = 1
    for i = 1,nBatch do
        newDataset[i] = {{},{}}
        for j = 1,batchSize do
            newDataset[i][1][j] = dataset[index][1]
            newDataset[i][2][j] = dataset[index][2]
            index = index + 1
        end
    end
    if lastBatchSize > 0 then
        local lastBatch = #newDataset + 1
        newDataset[lastBatch] = {{},{}}
        for j = 1,lastBatchSize do
            newDataset[lastBatch][1][j] = dataset[index][1]
            newDataset[lastBatch][2][j] = dataset[index][2]
            index = index + 1
        end
    end
    return newDataset
end
