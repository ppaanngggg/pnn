function pnn.GRU(inputSize, outputSize)
    local r_z = nn.Sequential()
        :add(nn.JoinTable(2))
        :add(
            nn.ConcatTable()
                :add(
                    nn.Sequential()
                        :add(nn.Linear(inputSize + outputSize, 1, false))
                        :add(nn.Sigmoid())
                )
                :add(
                nn.Sequential()
                    :add(nn.Linear(inputSize + outputSize, 1, false))
                    :add(nn.Sigmoid())
                )
        )
    local x_h_r_z = nn.Sequential()
        :add(
            nn.ConcatTable()
                :add(nn.Identity())
                :add(r_z)
        )
        :add(nn.FlattenTable())
    local model = nn.Sequential()
        :add(x_h_r_z)
        :add(
            nn.ConcatTable()
                :add(nn.SelectTable(2))
                :add(nn.SelectTable(1))
                :add(
                    nn.Sequential()
                        :add(nn.NarrowTable(2,2))
                        :add(
                            nn.ParallelTable()
                                :add(nn.Identity())
                                :add(nn.Replicate(outputSize, 2))
                        )
                        :add(nn.CMulTable())
                )
                :add(
                    nn.SelectTable(4)
                )
        )
        :add(
            nn.ConcatTable()
                :add(
                    nn.Sequential()
                        :add(nn.SelectTable(4))
                        :add(
                            nn.ConcatTable()
                                :add(
                                    nn.Sequential()
                                        :add(nn.MulConstant(-1))
                                        :add(nn.AddConstant(1))
                                )
                                :add(nn.Identity())
                        )
                        :add(nn.JoinTable(2))

                )
                :add(nn.SelectTable(1))
                :add(
                    nn.Sequential()
                        :add(nn.NarrowTable(2,2))
                        :add(nn.JoinTable(2))
                        :add(nn.Linear(inputSize + outputSize, outputSize, false))
                        :add(nn.Tanh())
                )
        )
        :add(
            nn.ConcatTable()
                :add(nn.SelectTable(1))
                :add(nn.NarrowTable(2,2))
        )
        :add(nn.MixtureTable())
    return model
end
