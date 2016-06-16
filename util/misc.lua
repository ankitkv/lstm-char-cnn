
-- misc utilities

function clone_list(tensor_list, zero_too)
    -- takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end

function tablesum(list)
    local acc
    for k, v in ipairs(list) do
        if 1 == k then
            acc = v
        else
            acc = acc + v
        end
    end
    return acc
end
