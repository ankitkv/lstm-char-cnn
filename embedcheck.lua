--[[
Evaluates a trained model

Much of the code is borrowed from the following implementations
https://github.com/karpathy/char-rnn
https://github.com/wojzaremba/lstm
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.misc'
require 'util.HLogSoftMax'
utf8 = require 'lua-utf8'

HSMClass = require 'util.HSMClass'
BatchLoader = require 'util.BatchLoaderUnk'
model_utils = require 'util.model_utils'

local stringx = require('pl.stringx')

cmd = torch.CmdLine()
cmd:text('Options')
-- data
cmd:option('-model', 'en-large-word-model.t7', 'model checkpoint file')
-- GPU/CPU these params must be passed in because it affects the constructors
cmd:option('-gpuid', 0,'which gpu to use. -1 = use CPU')
cmd:option('-cudnn', 1,'use cudnn (1 = yes, 0 = no)')
cmd:option('-save', 0,'create and save embeddings (1 = yes, 0 = no)')
cmd:option('-embfile', 'embeddings', 'embeddings file')
cmd:option('-k', 50, 'number of nearest neighbors')
cmd:option('-knn', '', 'get the k nearest neighbors of words (comma-separated)')

cmd:text()

-- parse input params
opt2 = cmd:parse(arg)
if opt2.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt2.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt2.gpuid + 1)
end

if opt2.cudnn == 1 then
    assert(opt2.gpuid >= 0, 'GPU must be used if using cudnn')
    print('using cudnn')
    require 'cudnn'
end

HighwayMLP = require 'model.HighwayMLP'
TDNN = require 'model.TDNN'
LSTMTDNN = require 'model.LSTMTDNN'

checkpoint = torch.load(opt2.model)
opt = checkpoint.opt
protos = checkpoint.protos
print('opt: ')
print(opt)
print('val_losses: ')
print(checkpoint.val_losses)
idx2word, word2idx, idx2char, char2idx = table.unpack(checkpoint.vocab)

function makeCharCNN(model)
    local i = 0
    modules = {}
    for indexNode, node in ipairs(model.forwardnodes) do
        if node.data.module then
            i = i+1
            --print(i)
            --print(node.data['annotations'])
            table.insert(modules, node.data.module)
        end
    end

    seq = nn.Sequential()
    seq:add(modules[2])
    seq:add(modules[3])
    seq:add(modules[5])
    return seq
end

charcnn = makeCharCNN(protos.rnn)
if opt.gpuid >= 0 then
    charcnn = charcnn:cuda()
end

print(charcnn)

-- recreate the data loader class
loader = BatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, opt.padding, opt.max_word_l)
print('Word vocab size: ' .. #loader.idx2word .. ', Char vocab size: ' .. #loader.idx2char
        .. ', Max word length (incl. padding): ', loader.max_word_l)

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(2, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

params, grad_params = model_utils.combine_all_parameters(protos.rnn)
if opt.hsm > 0 then
    hsm_params, hsm_grad_params = model_utils.combine_all_parameters(protos.criterion)
    print('number of parameters in the model: ' .. params:nElement() + hsm_params:nElement())
else
    print('number of parameters in the model: ' .. params:nElement())
end

-- for easy switch between using words/chars (or both)
function get_input(x, x_char, t, prev_states)
    local u = {}
    if opt.use_chars == 1 then
        table.insert(u, x_char[{{1,2},t}])
    end
    if opt.use_words == 1 then
        table.insert(u, x[{{1,2},t}])
    end
    for i = 1, #prev_states do table.insert(u, prev_states[i]) end
    return u
end

function get_embedding(word)
    local x = torch.ones(2, loader.max_word_l)
    local inword = opt.tokens.START .. word .. opt.tokens.END
    local l = utf8.len(inword)
    local i = 1
    for _, char in utf8.next, inword do
        local char = utf8.char(char) -- save as actual characters
        x[{{},i}] = char2idx[char]
        i = i+1
        if i == loader.max_word_l then
            x[{{},i}] = char2idx[opt.tokens.END]
            break
        end
    end
    if opt.gpuid >= 0 then
        x = x:cuda()
    end
    return charcnn:forward(x)[1]
end

function save_embeddings()
    print('Getting embeddings for vocabulary')
    emb_table = {}
    for word, _ in pairs(loader.word2idx) do
        local pout = out
        local out = get_embedding(word)
        emb_table[word] = out:clone()
    end
    print('Saving embeddings')
    torch.save(opt2.embfile, emb_table)
end

function Split(str, delim, maxNb)
    -- Eliminate bad cases...
    if string.find(str, delim) == nil then
        return { str }
    end
    if maxNb == nil or maxNb < 1 then
        maxNb = 0    -- No limit
    end
    local result = {}
    local pat = "(.-)" .. delim .. "()"
    local nb = 0
    local lastPos
    for part, pos in string.gfind(str, pat) do
        nb = nb + 1
        result[nb] = part
        lastPos = pos
        if nb == maxNb then break end
    end
    -- Handle the last field
    if nb ~= maxNb then
        result[nb + 1] = string.sub(str, lastPos)
    end
    return result
end

if opt2.save == 1 then
    save_embeddings()
else
    print('Loading embeddings')
    emb_table = torch.load(opt2.embfile)
    for _, word in ipairs(Split(opt2.knn, ',')) do
        print('\n\n== ' .. word .. ' ==\n')
        input = get_embedding(word)
        cosined = nn.CosineDistance():cuda()
        print('Sorting by distance')
        dist_table = {}
        for k,v in pairs(emb_table) do
            dist_table[k] = cosined:forward({input, v:cuda()})[1]
        end
        function compare(a, b)
            return a[2] > b[2]
        end
        tmp = {}
        for k,v in pairs(dist_table) do table.insert(tmp, {k,v}) end
        table.sort(tmp, compare)
        for i=1,opt2.k do
            print(i, tmp[i][1])
        end
        print('...')
        table.sort(tmp, compare)
        for i=#tmp-opt2.k,#tmp do
            print(i, tmp[i][1])
        end
    end
end
