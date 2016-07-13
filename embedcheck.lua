--[[
Evaluates a trained model

Much of the code is borrowed from the following implementations
https://github.com/karpathy/char-rnn
https://github.com/wojzaremba/lstm
]]--

require 'torch'
require 'io'
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
cmd:option('-model', 'cv/lm_char.t7',
           'model checkpoint file, overridden by global modelfile')
-- GPU/CPU these params must be passed in because it affects the constructors
cmd:option('-gpuid', 0,'which gpu to use. -1 = use CPU')
cmd:option('-cudnn', 1,'use cudnn (1 = yes, 0 = no)')
cmd:option('-save', 0,'create and save embeddings (1 = yes, 0 = no)')
cmd:option('-embfile', 'embeddings', 'embeddings file')
cmd:option('-vocabfile', '', 'vocab file')
cmd:option('-vocabembfile', 'vocab_embeddings', 'embeddings for provided vocab')

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

modelfile = modelfile or opt2.model
print('Model file: ', modelfile)
checkpoint = torch.load(modelfile)
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
    if opt.use_words == 1 then
        parallel = nn.ParallelTable()
        model_char = nn.Sequential()
        model_char:add(modules[2])
        model_char:add(modules[3])
        parallel:add(modules[5])
        parallel:add(model_char)
        seq:add(parallel)
        seq:add(modules[6])
        seq:add(modules[7])
    else
        seq:add(modules[2])
        seq:add(modules[3])
        seq:add(modules[5])
    end
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

function get_embedding(word, verbose)
    verbose = verbose or false
    local x_char = torch.ones(2, loader.max_word_l)
    local x = torch.ones(2)
    local inword = opt.tokens.START .. word .. opt.tokens.END
    local l = utf8.len(inword)
    local i = 1
    for _, char in utf8.next, inword do
        local char = utf8.char(char) -- save as actual characters
        x_char[{{},i}] = char2idx[char]
        i = i+1
        if i == loader.max_word_l then
            x_char[{{},i}] = char2idx[opt.tokens.END]
            break
        end
    end
    local known = true
    if word2idx[word] then
        x[{{}}] = word2idx[word]
        if verbose then
            print('[Known word]')
        end
    else
        known = false
        if verbose then
            print('[Unknown word]')
        end
    end
    if opt.gpuid >= 0 then
        x = x:cuda()
        x_char = x_char:cuda()
    end
    if opt.use_words == 1 then
        result = charcnn:forward({x, x_char})[1]
        if not known then
            result[{{-opt.word_vec_size, -1}}]:zero() -- zero out word vec
        end
        return result
    else
        return charcnn:forward(x_char)[1]
    end
end

function generate_embeddings()
    print('Getting embeddings provided vocabulary')
    local i = 0
    f = io.open(opt2.vocabembfile, 'w')
    for word in io.lines(opt2.vocabfile) do
        f:write(word)
        local out = get_embedding(word)
        for j = 1, out:size(1) do
            f:write(string.format(' %.10f', out[j]))
        end
        f:write('\n')
        i = i + 1
        if i % 10 == 0 then
            collectgarbage()
            if i % 100 == 0 then
                print(i)
            end
        end
    end
    f:close()
end

function save_embeddings()
    print('Getting embeddings for vocabulary')
    emb_table = {}
    local i = 0
    for word, _ in pairs(loader.word2idx) do
        local out = get_embedding(word)
        emb_table[word] = out:clone()
        i = i + 1
        if i % 10 == 0 then
            collectgarbage()
        end
    end
    --for k,v in pairs(emb_table) do
    --    print(k, v:sum())
    --end
    print('Saving embeddings')
    torch.save(opt2.embfile, emb_table)
end

if opt2.vocabfile ~= '' then
    generate_embeddings()
elseif opt2.save == 1 then
    save_embeddings()
else
    print('Loading embeddings')
    emb_table = torch.load(opt2.embfile)
end

function get_knn(word, num, verbose)
    num = num or 50
    verbose = verbose or true
    print('\n== ' .. word .. ' ==\n')
    input = get_embedding(word, verbose)
    cosined = nn.CosineDistance():cuda()
    dist_table = {}
    local i = 0
    for k,v in pairs(emb_table) do
        dist_table[k] = cosined:forward({input, v:cuda()})[1]
        i = i + 1
        if i % 10 == 0 then
            collectgarbage()
        end
    end
    function compare(a, b)
        return a[2] > b[2]
    end
    tmp = {}
    for k,v in pairs(dist_table) do table.insert(tmp, {k,v}) end
    table.sort(tmp, compare)
    for i=1,num do
        print(i, tmp[i][1])
    end
    collectgarbage()
end
