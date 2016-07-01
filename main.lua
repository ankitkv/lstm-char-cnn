--[[
Trains a word-level or character-level (for inputs) lstm language model
Predictions are still made at the word-level.

Much of the code is borrowed from the following implementations
https://github.com/karpathy/char-rnn
https://github.com/wojzaremba/lstm
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'lfs'
require 'util.misc'
require 'optim'

BatchLoader = require 'util.BatchLoaderUnk'
model_utils = require 'util.model_utils'

local stringx = require('pl.stringx')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/mimic','data directory. Should contain train.txt/valid.txt/test.txt with input data')
cmd:option('-context_size',2,'how many words on either side to consider as context')
-- model params
cmd:option('-highway_layers', 2, 'number of highway layers')
cmd:option('-char_vec_size', 15, 'dimensionality of character embeddings')
cmd:option('-feature_maps', '{50,100,150,200,200,200,200}', 'number of feature maps in the CNN')
cmd:option('-kernels', '{1,2,3,4,5,6,7}', 'conv net kernel widths')
-- optimization
cmd:option('-learning_rate',1e-3,'starting learning rate')
cmd:option('-param_init', 0.05, 'initialize parameters at')
cmd:option('-batch_size',20,'number of sequences to train on in parallel')
cmd:option('-max_epochs',25,'number of full passes through the training data')
cmd:option('-max_grad_norm',5,'normalize gradients at')
cmd:option('-max_word_l',65,'maximum word length')
cmd:option('-neg_samples',5,'number of negative samples')
cmd:option('-alpha',0.75,'smooth unigram frequencies')
cmd:option('-table_size',1e8,'table size from which to sample neg samples')
-- bookkeeping
cmd:option('-seed',3435,'torch manual random number generator seed')
cmd:option('-print_every',500,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every', 1, 'save every n epochs')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','char','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-EOS', '+', '<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')
cmd:option('-time', 0, 'print batch times')
-- GPU/CPU
cmd:option('-gpuid', -1,'which gpu to use. -1 = use CPU')
cmd:option('-cudnn', 0,'use cudnn (1=yes). this should greatly speed up convolutions')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
print(opt)
torch.manualSeed(opt.seed)

-- some housekeeping
loadstring('opt.kernels = ' .. opt.kernels)() -- get kernel sizes
loadstring('opt.feature_maps = ' .. opt.feature_maps)() -- get feature map sizes

-- global constants for certain tokens
opt.tokens = {}
opt.tokens.EOS = opt.EOS
opt.tokens.UNK = '|' -- unk word token
opt.tokens.START = '{' -- start-of-word token
opt.tokens.END = '}' -- end-of-word token
opt.tokens.ZEROPAD = ' ' -- zero-pad token

-- load necessary packages depending on config options
if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1)
end

if opt.cudnn == 1 then
   assert(opt.gpuid >= 0, 'GPU must be used if using cudnn')
   print('using cudnn...')
   require 'cudnn'
end

optim_state = {
    learningRate = opt.learning_rate,
}

-- create the data loader class
loader = BatchLoader.create(opt.data_dir, opt.context_size, opt.batch_size, opt.max_word_l, opt.alpha, opt.table_size)
print('Word vocab size: ' .. #loader.idx2word .. ', Char vocab size: ' .. #loader.idx2char
            .. ', Max word length (incl. padding): ', loader.max_word_l)
opt.max_word_l = loader.max_word_l

-- load model objects. we do this here because of cudnn options
TDNN = require 'model.TDNN'
SkipGram = require 'model.SkipGram'
HighwayMLP = require 'model.HighwayMLP'

-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- define the model: prototypes for one timestep, then clone them in time
print('creating a SkipGram-CNN layer')
charcnn = SkipGram.skipgram(#loader.idx2word,
                    #loader.idx2char, opt.char_vec_size, opt.feature_maps,
                    opt.kernels, loader.max_word_l, opt.highway_layers)
criterion = nn.BCECriterion()

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    charcnn = charcnn:cuda()
    criterion = criterion:cuda()
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(charcnn)
print('number of parameters in the model: ' .. params:nElement())

-- initialization
params:uniform(-opt.param_init, opt.param_init) -- small numbers uniform

-- get layers which will be referenced layer (during SGD or introspection)
function get_layer(layer)
    local tn = torch.typename(layer)
    if layer.name ~= nil then
        if layer.name == 'word_vecs' then
            word_vecs = layer
        elseif layer.name == 'char_vecs' then
            char_vecs = layer
        elseif layer.name == 'cnn' then
            cnn = layer
        end
    end
end
charcnn:apply(get_layer)

function sample_contexts(context, x)
    local labels = torch.zeros(opt.batch_size, (opt.neg_samples + 1) * opt.context_size * 2):int()
    local contexts = torch.IntTensor(opt.batch_size, (opt.neg_samples + 1) * opt.context_size * 2)
    for j = 1, opt.batch_size do
        local i = 1
    --if x[j] ~= loader.word2idx[opt.tokens.EOS] then
        start = opt.context_size + 1
        while start > 1 do
            if context[start-1][{{},1}]:int()[j] ~= loader.word2idx[opt.tokens.EOS] then
                start = start - 1
            else
                break
            end
        end
        stop = opt.context_size
        while stop < opt.context_size * 2 do
            if context[stop+1][{{},1}]:int()[j] ~= loader.word2idx[opt.tokens.EOS] then
                stop = stop + 1
            else
                break
            end
        end
        while start <= stop do
            contexts[j][i] = context[start][{{},1}]:int()[j]
            labels[j][i] = 1
            i = i + 1
            start = start + 1
        end
    --end
        while i <= (opt.neg_samples + 1) * opt.context_size * 2 do
            local neg_context = loader.table[torch.random(opt.table_size)]
            local notfound = true
            for k = 1,i-1 do
                if contexts[j][k] == neg_context then
                    notfound = false
                    break
                end
            end
            if notfound then
                contexts[j][i] = neg_context
                i = i + 1
            end
        end
    end
    return contexts, labels
end

function eval_split(split_idx, max_batches)
    print('evaluating loss over split index ' .. split_idx)
    local n = loader.split_sizes[split_idx]

    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_idx) -- move batch iteration pointer for this split to front
    local loss = 0
    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y, x_char = loader:next_batch(split_idx)
        if opt.gpuid >= 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x = x:float():cuda()
            for context, ydata in ipairs(y) do
                y[context] = ydata:float():cuda()
            end
            x_char = x_char:float():cuda()
        end
        -- forward pass
        local contexts, labels = sample_contexts(y, x[{{},1}])
        if opt.gpuid >= 0 then
            contexts = contexts:float():cuda()
            labels = labels:float():cuda()
        end
        local prediction = charcnn:forward({x_char[{{},1}], contexts})
        loss = loss + criterion:forward(prediction, labels)
        if i % 10 == 0 then collectgarbage() end
    end
    loss = loss / n
    --local perp = torch.exp(loss)
    return loss
end

-- do fwd/bwd and return loss, grad_params
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()
    ------------------ get minibatch -------------------
    local x, y, x_char = loader:next_batch(1) --from train
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        for context, ydata in ipairs(y) do
            y[context] = ydata:float():cuda()
        end
        x_char = x_char:float():cuda()
    end
--    print(loader.idx2word[y[1][1][1]], loader.idx2word[y[2][1][1]], '**'..loader.idx2word[x[1][1]]..'**', loader.idx2word[y[3][1][1]], loader.idx2word[y[4][1][1]])
--    print(x_char[{{},1}][1])
    ------------------- forward pass -------------------
    local contexts, labels = sample_contexts(y, x[{{},1}])
--    for k = 1,(opt.neg_samples + 1) * opt.context_size * 2 do
--        io.write(loader.idx2word[contexts[1][k]] .. ' ')
--    end
--    print('')
--    print(labels[1]:view(1,-1))
--    print('')
    if opt.gpuid >= 0 then
        contexts = contexts:float():cuda()
        labels = labels:float():cuda()
    end
    local predictions = charcnn:forward({x_char[{{},1}], contexts})
    local loss = criterion:forward(predictions, labels)
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
        -- backprop through loss, and softmax/linear
    local doutput_t = criterion:backward(predictions, labels)
    local dlst = charcnn:backward({x_char[{{},1}], contexts}, doutput_t)

    ------------------------ misc ----------------------
    return loss, grad_params
end


-- start optimization here
train_losses = {}
val_losses = {}
lr = opt.learning_rate -- starting learning rate which will be decayed
local iterations = opt.max_epochs * loader.split_sizes[1]
if char_vecs ~= nil then char_vecs.weight[1]:zero() end -- zero-padding vector is always zero
for i = 1, iterations do
    local epoch = i / loader.split_sizes[1]

    local timer = torch.Timer()
    local time = timer:time().real

    _, train_loss = optim.adam(feval, params, optim_state) -- fwd/backprop and update params
    train_loss = train_loss[1]
    if char_vecs ~= nil then -- zero-padding vector is always zero
        char_vecs.weight[1]:zero()
        char_vecs.gradWeight[1]:zero()
    end
    train_losses[i] = train_loss

    -- every now and then or on last iteration
    if i % loader.split_sizes[1] == 0 then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        print('Loss: ' .. val_loss .. ' (' ..  100.0*torch.exp(-val_loss) .. '%)')
        val_losses[#val_losses+1] = val_loss
        local savefile = string.format('%s/lm_%s.t7', opt.checkpoint_dir, opt.savefile)
        local checkpoint = {}
        checkpoint.charcnn = charcnn
        checkpoint.criterion = criterion
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = {loader.idx2word, loader.word2idx, loader.idx2char, loader.char2idx}
        checkpoint.lr = lr
        if epoch == opt.max_epochs or epoch % opt.save_every == 0 then
            print('saving checkpoint to ' .. savefile)
            torch.save(savefile, checkpoint)
        end
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.2f), train_loss = %6.4f (%3.2f%%)", i, iterations, epoch, train_loss, 100.0*torch.exp(-train_loss)))
    end
    if i % 10 == 0 then collectgarbage() end
    if opt.time ~= 0 then
       print("Batch Time:", timer:time().real - time)
    end
end

--evaluate on full test set. this just uses the model from the last epoch
--rather than best-performing model. it is also incredibly inefficient
--because of batch size issues. for faster evaluation, use evaluate.lua, i.e.
--th evaluate.lua -model m
--where m is the path to the best-performing model

test_perp = eval_split(3)
print('Perplexity on test set: ' .. test_perp)
