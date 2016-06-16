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
-- model params
cmd:option('-highway_layers', 2, 'number of highway layers')
cmd:option('-char_vec_size', 15, 'dimensionality of character embeddings')
cmd:option('-feature_maps', '{50,100,150,200,200,200,200}', 'number of feature maps in the CNN')
cmd:option('-kernels', '{1,2,3,4,5,6,7}', 'conv net kernel widths')
-- optimization
cmd:option('-learning_rate',1,'starting learning rate')
cmd:option('-learning_rate_decay',0.5,'learning rate decay')
cmd:option('-decay_when',1,'decay if validation perplexity does not improve by more than this much')
cmd:option('-param_init', 0.05, 'initialize parameters at')
cmd:option('-seq_length',35,'number of timesteps to unroll for')
cmd:option('-batch_size',20,'number of sequences to train on in parallel')
cmd:option('-max_epochs',25,'number of full passes through the training data')
cmd:option('-max_grad_norm',5,'normalize gradients at')
cmd:option('-max_word_l',65,'maximum word length')
-- bookkeeping
cmd:option('-seed',3435,'torch manual random number generator seed')
cmd:option('-print_every',500,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every', 5, 'save every n epochs')
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

-- create the data loader class
loader = BatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, opt.max_word_l)
print('Word vocab size: ' .. #loader.idx2word .. ', Char vocab size: ' .. #loader.idx2char
            .. ', Max word length (incl. padding): ', loader.max_word_l)
opt.max_word_l = loader.max_word_l

hsm = torch.round(torch.sqrt(#loader.idx2word))

-- partition into hsm clusters
-- we want roughly equal number of words in each cluster
HSMClass = require 'util.HSMClass'
require 'util.HLogSoftMax'
mapping = torch.LongTensor(#loader.idx2word, 2):zero()
local n_in_each_cluster = #loader.idx2word / hsm
local _, idx = torch.sort(torch.randn(#loader.idx2word), 1, true)   
local n_in_cluster = {} --number of tokens in each cluster
local c = 1
for i = 1, idx:size(1) do
    local word_idx = idx[i] 
    if n_in_cluster[c] == nil then
        n_in_cluster[c] = 1
    else
        n_in_cluster[c] = n_in_cluster[c] + 1
    end
    mapping[word_idx][1] = c
    mapping[word_idx][2] = n_in_cluster[c]        
    if n_in_cluster[c] >= n_in_each_cluster then
        c = c+1
    end
    if c > hsm then --take care of some corner cases
        c = hsm
    end
end
print(string.format('using hierarchical softmax with %d classes', hsm))


-- load model objects. we do this here because of cudnn and hsm options
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
-- training criterion (negative log likelihood)
criterion = nn.HLogSoftMax(mapping, tablesum(opt.feature_maps))
-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    charcnn = charcnn:cuda()
    criterion = criterion:cuda()
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(charcnn)
-- hsm has its own params
hsm_params, hsm_grad_params = criterion:getParameters()
hsm_params:uniform(-opt.param_init, opt.param_init)
print('number of parameters in the model: ' .. params:nElement() + hsm_params:nElement())

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


function eval_split(split_idx, max_batches)
    print('evaluating loss over split index ' .. split_idx)
    local n = loader.split_sizes[split_idx]
    protos.criterion:change_bias()

    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_idx) -- move batch iteration pointer for this split to front
    local loss = 0
    if split_idx<=2 then -- batch eval        
        for i = 1,n do -- iterate over batches in the split
            -- fetch a batch
            local x, y, x_char = loader:next_batch(split_idx)
            if opt.gpuid >= 0 then -- ship the input arrays to GPU
                -- have to convert to float because integers can't be cuda()'d
                x = x:float():cuda()
                y = y:float():cuda()
                x_char = x_char:float():cuda()
            end
            -- forward pass
            for t=1,opt.seq_length do
                local lst = charcnn:forward(x_char[{{},t}])
                prediction = lst
                loss = loss + criterion:forward(prediction, y[{{}, t}])
            end
        end
        loss = loss / opt.seq_length / n
    else -- full eval on test set
        local token_perp = torch.zeros(#loader.idx2word, 2) 
        local x, y, x_char = loader:next_batch(split_idx)
        if opt.gpuid >= 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x = x:float():cuda()
            y = y:float():cuda()
            x_char = x_char:float():cuda()
        end
        for t = 1, x:size(2) do
            local lst = charcnn:forward(x_char[{{},t}])
            prediction = lst
            local tok_perp
            tok_perp = criterion:forward(prediction, y[{{},t}])
            loss = loss + tok_perp
            token_perp[y[1][t]][1] = token_perp[y[1][t]][1] + 1 --count
            token_perp[y[1][t]][2] = token_perp[y[1][t]][2] + tok_perp
        end
        loss = loss / x:size(2)
    end    
    local perp = torch.exp(loss)    
    return perp, token_perp
end

-- do fwd/bwd and return loss, grad_params
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()
    if hsm > 0 then
        hsm_grad_params:zero()
    end
    ------------------ get minibatch -------------------
    local x, y, x_char = loader:next_batch(1) --from train
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
        x_char = x_char:float():cuda()
    end
    ------------------- forward pass -------------------
    local predictions = {}           -- softmax outputs
    local loss = 0
    for t=1,opt.seq_length do
        local lst = charcnn:forward(x_char[{{},t}])
        predictions[t] = lst -- last element is the prediction
        loss = loss + criterion:forward(predictions[t], y[{{}, t}])
    end
    loss = loss / opt.seq_length
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = criterion:backward(predictions[t], y[{{}, t}])
        local dlst = charcnn:backward(x_char[{{},t}], doutput_t)
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    -- renormalize gradients
    local grad_norm, shrink_factor
    grad_norm = torch.sqrt(grad_params:norm()^2 + hsm_grad_params:norm()^2)
    if grad_norm > opt.max_grad_norm then
        shrink_factor = opt.max_grad_norm / grad_norm
        grad_params:mul(shrink_factor)
        hsm_grad_params:mul(shrink_factor)
    end    
    params:add(grad_params:mul(-lr)) -- update params
    hsm_params:add(hsm_grad_params:mul(-lr))
    return torch.exp(loss)
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
    
    train_loss = feval(params) -- fwd/backprop and update params
    if char_vecs ~= nil then -- zero-padding vector is always zero
        char_vecs.weight[1]:zero() 
        char_vecs.gradWeight[1]:zero()
    end 
    train_losses[i] = train_loss

    -- every now and then or on last iteration
    if i % loader.split_sizes[1] == 0 then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[#val_losses+1] = val_loss
        local savefile = string.format('%s/lm_%s_epoch%.2f_%.2f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = {loader.idx2word, loader.word2idx, loader.idx2char, loader.char2idx}
        checkpoint.lr = lr
        print('saving checkpoint to ' .. savefile)
        if epoch == opt.max_epochs or epoch % opt.save_every == 0 then
            torch.save(savefile, checkpoint)
        end
    end

    -- decay learning rate after epoch
    if i % loader.split_sizes[1] == 0 and #val_losses > 2 then
        if val_losses[#val_losses-1] - val_losses[#val_losses] < opt.decay_when then
            lr = lr * opt.learning_rate_decay
        end
    end    

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.2f), train_loss = %6.4f", i, iterations, epoch, train_loss))
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

test_perp, token_perp = eval_split(3)
print('Perplexity on test set: ' .. test_perp)
torch.save('token_perp-ss.t7', {token_perp, loader.idx2word})

