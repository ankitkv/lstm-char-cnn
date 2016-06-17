
-- Modified from https://github.com/karpathy/char-rnn
-- This version is for cases where one has already segmented train/val/test splits

local BatchLoaderUnk = {}
local stringx = require('pl.stringx')
BatchLoaderUnk.__index = BatchLoaderUnk
utf8 = require 'lua-utf8'

function BatchLoaderUnk.create(data_dir, context_size, batch_size, max_word_l, alpha, table_size)
    local self = {}
    setmetatable(self, BatchLoaderUnk)

    local train_file = path.join(data_dir, 'train.txt')
    local valid_file = path.join(data_dir, 'valid.txt')
    local test_file = path.join(data_dir, 'test.txt')
    local input_files = {train_file, valid_file, test_file}
    local vocab_file = path.join(data_dir, 'vocab.t7')
    local tensor_file = path.join(data_dir, 'data.t7')
    local char_file = path.join(data_dir, 'data_char.t7')

    -- construct a tensor with all the data
    if not (path.exists(vocab_file) or path.exists(tensor_file) or path.exists(char_file)) then
        print('one-time setup: preprocessing input train/valid/test files in dir: ' .. data_dir)
        BatchLoaderUnk.text_to_tensor(input_files, vocab_file, tensor_file, char_file, max_word_l, alpha, table_size)
    end

    print('loading data files...')
    local all_data = torch.load(tensor_file) -- train, valid, test tensors
    local all_data_char = torch.load(char_file) -- train, valid, test character indices
    local vocab_mapping = torch.load(vocab_file)
    self.idx2word, self.word2idx, self.idx2freq, self.table, self.idx2char, self.char2idx = table.unpack(vocab_mapping)
    self.vocab_size = #self.idx2word
    print(string.format('Word vocab size: %d, Char vocab size: %d', #self.idx2word, #self.idx2char))
    -- create word-char mappings
    self.max_word_l = all_data_char[1]:size(2)
    -- cut off the end for train/valid sets so that it divides evenly
    -- test set is not cut off
    self.batch_size = batch_size
    self.split_sizes = {}
    self.all_batches = {}
    print('reshaping tensors...')
    local x_batches, y_batches, nbatches
    for split, data in ipairs(all_data) do
       local len = data:size(1)
       if len % batch_size ~= 0 then
          data = data:sub(1, batch_size * math.floor(len / batch_size))
       end
       local all_ydata = {}
       for i=1,context_size do
           local ydata = data:clone()
           ydata:sub(1,-i-1):copy(data:sub(i+1,-1))
           for j=1,i do
               ydata[j-i-1] = 0
           end
           all_ydata[i] = ydata
       end
       for i=1,context_size do
           local ydata = data:clone()
           ydata:sub(i+1,-1):copy(data:sub(1,-i-1))
           for j=1,i do
               ydata[j] = 0
           end
           all_ydata[context_size+i] = ydata
       end
       local data_char = torch.zeros(data:size(1), self.max_word_l):long()
       for i = 1, data:size(1) do
          data_char[i] = all_data_char[split][i]
       end
       x_batches = data:view(batch_size, -1):split(1, 2)
       local all_y_batches = {}
       for context, ydata in ipairs(all_ydata) do
           all_y_batches[context] = ydata:view(batch_size, -1):split(1, 2)
       end
       x_char_batches = data_char:view(batch_size, -1, self.max_word_l):split(1,2)
       nbatches = #x_batches
       self.split_sizes[split] = nbatches
       assert(#x_batches == #all_y_batches[1])
       assert(#x_batches == #x_char_batches)
       self.all_batches[split] = {x_batches, all_y_batches, x_char_batches}
    end
    self.batch_idx = {0,0,0}
    print(string.format('data load done. Number of batches in train: %d, val: %d, test: %d',
          self.split_sizes[1], self.split_sizes[2], self.split_sizes[3]))
    collectgarbage()
    return self
end

function BatchLoaderUnk.build_table(alpha, table_size, idx2freq)
    local start = sys.clock()
    local total_count_pow = 0
    print("Building a table of unigram frequencies... ")
    for _, count in pairs(idx2freq) do
            total_count_pow = total_count_pow + count^alpha
    end
    local table = torch.IntTensor(table_size)
    local word_index = 1
    local word_prob = idx2freq[word_index]^alpha / total_count_pow
    for idx = 1, table_size do
        table[idx] = word_index
        if idx / table_size > word_prob then
            word_index = word_index + 1
            if word_index <= #idx2freq then
                word_prob = word_prob + idx2freq[word_index]^alpha / total_count_pow
            end
        end
    end
    print(string.format("Done in %.2f seconds.", sys.clock() - start))
    return table
end

function BatchLoaderUnk:reset_batch_pointer(split_idx, batch_idx)
    batch_idx = batch_idx or 0
    self.batch_idx[split_idx] = batch_idx
end

function BatchLoaderUnk:next_batch(split_idx)
    -- split_idx is integer: 1 = train, 2 = val, 3 = test
    self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
    if self.batch_idx[split_idx] > self.split_sizes[split_idx] then
        self.batch_idx[split_idx] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local idx = self.batch_idx[split_idx]
    ydata = self.all_batches[split_idx][2]
    ysplit = {}
    for context, data in ipairs(ydata) do
        ysplit[context] = data[idx]
    end
    return self.all_batches[split_idx][1][idx], ysplit, self.all_batches[split_idx][3][idx]
end

function BatchLoaderUnk.text_to_tensor(input_files, out_vocabfile, out_tensorfile, out_charfile, max_word_l, alpha, table_size)
    print('Processing text into tensors...')
    local tokens = opt.tokens -- inherit global constants for tokens
    local f, rawdata
    local output_tensors = {} -- output tensors for train/val/test
    local output_chars = {} -- output character tensors for train/val/test sets
    local vocab_count = {} -- vocab count
    local max_word_l_tmp = 0 -- max word length of the corpus
    local idx2word = {tokens.UNK} -- unknown word token
    local idx2freq = {0}
    local word2idx = {}; word2idx[tokens.UNK] = 1
    local idx2char = {tokens.ZEROPAD, tokens.START, tokens.END} -- zero-pad, start-of-word, end-of-word tokens
    local char2idx = {}; char2idx[tokens.ZEROPAD] = 1; char2idx[tokens.START] = 2; char2idx[tokens.END] = 3
    local split_counts = {}

    -- first go through train/valid/test to get max word length
    -- if actual max word length is smaller than specified
    -- we use that instead. this is inefficient, but only a one-off thing so should be fine
    -- also counts the number of tokens
    for split = 1,3 do -- split = 1 (train), 2 (val), or 3 (test)
       f = io.open(input_files[split], 'r')
       local counts = 0
       for line in f:lines() do
          line = stringx.replace(line, '<unk>', tokens.UNK) -- replace unk with a single character
          line = stringx.replace(line, tokens.START, '') --start-of-word token is reserved
          line = stringx.replace(line, tokens.END, '') --end-of-word token is reserved
          for word in line:gmatch'([^%s]+)' do
             max_word_l_tmp = math.max(max_word_l_tmp, utf8.len(word) + 2) -- add 2 for start/end chars
             counts = counts + 1
          end
          if tokens.EOS ~= '' then
              counts = counts + 1 --PTB uses \n for <eos>, so need to add one more token at the end
          end
       end
       f:close()
       split_counts[split] = counts
    end

    print('After first pass of data, max word length is: ' .. max_word_l_tmp)
    print(string.format('Token count: train %d, val %d, test %d',
                            split_counts[1], split_counts[2], split_counts[3]))

    -- if actual max word length is less than the limit, use that
    max_word_l = math.min(max_word_l_tmp, max_word_l)

    for split = 1,3 do -- split = 1 (train), 2 (val), or 3 (test)
       -- Preallocate the tensors we will need.
       -- Watch out the second one needs a lot of RAM.
       output_tensors[split] = torch.LongTensor(split_counts[split])
       output_chars[split] = torch.ones(split_counts[split], max_word_l):long()

       f = io.open(input_files[split], 'r')
       local word_num = 0
       for line in f:lines() do
          line = stringx.replace(line, '<unk>', tokens.UNK)
          line = stringx.replace(line, tokens.START, '') -- start and end of word tokens are reserved
          line = stringx.replace(line, tokens.END, '')
          for rword in line:gmatch'([^%s]+)' do
             function append(word)
                word_num = word_num + 1
                -- Collect garbage.
                if word_num % 10000 == 0 then
                   collectgarbage()
                end
                local chars = {char2idx[tokens.START]} -- start-of-word symbol
                if string.sub(word,1,1) == tokens.UNK and word:len() > 1 then -- unk token with character info available
                   word = string.sub(word, 3)
                   output_tensors[split][word_num] = word2idx[tokens.UNK]
                   idx2freq[1] = idx2freq[1] + 1
                else
                   if word2idx[word]==nil then
                      idx2word[#idx2word + 1] = word -- create word-idx/idx-word mappings
                      word2idx[word] = #idx2word
                      idx2freq[#idx2word] = 0
                   end
                   index = word2idx[word]
                   idx2freq[index] = idx2freq[index] + 1
                   output_tensors[split][word_num] = index
                end
                local l = utf8.len(word)
                for _, char in utf8.next, word do
                   char = utf8.char(char) -- save as actual characters
                   if char2idx[char]==nil then
                      idx2char[#idx2char + 1] = char -- create char-idx/idx-char mappings
                      char2idx[char] = #idx2char
                   end
                   chars[#chars + 1] = char2idx[char]
                end
                chars[#chars + 1] = char2idx[tokens.END] -- end-of-word symbol
                for i = 1, math.min(#chars, max_word_l) do
                   output_chars[split][word_num][i] = chars[i]
                end
                if #chars == max_word_l then
                    chars[#chars] = char2idx[tokens.END]
                end
             end
             append(rword)
          end
          if tokens.EOS ~= '' then --PTB does not have <eos> so we add a character for <eos> tokens
              append(tokens.EOS)   --other datasets don't need this
          end
       end
    end
    local table = BatchLoaderUnk.build_table(alpha, table_size, idx2freq)
    print "done"
    -- save output preprocessed files
    print('saving ' .. out_vocabfile)
    torch.save(out_vocabfile, {idx2word, word2idx, idx2freq, table, idx2char, char2idx})
    print('saving ' .. out_tensorfile)
    torch.save(out_tensorfile, output_tensors)
    print('saving ' .. out_charfile)
    torch.save(out_charfile, output_chars)
end

return BatchLoaderUnk

