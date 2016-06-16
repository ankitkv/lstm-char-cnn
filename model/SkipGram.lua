local SkipGram = {}

local ok, cunn = pcall(require, 'fbcunn')

function SkipGram.skipgram(word_vocab_size, char_vocab_size, char_vec_size, feature_maps, kernels, length, highway_layers)
    -- word_vocab_size = num words in the vocab
    -- char_vocab_size = num chars in the character vocab
    -- char_vec_size = dimensionality of char embeddings
    -- feature_maps = table of feature map sizes for each kernel width
    -- kernels = table of kernel widths
    -- length = max length of a word
    -- highway_layers = number of highway layers to use, if any

    local char_vec_layer, x, input_size_L, char_vec
    local highway_layers = highway_layers or 0
    local length = length
    local x_char = nn.Identity()() -- batch_size x word length (char indices)
    char_vec_layer = nn.LookupTable(char_vocab_size, char_vec_size)
    char_vec_layer.name = 'char_vecs' -- change name so we can refer to it easily later
    char_vec = char_vec_layer(x_char)
    local char_cnn = TDNN.tdnn(length, char_vec_size, feature_maps, kernels)
    char_cnn.name = 'cnn' -- change name so we can refer to it later
    local cnn_output = char_cnn(char_vec)
    input_size_L = torch.Tensor(feature_maps):sum()
    x = nn.Identity()(cnn_output)
    if highway_layers > 0 then
        local highway_mlp = HighwayMLP.mlp(input_size_L, highway_layers)
        highway_mlp.name = 'highway'
        x = highway_mlp(x)
    end
    return nn.gModule({x_char}, {x})
end

return SkipGram

