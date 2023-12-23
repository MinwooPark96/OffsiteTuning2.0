import torch
from torch import nn


class AE_auto_layer(nn.Module): #dropout, linear init 추가
    def __init__(self, **kwargs):
        super(AE_auto_layer, self).__init__()

        dims = [kwargs[key] for key in kwargs] #key,value
        
        #[768,256,128,256,768]
        
        self.hidden_layers = nn.ModuleList()
        
        for idx in range(len(dims) - 1):
            self.hidden_layers.append(nn.Linear(dims[idx], dims[idx+1]))
        
        # for linears in self.hidden_layers:
        #     torch.nn.init.xavier_uniform_(linears.weight)
        
        
        
        self.layer_norm = nn.LayerNorm(dims[-1], eps=1e-05)
        # self.dropout = torch.nn.Dropout(p=0.3)
        #self.encoder = nn.Linear(
        #    in_features=kwargs["input_dim"], out_features=kwargs["compress_dim"]
        #)

        # mean-squared error loss
        self.criterion = nn.CrossEntropyLoss()
        self.activation = nn.LeakyReLU()

    #def encoding(self, features):
    #    return self.encoder(features)

    def forward(self, features):
        '''
        encoded_emb = self.encoding(features)
        encoded_emb = torch.relu(encoded_emb)
        return encoded_emb
        '''
        for layer in self.hidden_layers:
            features = self.activation(layer(features))

        features = self.layer_norm(features)
        
        return features



class AE_0_layer_mutiple_100(nn.Module):
    def __init__(self, **kwargs):
        super(AE_0_layer_mutiple_100, self).__init__()
        self.encoder = nn.Linear(
            in_features=kwargs["dim_0"], out_features=kwargs["dim_1"]
        )

        # mean-squared error loss
        self.criterion = nn.CrossEntropyLoss()
        self.activation = nn.LeakyReLU()

    def encoding(self, features):
        return self.encoder(features)

    def forward(self, features):
        encoded_emb = self.encoding(features)
        encoded_emb = self.activation(encoded_emb)
        return encoded_emb


class AE_0_layer(nn.Module):
    def __init__(self, **kwargs):
        super(AE_0_layer, self).__init__()
        self.encoder = nn.Linear(
            in_features=kwargs["dim_0"], out_features=kwargs["dim_1"]
        )

        # mean-squared error loss
        self.criterion = nn.CrossEntropyLoss()
        self.activation = nn.LeakyReLU()

    def encoding(self, features):
        return self.encoder(features)

    def forward(self, features):
        encoded_emb = self.encoding(features)
        encoded_emb = self.activation(encoded_emb)
        return encoded_emb




class AE_1_layer_mutiple_100(nn.Module):
    def __init__(self, **kwargs):
        super(AE_1_layer_mutiple_100, self).__init__()
        self.encoder = nn.Linear(
            in_features=kwargs["dim_0"], out_features=int(kwargs["dim_1"])
        )
        self.decoder = nn.Linear(
            in_features=int(kwargs["dim_1"]), out_features=kwargs["dim_2"]
        )

        self.dim = int(kwargs["dim_2"]/100)
        #########################
        self.layer_norm = nn.LayerNorm(self.dim)
        #########################

        # mean-squared error loss
        self.criterion = nn.CrossEntropyLoss()
        self.activation = nn.LeakyReLU()
        #self.activation = nn.Tanh()
        #self.activation = nn.Softmax(dim=-1)
        #self.activation = nn.Softmax(dim=0)

    def encoding(self, features):
        return self.encoder(features)
    def decoding(self, features):
        return self.decoder(features)
    
    def forward(self, features):
        encoded_emb = self.encoding(features)
        encoded_emb = self.activation(encoded_emb)
        decoded_emb = self.decoding(encoded_emb)
        decoded_emb = decoded_emb.reshape(int(decoded_emb.shape[0]),100,self.dim)
        decoded_emb = self.layer_norm(decoded_emb)
        decoded_emb = decoded_emb.reshape(int(decoded_emb.shape[0]),100*self.dim)
        ###
        return decoded_emb



class AE_1_layer_mutiple_100_paper(nn.Module):
    def __init__(self, **kwargs):
        super(AE_1_layer_mutiple_100_paper, self).__init__()
        self.encoder = nn.Linear(
            in_features=int(kwargs["dim_0"]), out_features=int(kwargs["dim_1"])
        )
        self.decoder = nn.Linear(
            in_features=int(kwargs["dim_1"]), out_features=int(kwargs["dim_2"])
        )

        self.dim = int(kwargs["dim_2"]/100)

        #########################
        self.layer_norm = nn.LayerNorm(self.dim, eps=1e-05)
        #########################

        # mean-squared error loss
        self.criterion = nn.CrossEntropyLoss()
        self.activation = nn.Tanh()
        #self.activation = nn.LeakyReLU()

    def encoding(self, features):
        return self.encoder(features)
    def decoding(self, features):
        return self.decoder(features)

    def forward(self, features):
        encoded_emb = self.encoding(features)
        encoded_emb = self.activation(encoded_emb)
        decoded_emb = self.decoding(encoded_emb)
        #layer_norm = nn.LayerNorm(int(decoded_emb.shape[0]),100,768)
        layer_norm = nn.LayerNorm(int(decoded_emb.shape[0]),100,self.dim)
        decoded_emb = decoded_emb.reshape(int(decoded_emb.shape[0]),100,self.dim)
        decoded_emb = self.layer_norm(decoded_emb)
        decoded_emb = decoded_emb.reshape(int(decoded_emb.shape[0]),100*self.dim)
        return decoded_emb






class AE_1_layer(nn.Module):
    def __init__(self, **kwargs):
        super(AE_1_layer, self).__init__()
        self.encoder = nn.Linear(
            in_features=kwargs["dim_0"], out_features=int(kwargs["dim_1"])
        )
        self.decoder = nn.Linear(
            in_features=int(kwargs["dim_1"]), out_features=kwargs["dim_2"]
        )

        self.layer_norm = nn.LayerNorm(kwargs['dim_2'], eps=1e-05)
        
        # mean-squared error loss
        self.criterion = nn.CrossEntropyLoss()
        self.activation = nn.LeakyReLU()

    def encoding(self, features):
        return self.encoder(features)
    def decoding(self, features):
        return self.decoder(features)

    def forward(self, features):
        encoded_emb = self.encoding(features)
        encoded_emb = self.activation(encoded_emb)
        decoded_emb = self.decoding(encoded_emb)
        decoded_emb = self.layer_norm(decoded_emb)
        return decoded_emb


class AE_transformer_layer(nn.Module):
    def __init__(self, **kwargs):
        super(AE_transformer_layer, self).__init__()
        
        d_model = 128
        self.transformerEncoderLayer = torch.nn.TransformerEncoderLayer(d_model = d_model, nhead = 8, batch_first = True)
        self.transformerEnocder = torch.nn.TransformerEncoder(self.transformerEncoderLayer,num_layers = 6)

        self.encoder = nn.Linear(
            in_features=int(kwargs["dim_0"]), out_features=d_model)
        
        # self.expander = nn.Linear(in_features=128,out_features=256)
        
        self.decoder = nn.Linear(
            in_features=d_model, out_features=kwargs["dim_1"])    
        
        self.criterion = nn.CrossEntropyLoss()
        self.activation = nn.LeakyReLU()
        
        self.layer_norm = nn.LayerNorm(kwargs['dim_1'], eps=1e-05)
        
        self.init_weights()
    
    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

        
    def forward(self, features):
        
        encoded_emb = self.encoder(features)
        encoded_emb = self.transformerEnocder(encoded_emb)
        decoded_emb = self.decoder(encoded_emb)
        # decoded_emb = self.layer_norm(decoded_emb)
        return decoded_emb


class AE_1_layer_tokenwise(nn.Module):
    
    def __init__(self, **kwargs):
        super(AE_1_layer_tokenwise, self).__init__()
        
        self.encoder_list = nn.ModuleList([nn.Linear(in_features=kwargs["dim_0"], out_features=int(kwargs["dim_1"])) for _ in range(100)])
        self.decoder_list = nn.ModuleList([nn.Linear(in_features=kwargs["dim_1"], out_features=int(kwargs["dim_2"])) for _ in range(100)])
        
        self.activation = nn.LeakyReLU()

        self.activation_list = nn.ModuleList([self.activation for _ in range(100)])
        
        self.encoder = nn.Linear(
            in_features=kwargs["dim_0"], out_features=int(kwargs["dim_1"])
        )
        self.decoder = nn.Linear(
            in_features=int(kwargs["dim_1"]), out_features=kwargs["dim_2"]
        )

        # self.layer_norm = nn.LayerNorm(kwargs['dim_2'], eps=1e-05)
        
        # mean-squared error loss
        
        self.criterion = nn.CrossEntropyLoss()
        
        
    def encoding(self, features):
        return self.encoder(features)
    def decoding(self, features):
        return self.decoder(features)

    def encoding_tokenwise(self,features):
        outputs = [encoder_onetoken(features[:, i, :]) for i, encoder_onetoken in enumerate(self.encoder_list)]
        final_output = torch.stack(outputs, dim=1)
        return final_output
    
    def decoding_tokenwise(self,features):
        outputs = [decoder_onetoken(features[:, i, :]) for i, decoder_onetoken in enumerate(self.decoder_list)]
        final_output = torch.stack(outputs, dim=1)
        return final_output
    
    def activation_tokenwise(self,features):
        outputs = [activation_onetoken(features[:, i, :]) for i, activation_onetoken in enumerate(self.activation_list)]
        final_output = torch.stack(outputs, dim=1)
        return final_output
    
    
    def forward(self, features):
        encoded_emb = self.encoding_tokenwise(features)
        encoded_emb = self.activation_tokenwise(encoded_emb)
        decoded_emb = self.decoding_tokenwise(encoded_emb)
        #decoded_emb = self.layer_norm(decoded_emb)
        return decoded_emb



    
# import torch.nn.functional as F
# import math    
# class AE_attention(nn.Module):
#     def __init__(self, **kwargs):
#         super(AE_attention, self).__init__()
#         self.input_dim = kwargs["dim_0"]
#         self.output_dim = kwargs["dim_1"]
        
#         self.head_num = 8
#         self.activation = nn.LeakyReLU()
#         self.bias = True
#         self.hidden_dim = 256
        
#         self.linear_q = nn.Linear(self.input_dim, self.hidden_dim, self.bias)
#         self.linear_k = nn.Linear(self.input_dim, self.hidden_dim, self.bias)
#         self.linear_v = nn.Linear(self.input_dim, self.hidden_dim, self.bias)
#         self.linear_o = nn.Linear(self.hidden_dim, self.output_dim, self.bias)
    
#     def gen_history_mask(x):
#         """Generate the mask that only uses history data.
#         :param x: Input tensor.
#         :return: The mask.
#         """
#         batch_size, seq_len, _ = x.size()
#         return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

#     def _reshape_to_batches(self, x):
#         batch_size, seq_len, in_feature = x.size()
#         sub_dim = in_feature // self.head_num
#         return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
#                 .permute(0, 2, 1, 3)\
#                 .reshape(batch_size * self.head_num, seq_len, sub_dim)

#     def _reshape_from_batches(self, x):
#         batch_size, seq_len, in_feature = x.size()
#         batch_size //= self.head_num
#         out_dim = in_feature * self.head_num
#         return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
#                 .permute(0, 2, 1, 3)\
#                 .reshape(batch_size, seq_len, out_dim)

#     def extra_repr(self):
#         return 'input_dim={}, head_num={}, bias={}, activation={}'.format(
#             self.input_dim, self.head_num, self.bias, self.activation,
#         )
#     def forward(self,features,mask=None):
#         q, k, v = self.linear_q(features), self.linear_k(features), self.linear_v(features)
        
#         if self.activation is not None:
#             q = self.activation(q)
#             k = self.activation(k)
#             v = self.activation(v)

#         q = self._reshape_to_batches(q)
#         k = self._reshape_to_batches(k)
#         v = self._reshape_to_batches(v)
#         if mask is not None:
#             mask = mask.repeat(self.head_num, 1, 1)
        
#         dk = q.size()[-1]
#         scores = q.matmul(k.transpose(-2, -1)) / math.sqrt(dk)
        
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
        
#         attention = F.softmax(scores, dim=-1)
#         y= attention.matmul(v)
#         y = self._reshape_from_batches(y)

#         y = self.linear_o(y)
#         y = self.activation(y)
#         return y
