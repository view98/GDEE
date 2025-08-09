import torch
import torch.nn as nn
import itertools
import gc

torch.set_printoptions(profile="full")

class GDEE(nn.Module):
    def __init__(self,args,word_type_tag_num):
        super(GDEE, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)

        num_embeddings, embed_dim = args.embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(args.embedding,requires_grad=False)
        self.word_type_embed = nn.Embedding(word_type_tag_num, args.word_type_embedding_dim)

        in_dim = args.token_embedding_dim+args.word_type_embedding_dim

        self.token_bilstm = nn.LSTM(input_size=in_dim, hidden_size=args.hidden_size,
                              bidirectional=True, batch_first=True, num_layers=args.num_layers)


        last_hidden_size = 4*args.hidden_size
        layers_trigger = [nn.Linear(last_hidden_size, args.final_hidden_size), nn.LeakyReLU()]
        for _ in range(args.num_mlps - 1):
            layers_trigger += [nn.Linear(args.final_hidden_size, args.final_hidden_size), nn.LeakyReLU()]
        self.fcs = nn.Sequential(*layers_trigger)
        self.fc_final = nn.Linear(args.final_hidden_size, args.classify_nums)

    def forward(self,token_ids,wType_ids):
        token_feature = self.embed(token_ids)
        token_feature = self.dropout(token_feature)
        token_type_feature = self.word_type_embed(wType_ids)
        token_type_feature = self.dropout(token_type_feature)

        all_token_feature = torch.cat([token_feature, token_type_feature], dim=1)

        token_out_bilstm, _ = self.token_bilstm(all_token_feature.unsqueeze(0))  # (1,T,2D) batch_size=1 seq_len = T
        token_out_bilstm = self.dropout(token_out_bilstm).squeeze(0)  # (T,2D)

        # with torch.no_grad():
        #     ent_ent_list = list(itertools.product(token_out_bilstm, repeat=2))
        #     ent_ent_emb = []
        #     for ent_ent in ent_ent_list:
        #         ent_ent_emb.append(torch.cat([ent_ent[0], ent_ent[1]], dim=0))
        #
        #     ent_ent_feature = torch.stack(ent_ent_emb, dim=0)

        # 实体对分类
        # table_list = []
        # with torch.no_grad():
        repeated_in_chunks = token_out_bilstm.repeat_interleave(token_out_bilstm.shape[0], dim=0)
        repeated_alternating = token_out_bilstm.repeat(token_out_bilstm.shape[0], 1)
        token_pair = torch.cat([repeated_in_chunks, repeated_alternating], dim=1)
            # torch.cuda.empty_cache()

        # for i in range(len(token_ids)):
        #     pair_list = []
        #     for j in range(len(token_ids)):
        #         a = token_out_bilstm[i]
        #         b = token_out_bilstm[j]
        #         embedding= torch.cat([a, b])     # (4D)
        #         pair_list.append(embedding)
        #     pair_tensor = torch.stack(pair_list)  # (T,4D)
        #     table_list.append(pair_tensor)
        # table_tensor = torch.stack(table_list)  # (T,T,4D)


        out = self.fcs(token_pair)
        logit = self.fc_final(out)  # (T,T,120)
        return logit  # (T*T,120)

