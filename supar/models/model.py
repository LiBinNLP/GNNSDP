# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from supar.modules import (CharLSTM, ELMoEmbedding, IndependentDropout,
                           SharedDropout, TransformerEmbedding,
                           VariationalLSTM)
from supar.modules.anchor import AnchorGCN
from supar.modules.gnn import GCN, GAT, GraphSAGE
from supar.utils import Config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Model(nn.Module):

    def __init__(self,
                 n_words,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 n_labels=None,
                 encoder='lstm',
                 feat=['char'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 char_dropout=0,
                 elmo_bos_eos=(True, True),
                 elmo_dropout=0.5,
                 bert='/mnt/sda1_hd/atur/libin/projects/BERT/en_bert_base_uncased/',
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=False,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 pad_index=0,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())

        if encoder != 'bert':
            self.word_embed = nn.Embedding(num_embeddings=n_words,
                                           embedding_dim=n_embed)

            n_input = n_embed
            if n_pretrained != n_embed:
                n_input += n_pretrained
            if 'tag' in feat:
                self.tag_embed = nn.Embedding(num_embeddings=n_tags,
                                              embedding_dim=n_feat_embed)
                n_input += n_feat_embed
            if 'char' in feat:
                self.char_embed = CharLSTM(n_chars=n_chars,
                                           n_embed=n_char_embed,
                                           n_hidden=n_char_hidden,
                                           n_out=n_feat_embed,
                                           pad_index=char_pad_index,
                                           dropout=char_dropout)
                n_input += n_feat_embed
            if 'lemma' in feat:
                self.lemma_embed = nn.Embedding(num_embeddings=n_lemmas,
                                                embedding_dim=n_feat_embed)
                n_input += n_feat_embed
            if 'elmo' in feat:
                self.elmo_embed = ELMoEmbedding(n_out=n_feat_embed,
                                                bos_eos=elmo_bos_eos,
                                                dropout=elmo_dropout,
                                                requires_grad=(not freeze))
                n_input += self.elmo_embed.n_out
            if 'bert' in feat:
                self.bert_embed = TransformerEmbedding(model=bert,
                                                       n_layers=n_bert_layers,
                                                       n_out=n_feat_embed,
                                                       pooling=bert_pooling,
                                                       pad_index=bert_pad_index,
                                                       dropout=mix_dropout,
                                                       requires_grad=(not freeze))
                n_input += self.bert_embed.n_out
            self.embed_dropout = IndependentDropout(p=embed_dropout)

        if encoder == 'lstm':
            self.encoder = VariationalLSTM(input_size=n_input,
                                           hidden_size=n_lstm_hidden,
                                           num_layers=n_lstm_layers,
                                           bidirectional=True,
                                           dropout=encoder_dropout)
            self.encoder_dropout = SharedDropout(p=encoder_dropout)
            self.args.n_hidden = n_lstm_hidden * 2
        else:
            self.encoder = TransformerEmbedding(model=bert,
                                                n_layers=n_bert_layers,
                                                pooling=bert_pooling,
                                                pad_index=pad_index,
                                                dropout=mix_dropout,
                                                requires_grad=True)
            self.encoder_dropout = nn.Dropout(p=encoder_dropout)
            self.args.n_hidden = self.encoder.n_out

        # define gnn module
        if self.args.scalable_run:
            gnn_module = AnchorGCN
        elif self.args.gnn_module == 'GCN':
            gnn_module = GCN(
                nfeat=n_embed,
                nhid=n_embed,
                nclass=n_embed,
                graph_hops=self.args.graph_hops,
                dropout=self.args.encoder_dropout,
                batch_norm=self.args.graph_batch_norm)
        elif self.args.gnn_module == 'GAT':
            gnn_module = GAT(
                nfeat=n_embed,
                nhid=n_embed,
                nclass=n_embed,
                dropout=self.args.encoder_dropout,
                nheads=self.args.gat_nhead,
                alpha=self.args.gat_alpha)
        elif self.args.gnn_module == 'GraphSAGE':
            gnn_module = GraphSAGE(
                in_feats=n_embed,
                n_hidden=n_embed,
                n_classes=n_embed,
                n_layers=self.args.graph_hops,
                dropout=self.args.encoder_dropout,
                activation=F.relu,
                aggregator_type=self.args.graphsage_agg_type
            )
        else:
            raise Exception("unknown gnn_module")
        self.graph_encoder = gnn_module

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed.to(self.args.device))
            if embed.shape[1] != self.args.n_pretrained:
                self.embed_proj = nn.Linear(embed.shape[1], self.args.n_pretrained).to(self.args.device)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def forward(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def word_embedding(self, words):
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.args.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        return word_embed

    def feat_embedding(self, words, feats, word_embed):
        if hasattr(self, 'pretrained'):
            pretrained = self.pretrained(words)
            if self.args.n_embed == self.args.n_pretrained:
                word_embed += pretrained
            else:
                word_embed = torch.cat((word_embed, self.embed_proj(pretrained)), -1)

        feat_embeds = []
        if 'tag' in self.args.feat:
            feat_embeds.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embeds.append(self.char_embed(feats.pop(0)))
        if 'elmo' in self.args.feat:
            feat_embeds.append(self.elmo_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embeds.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embeds.append(self.lemma_embed(feats.pop(0)))
        word_embed, feat_embed = self.embed_dropout(word_embed, torch.cat(feat_embeds, -1))
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        return embed

    def embed(self, words, feats):
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.args.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            pretrained = self.pretrained(words)
            if self.args.n_embed == self.args.n_pretrained:
                word_embed += pretrained
            else:
                word_embed = torch.cat((word_embed, self.embed_proj(pretrained)), -1)

        feat_embeds = []
        if 'tag' in self.args.feat:
            feat_embeds.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embeds.append(self.char_embed(feats.pop(0)))
        if 'elmo' in self.args.feat:
            feat_embeds.append(self.elmo_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embeds.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embeds.append(self.lemma_embed(feats.pop(0)))
        word_embed, feat_embed = self.embed_dropout(word_embed, torch.cat(feat_embeds, -1))
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        return embed

    def gnn_encode(self, node_embedding, adj):
        if self.args.gnn_module == 'GCN':
            node_embedding = self.graph_encoder(node_embedding, adj)
        elif self.args.gnn_module == 'GAT':
            batch_size = node_embedding.shape[0]
            embedding_list = []
            for idx_sample in range(batch_size):
                sample_this = torch.index_select(node_embedding, dim=0, index=torch.tensor([idx_sample], device=self.args.device)).squeeze()
                adj_this = torch.index_select(adj, dim=0, index=torch.tensor([idx_sample], device=self.args.device)).squeeze()
                node_vec = self.graph_encoder(sample_this, adj_this)
                embedding_this = F.log_softmax(node_vec, dim=-1)
                embedding_list.append(embedding_this)
            node_embedding = torch.stack(embedding_list, 0).to(self.args.device)
            node_embedding = F.log_softmax(node_embedding, dim=-1)

        elif self.args.gnn_module == 'GraphSAGE':
            import dgl
            from scipy import sparse
            binarized_adj = sparse.coo_matrix(adj.detach().cpu().numpy() != 0)
            dgl_graph = dgl.DGLGraph(binarized_adj)
            dgl_graph = dgl_graph.to(self.args.device)
            node_vec = self.graph_encoder(dgl_graph, node_embedding)
            node_embedding = F.log_softmax(node_vec, dim=-1)

        # node_vec = torch.relu(self.graph_encoder.graph_encoders[0](node_embedding, adj))
        # node_vec = F.dropout(node_vec, self.args.encoder_dropout, training=(self.args.mode == 'train'))
        # # Add mid GNN layers
        # for encoder in self.graph_encoder.graph_encoders[1:-1]:
        #     node_vec = torch.relu(encoder(node_vec, adj))
        #     node_vec = F.dropout(node_vec, self.args.encoder_dropout, training=(self.args.mode == 'train'))
        # # 用graph_encoder将node_vec映射到输出空间output
        # node_embedding = self.graph_encoder.graph_encoders[-1](node_vec, adj)

        return node_embedding

    # 可用版本
    def encode(self, words, feats=None, adj=None):
        word_embedding = self.word_embedding(words)
        node_embedding = self.gnn_encode(word_embedding, adj)

        word_feat_embed = self.feat_embedding(words, feats, node_embedding)
        if self.args.encoder == 'lstm':
            x = pack_padded_sequence(word_feat_embed, words.ne(self.args.pad_index).sum(1).tolist(), True, False)
            x, _ = self.encoder(x)
            x, _ = pad_packed_sequence(x, True, total_length=words.shape[1])
        else:
            x = self.encoder(words)
        return self.encoder_dropout(x)

    # # 尝试版本
    # def encode(self, words, feats=None, adj=None):
    #     word_embedding = self.word_embedding(words)
    #     node_embedding = self.gnn_encode(word_embedding, adj)
    #
    #     word_feat_embed = self.word_feat_embedding(words, feats, node_embedding)
    #     if self.args.encoder == 'lstm':
    #         x = pack_padded_sequence(word_feat_embed, words.ne(self.args.pad_index).sum(1).tolist(), True, False)
    #         x, _ = self.encoder(x)
    #         x, _ = pad_packed_sequence(x, True, total_length=words.shape[1])
    #     else:
    #         x = self.encoder(words)
    #
    #     word_embed, feat_embed = self.embed_dropout(node_embedding, x)
    #     # concatenate the word and feat representations
    #     embed = torch.cat((word_embed, feat_embed), -1)
    #     return embed

    # 梯度不稳定
    # def encode(self, words, feats=None, adj=None):
    #     word_feat_embedding = self.embed(words, feats)
    #     node_embedding = self.gnn_encode(word_feat_embedding, adj)
    #
    #     if self.args.encoder == 'lstm':
    #         x = pack_padded_sequence(node_embedding, words.ne(self.args.pad_index).sum(1).tolist(), True, False)
    #         x, _ = self.encoder(x)
    #         x, _ = pad_packed_sequence(x, True, total_length=words.shape[1])
    #     else:
    #         x = self.encoder(words)
    #     return self.encoder_dropout(x)


    def decode(self):
        raise NotImplementedError
