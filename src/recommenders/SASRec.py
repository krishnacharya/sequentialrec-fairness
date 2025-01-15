import torch.utils
from src.recommenders.utils.sequencer import Sequencer
from src.recommenders.sequential_model import SequentialModel
from src.recommenders.sequential_recommender import SequentialRecommender, SequentialRecommenderConfig
import torch

class SASRecConfig(SequentialRecommenderConfig):
    def __init__(self, embedding_size=256, num_layers=3, nhead=4, dim_feedforward=1024, dropout=0.5, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout=dropout


class SASRecModel(SequentialModel):
    def __init__(self, num_items, config: SASRecConfig):
        super().__init__()
        self.position_embeddings = torch.nn.Embedding(config.sequence_len, config.embedding_size) # P  
        self.item_embeddings = torch.nn.Embedding(num_items, config.embedding_size) # M
        self.unembeddings = torch.nn.Embedding(num_items, config.embedding_size) # seperate embeddings 
        transformer_encoder_layer = torch.nn.TransformerEncoderLayer(config.embedding_size, config.nhead, \
                                                                     config.dim_feedforward, config.dropout, batch_first=True, norm_first=True) # d_model the first parameter is the dimensionality of the embedding passed thru attention
        self.transformer = torch.nn.TransformerEncoder(transformer_encoder_layer, num_layers=config.num_layers)
        self.num_items = num_items
        self.emb_dropout = torch.nn.Dropout(config.dropout) # apply a dropout after the input embeddings too
        self.output_norm = torch.nn.LayerNorm(config.embedding_size)

    def split_training_batch(self, batch):
        item_ids = batch["sequences"]
        inputs = item_ids[:,:-1]
        labels = item_ids[:,1:].unsqueeze(-1)
        padding_mask = batch["padding_mask"][:,:-1]
        position_ids = batch["position_ids"][:-1] + 1
        return {"sequences": inputs, "labels": labels, "padding_mask": padding_mask, "position_ids": position_ids}

    def forward(self, batch, cbloss=False, cbsmooth=False):
        train_batch = self.split_training_batch(batch) # train_batch["labels"] has shape (B, L, 1)
        transformer_output, embedding_weights = self.hidden_state(train_batch) #transformer_output (B, L, d), emb weights has shape (I, d)
        logits = torch.einsum("bse, ie -> bsi", transformer_output, embedding_weights) # (B,L,I) product along embedding dimension, to get relevance score for each item
        logprobs = torch.log_softmax(logits, -1).gather(2, train_batch["labels"]).squeeze(-1) # (B, L) shape, relevance score for each +ve example at each position, torch.log_softmax(logits, -1) has shape (B,L,I)
        non_masked = 1-train_batch["padding_mask"] #shape (B, L), just to pick the indices that are not padding
        if not cbloss:
            loss = -(logprobs * non_masked).sum(axis=-1)/non_masked.sum(axis=-1)
            return {"loss_per_sample": loss}
        else:
            inv_weight = batch['seqinv_weight'][:,1:]
            if cbsmooth: #log smoothed
                inv_weight = torch.log(inv_weight)
            loss = -(logprobs * non_masked * inv_weight).sum(axis=-1)/non_masked.sum(axis=-1) #1/L_active for each user sequence
            return {"loss_per_sample": loss}


    def hidden_state(self, batch):
        item_embeddings = self.item_embeddings(batch["sequences"])
        position_embeddings = self.position_embeddings(batch["position_ids"]).unsqueeze(0) 
        transformer_input = item_embeddings + position_embeddings
        transformer_input = self.emb_dropout(transformer_input)
        causality_mask = torch.nn.Transformer.generate_square_subsequent_mask(batch["sequences"].shape[1], device=batch["sequences"].device)
        transformer_output = self.transformer(is_causal=True, src_key_padding_mask = batch["padding_mask"], mask=causality_mask,src=transformer_input)
        transformer_output = self.output_norm(transformer_output)

        embedding_weights = self.emb_dropout(self.unembeddings.weight)
        return transformer_output,embedding_weights

    def predict(self, batch):
        transformer_output, embedding_weights = self.hidden_state(batch)
        transformer_output = transformer_output[:,-1, :]# last token prediction B,-1,d
        logits = torch.einsum("be, ie -> bi", transformer_output, embedding_weights) #transformer output has embedding for predicted output, just find score across all items
        return logits



class SASRecRecommender(SequentialRecommender):
    def __init__(self, config: SASRecConfig) -> None:
        super().__init__(config)
        self.config = config
    
    def init_model(self, sequencer: Sequencer) -> SequentialModel:
        return SASRecModel(num_items=len(sequencer.item_mapping), config=self.config).to(self.config.device)