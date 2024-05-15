import torch
import torch.nn as nn
from torch.nn import Module, ModuleList, Sequential, Linear, LeakyReLU, LayerNorm, Tanh
from models.transformer_layers import TransformerDecoder, TransformerDecoderLayer


class TorsionalAnglesTransformerDecoder(Module):

	def __init__(self, embed_dim=640, hidden_dim=256, num_heads=4, num_layers=3, dropout=0.2, final_pred_dim=18):
		super().__init__()

		decoder_layer = TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
		self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

		self.final_prediction = Sequential(
			Linear(embed_dim, hidden_dim),
			LeakyReLU(),
			Linear(hidden_dim, final_pred_dim),
            Tanh()
			)

	def forward(self, rna_fm_embeddings, padding_mask, initial_embeddings, return_attn=False):

		# Have the mask incorporate the start and end tokens as padded objects   
		#if padding_mask.shape[0] != 1:
		decoder_padding_mask = torch.cat([torch.ones((padding_mask.shape[0],1), dtype=torch.bool, device=padding_mask.device),
                                          padding_mask,
                                         torch.ones((padding_mask.shape[0],1), dtype=torch.bool, device=padding_mask.device)],
                                        -1)


		decoder_output, all_attn_output = self.transformer_decoder(initial_embeddings, rna_fm_embeddings,
                                                                      tgt_key_padding_mask=decoder_padding_mask,
                                                                      memory_key_padding_mask=decoder_padding_mask)
		decoder_output = decoder_output[:,1:-1,:]	# ignore the decoder output after the start and at the end
		final_predictions = self.final_prediction(decoder_output) * (~padding_mask).unsqueeze(-1)	# Make the predictions of the padded values 0

		# print(f"final_predictions: {final_predictions.shape}")
		# print(final_predictions)
		if return_attn:
			return final_predictions, all_attn_output

		return final_predictions
