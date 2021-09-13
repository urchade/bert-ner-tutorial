import pytorch_lightning as pl
import pickle
from typing import List

import torch
from allennlp.modules.conditional_random_field import (ConditionalRandomField,
                                                       allowed_transitions)
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from tqdm import tqdm

from seqeval.metrics.sequence_labeling import f1_score, get_entities
# from utils import collate_fn, tokenize_and_track

with open('data/tdm.pkl', 'rb') as f:
    data = pickle.load(f)

label_map = data['tag_to_id']

train_data = data['train']

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")


batch_list = [i.values() for i in train_data[:10]]

# x = collate_fn(batch_list, tokenizer, label_map)


def select_first_subword(hidden_state, first_subtoken_mask, seq_length, padding_value=-1):
    """[summary]

    Args:
        hidden_state ([type]): [description]
        first_subtoken_mask ([type]): [description]
        seq_length ([type]): [description]
        padding_value (int, optional): [description]. Defaults to -1.

    Returns:
        [type]: [description]
    """

    d_model = hidden_state.size(-1)

    first_subtoken_mask = first_subtoken_mask.unsqueeze(-1)

    mask_sel = torch.masked_select(
        hidden_state, first_subtoken_mask).view(-1, d_model)

    valid_seq = torch.split(mask_sel, seq_length)

    padded = pad_sequence(valid_seq, batch_first=True,
                          padding_value=padding_value)

    word_mask = padded.sum(-1) != padding_value * padded.shape[-1]

    return {
        'first_subword': padded,
        'word_mask': word_mask
    }


class BertCRF(nn.Module):

    def __init__(self, label_map, model_name='allenai/scibert_scivocab_uncased', tag_format='BIO'):

        super().__init__()

        self.label_to_id = label_map
        self.id_to_label = {v: k for k, v in label_map.items()}

        constraints_for_crf = allowed_transitions(tag_format, self.id_to_label)

        n_labels = len(self.id_to_label)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.transformer_layer = AutoModel.from_pretrained(model_name)

        self.hidden_size = self.transformer_layer.config.hidden_size

        self.output_layer = nn.Linear(self.hidden_size, n_labels)

        self.conditional_random_field = ConditionalRandomField(
            n_labels, constraints=constraints_for_crf)

    def forward(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """

        hidden_state = self.transformer_layer(
            x['input_ids'], x['attention_mask']).last_hidden_state

        out_sel = select_first_subword(
            hidden_state, x['first_subtoken_mask'], x['seq_length'])

        hidden_state, mask = out_sel.values()

        logits = self.output_layer(hidden_state)

        outputs = {'logits': logits, 'mask': mask}

        if 'y' in x:
            loss = - self.conditional_random_field(logits, x['y'], mask)
            outputs['loss'] = loss

        return outputs

    def predict(self, x, return_loss=False):
        """[summary]

        Args:
            x ([type]): [description]
            return_loss (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """

        outputs = self.forward(x)
        prediction = self.conditional_random_field.viterbi_tags(
            outputs['logits'], outputs['mask'])
        prediction = [self.id_to_IOB(i[0]) for i in prediction]

        if return_loss:
            return prediction, outputs['loss']

        return prediction

    def id_to_IOB(self, sequence):
        """[summary]

        Args:
            sequence ([type]): [description]

        Returns:
            [type]: [description]
        """

        out = []
        for i in sequence:
            out.append(self.id_to_label[i])
        return out

    def predict_from_tokens(self, tokens: List[List[str]], **args):
        """[summary]

        Args:
            tokens (List[List[str]]): [description]

        Returns:
            [type]: [description]
        """

        data = [(t, None) for t in tokens]
        loader = self.create_dataloader(data, **args)
        device = next(self.parameters()).device

        all_preds = []
        for x in tqdm(loader):

            for k, v in x.items():
                if torch.is_tensor(v):
                    x[k] = v.to(device)

            pred = self.predict(x)

            all_preds.extend(pred)

        return all_preds

    def extract_entities(self, tokens: List[List[str]], **args):
        """[summary]

        Args:
            tokens (List[List[str]]): [description]

        Returns:
            [type]: [description]
        """

        predictions = self.predict_from_tokens(tokens, **args)

        entities = []

        for pred, d in zip(predictions, tokens):
            ent_span = []
            pred = get_entities(pred)
            for p in pred:
                ent_type, start, end = p
                ent_span.append(
                    (ent_type, start, end, ' '.join(d[start: end + 1])))
            entities.append(ent_span)

        return entities

    def tokenize_and_track(self, tokens, labels=None):
        """[summary]

        Args:
            tokens ([type]): [description]
            labels ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """

        if labels is None:
            labels = len(tokens) * ['O']

        input_ids = []
        first_subtoken_mask = []
        encoded_label = []

        for token, label in zip(tokens, labels):

            sub_tokens = self.tokenizer.tokenize(token)

            if len(sub_tokens) < 1:
                sub_tokens = [tokenizer.unk_token]

            sub_tokens = self.tokenizer.convert_tokens_to_ids(sub_tokens)

            num_sub_tokens = len(sub_tokens)

            input_ids.extend(sub_tokens)

            first_subtoken_mask.extend(
                [1] + (num_sub_tokens - 1) * [-1]
            )

            encoded_label.append(self.label_to_id[label])

        assert len(first_subtoken_mask) == len(input_ids)

        input_ids, first_subtoken_mask, encoded_label = map(
            torch.LongTensor, [input_ids, first_subtoken_mask, encoded_label])

        return {
            'input_ids': input_ids,
            'first_subtoken_mask': first_subtoken_mask,
            'y': encoded_label,
            'raw_y': labels,
            'seq_length': len(labels)
        }

    def collate_fn(self, batch_list):
        """[summary]

        Args:
            batch_list ([type]): [description]

        Returns:
            [type]: [description]
        """

        batch = [self.tokenize_and_track(tokens, labels)
                 for (tokens, labels) in batch_list]

        input_ids = pad_sequence([el['input_ids'] for el in batch],
                                 batch_first=True, padding_value=self.tokenizer.pad_token_id)

        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()

        y = pad_sequence([el['y'] for el in batch],
                         batch_first=True, padding_value=0)

        seq_length = [el['seq_length'] for el in batch]

        first_subtoken_mask = pad_sequence(
            [el['first_subtoken_mask'] for el in batch], batch_first=True, padding_value=-1) != -1

        raw_y = [el['raw_y'] for el in batch]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'y': y,
            'raw_y': raw_y,
            'first_subtoken_mask': first_subtoken_mask,
            'seq_length': seq_length
        }

    def create_dataloader(self, data, **kwargs):
        """[summary]

        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """
        return DataLoader(data, collate_fn=self.collate_fn, **kwargs)


class LightningWrapper(pl.LightningModule):

    def __init__(self, label_map):

        super().__init__()

        self.model = BertCRF(label_map)

    def training_step(self, batch, batch_idx):

        x = batch
        loss = self.model(x)['loss']
        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        x = val_batch

        pred_base, loss_base = self.model.predict(x, return_loss=True)

        f1_micro_base = f1_score(
            x['raw_y'], pred_base, average='macro')

        self.log('f1_base', f1_micro_base, prog_bar=True)

        self.log('loss_base', loss_base)

    def train_dataloader(self):
        with open('data/tdm.pkl', 'rb') as f:
            train_data = pickle.load(f)['train']

        train_data = [i.values() for i in train_data]

        return self.model.create_dataloader(train_data, batch_size=16, num_workers=10, shuffle=True)

    def val_dataloader(self):
        with open('data/tdm.pkl', 'rb') as f:
            dev_data = pickle.load(f)['dev']

        dev_data = [i.values() for i in dev_data]

        return self.model.create_dataloader(dev_data, batch_size=16, num_workers=10, shuffle=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer

lightning_model = LightningWrapper(label_map)

trainer = pl.Trainer(gpus=1, precision=32, max_epochs=1)

trainer.fit(lightning_model)
