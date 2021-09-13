import torch
from torch.nn.utils.rnn import pad_sequence


def tokenize_and_track(tokens, labels, tokenizer, label_map):
    """[summary]

    Args:
        tokens (list[str]): [description]
        labels (list[str]): [description]
        tokenizer (callable): [description]
        label_map (dict): [description]

    Returns:
        dict
    """

    input_ids = []
    first_subtoken_mask = []
    encoded_label = []

    for token, label in zip(tokens, labels):

        sub_tokens = tokenizer.tokenize(token)

        if len(sub_tokens) < 1:
            sub_tokens = [tokenizer.unk_token]

        sub_tokens = tokenizer.convert_tokens_to_ids(sub_tokens)

        num_sub_tokens = len(sub_tokens)

        input_ids.extend(sub_tokens)

        first_subtoken_mask.extend(
            [1] + (num_sub_tokens - 1) * [-1]
        )

        encoded_label.append(label_map[label])

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


def collate_fn(batch_list, tokenizer, label_map):

    batch = [tokenize_and_track(tokens, labels, tokenizer, label_map)
             for (tokens, labels) in batch_list]

    input_ids = pad_sequence([el['input_ids'] for el in batch],
                             batch_first=True, padding_value=tokenizer.pad_token_id)

    attention_mask = (input_ids != tokenizer.pad_token_id).float()

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
