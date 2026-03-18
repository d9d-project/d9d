import torch


def build_decoder_inputs_hf():
    torch.manual_seed(428)

    input_ids = torch.randint(size=(8, 129), low=0, high=100 - 1, dtype=torch.long, device="cuda")

    labels = input_ids.clone()
    labels[0, :119] = -100
    labels[0, :100] = -100

    position_ids = torch.arange(0, 129, dtype=torch.long, device="cuda")[None, :].repeat(8, 1)

    return input_ids, position_ids, labels


def build_decoder_inputs_my():
    input_ids, position_ids, labels = build_decoder_inputs_hf()

    # shift
    input_ids = input_ids[:, :-1]
    position_ids = position_ids[:, :-1]
    labels = labels[:, 1:]

    return input_ids, position_ids, labels
