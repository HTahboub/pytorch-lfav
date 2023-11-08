import torch


def create_attention_mask(length, layer_num, device):
    # Determine the size of the small square based on the layer number
    small_square_size = 2 ** layer_num
    
    # Create a full mask of float('-inf') where the attention should be masked
    mask = torch.full((length, length), float('-inf'), device=device, dtype=torch.float)
    
    # Create a band matrix where the valid attention window is set to 0
    for start in range(0, length, small_square_size):
        end = start + small_square_size
        mask[start:end, start:end] = 0  # Unmask this sub-matrix

    return mask


# testing
if __name__ == "__main__":
    # print(create_attention_mask(16, 1, torch.device("cpu")))
    print(create_attention_mask(16, 2, torch.device("cpu")))
    # print(create_attention_mask(16, 3, torch.device("cpu")))
    # print(create_attention_mask(16, 4, torch.device("cpu")))
    print(create_attention_mask(15, 2, torch.device("cpu")))
