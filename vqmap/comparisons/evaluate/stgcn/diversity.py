import torch
import numpy as np


# from action2motion
def calculate_diversity_multimodality(activations, labels, num_labels, seed=None):
    diversity_times = 200
    multimodality_times = 20
    labels = labels.long()
    num_motions = len(labels)

    diversity = 0

    if seed is not None:
        np.random.seed(seed)
        
    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    for first_idx, second_idx in zip(first_indices, second_indices):
        diversity += torch.dist(activations[first_idx, :],
                                activations[second_idx, :])
    diversity /= diversity_times
    
    # diversity_by_action = np.zeros((num_labels))
    diversity_by_action = []
    # diversity by action
    for action in range(num_labels):
        action_indices = torch.where(labels == action)[0]
        num = len(action_indices)
        if num ==  0:
            diversity_by_action.append(0.0)
            continue
        first_indices = np.random.randint(0, num, diversity_times)
        second_indices = np.random.randint(0, num, diversity_times)
        for first_idx, second_idx in zip(first_indices, second_indices):
            diversity += torch.dist(activations[action_indices[first_idx], :],
                                    activations[action_indices[second_idx], :])
        diversity /= diversity_times
        
        # diversity_by_action[action] = diversity.item()
        diversity_by_action.append(diversity.item())
        
    

    multimodality = 0
    label_quotas = np.repeat(multimodality_times, num_labels)
    mask = []
    for label in range(num_labels):
        num = len(torch.where(labels == label)[0])
        if num == 0:
            label_quotas[label] = 0
            continue
        mask.append(label)
        
    while np.any(label_quotas > 0):
        # print(label_quotas)
        first_idx = np.random.randint(0, num_motions)
        first_label = labels[first_idx]
        if not label_quotas[first_label]:
            continue

        second_idx = np.random.randint(0, num_motions)
        second_label = labels[second_idx]
        while first_label != second_label:
            second_idx = np.random.randint(0, num_motions)
            second_label = labels[second_idx]

        label_quotas[first_label] -= 1

        first_activation = activations[first_idx, :]
        second_activation = activations[second_idx, :]
        multimodality += torch.dist(first_activation,
                                    second_activation)

    multimodality /= (multimodality_times * len(mask))

    return diversity.item(), diversity_by_action, multimodality.item()

