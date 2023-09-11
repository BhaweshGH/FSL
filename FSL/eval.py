if __name__ == '__main__':
    import torch
    from torch import nn
    from torchvision.models import resnet18
    from prototypical_networks import PrototypicalNetworks
    from data_transform import SymbolDataset
    from utils import evaluate
    from wrap_fs_dataset import WrapFewShotDataset
    from torch.utils.data import DataLoader
    from tasksampler import TaskSampler
    from torchvision.transforms import transforms

    convolutional_network = resnet18(pretrained=True)
    convolutional_network.fc = nn.Flatten()
    loaded_model = PrototypicalNetworks(convolutional_network)
    FILE = "./model/model.pth"
    loaded_model.load_state_dict(torch.load(FILE))
    loaded_model.cuda()
    # for param in model.parameters():
    #     print(param)


    N_WAY = 5  # Number of classes in a task
    N_SHOT = 5  # Number of images per class in the support set
    N_QUERY = 5  # Number of images per class in the query set
    N_EVALUATION_TASKS = 100

    # The sampler needs a dataset with a "get_labels" method. Check the code if you have any doubt!
    # test_set.get_labels = lambda: [instance[1] for instance in test_set._flat_character_images]
    my_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    test_dataset = SymbolDataset(csv_file='Symbols_new.csv', root_dir='./data', transform=my_transform)
    test_set = WrapFewShotDataset(test_dataset,0,1)


    test_sampler = TaskSampler(
        test_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
    )


    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )

    print("Test metric:")
    evaluate(test_loader, loaded_model)
