import torch

if __name__ == '__main__':

    from data_transform import SymbolDataset
    from torchvision.transforms import transforms
    from torch.utils.data import DataLoader
    from torch import nn, optim
    from wrap_fs_dataset import WrapFewShotDataset
    from tasksampler import TaskSampler
    from torchvision.models import resnet18
    from tqdm import tqdm
    from prototypical_networks import PrototypicalNetworks
    from utils import plot_images,sliding_average,evaluate



    my_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = SymbolDataset(csv_file='Symbols_mix_train_data.csv', root_dir='./data', transform=my_transform)
    train_set = WrapFewShotDataset(train_dataset,0,1)

    # test_dataset = SymbolDataset(csv_file='Symbols_valve_test.csv', root_dir='./data', transform=my_transform)
    # test_set = WrapFewShotDataset(test_dataset,0,1)



    # Data Loading and Few Shot sampling

    N_WAY = 5  # Number of classes in a task
    N_SHOT = 3  # Number of images per class in the support set
    N_QUERY = 7  # Number of images per class in the query set
    N_TRAINING_EPISODES = 3000

    # The sampler needs a dataset with a "get_labels" method. Check the code if you have any doubt!
    # test_set.get_labels = lambda: [instance[1] for instance in test_set._flat_character_images]
    train_sampler = TaskSampler(
        train_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_TRAINING_EPISODES
    )


    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )

    # (
    #     example_support_images,
    #     example_support_labels,
    #     example_query_images,
    #     example_query_labels,
    #     example_class_ids,
    # ) = next(iter(train_loader))
    #
    # print(len(train_loader))

    convolutional_network = resnet18(pretrained=True)
    convolutional_network.fc = nn.Flatten()
    print(convolutional_network)

    model = PrototypicalNetworks(convolutional_network).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    def fit(
            support_images: torch.Tensor,
            support_labels: torch.Tensor,
            query_images: torch.Tensor,
            query_labels: torch.Tensor,
    ) -> float:
        optimizer.zero_grad()
        classification_scores = model(
            support_images.cuda(), support_labels.cuda(), query_images.cuda()
        )

        loss = criterion(classification_scores, query_labels.cuda())
        loss.backward()
        optimizer.step()

        return loss.item()


    # Train the model yourself with this cell

    log_update_frequency = 10

    all_loss = []
    model.train()
    with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
        for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
        ) in tqdm_train:
            loss_value = fit(support_images, support_labels, query_images, query_labels)
            all_loss.append(loss_value)

            if episode_index % log_update_frequency == 0:
                tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency))





    # print(example_class_ids)
    # print(example_query_labels)

    # plot_images(example_support_images,"Support Image", images_per_row=N_SHOT)
    # plot_images(example_query_images, "query images", images_per_row=N_QUERY)

    # model.eval()
    #
    # example_scores = model(
    #     example_support_images.cuda(),
    #     example_support_labels.cuda(),
    #     example_query_images.cuda(),
    # ).detach()
    #
    #
    # _, predicted_labels = torch.max(example_scores.data, 1)
    #
    # print("Ground Truth / Predicted")
    # for i in range(len(example_query_labels)):
    #     print(
    #         f"{example_class_ids[example_query_labels[i]]} / {example_class_ids[predicted_labels[i]]}"
    #     )
    # print("Train metric:")
    # evaluate(train_loader,model)

    # for param in model.parameters():
    #     print(param)

    FILE = "./model/model.pth"
    torch.save(model.state_dict(),FILE)
    # N_WAY = 3  # Number of classes in a task
    # N_SHOT = 3  # Number of images per class in the support set
    # N_QUERY = 6  # Number of images per class in the query set
    # N_EVALUATION_TASKS = 100
    #
    # # The sampler needs a dataset with a "get_labels" method. Check the code if you have any doubt!
    # # test_set.get_labels = lambda: [instance[1] for instance in test_set._flat_character_images]
    # test_sampler = TaskSampler(
    #     test_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
    # )
    #
    # test_loader = DataLoader(
    #     test_set,
    #     batch_sampler=test_sampler,
    #     num_workers=12,
    #     pin_memory=True,
    #     collate_fn=test_sampler.episodic_collate_fn,
    # )
    #
    # print("Test metric:")
    # evaluate(test_loader, model)
