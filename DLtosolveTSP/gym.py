import torch
from tqdm import tqdm
from DLtosolveTSP.tools.dir_manager import Dir_Manager
from DLtosolveTSP.test.test_reconstruction import test_the_model
from DLtosolveTSP.test.test_tsplib_files import results_on_tsplib
from DLtosolveTSP.network.CNN_for_TSP import CNN_for_GoodEdgeDistribution
from DLtosolveTSP.generation_data.data_manager import TrainDatasetHandler


def train_the_model(settings):
    dir_ent = Dir_Manager(settings)
    data_handler = TrainDatasetHandler(settings, dir_ent.folder_images, 0)
    print(f"total number of instances {data_handler.len}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device : {device}')

    model  = CNN_for_GoodEdgeDistribution()
    model = model.to(device)
    model.apply(model.weight_init)

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_iterations = settings.total_number_instances_train // settings.bs
    steps_logger = tqdm(range(num_iterations), desc='step')
    epoch_logger = tqdm(range(5), desc="epoch")
    val_loss = 1000.
    for epoch in epoch_logger:
        name, checkpoint_path = dir_ent.create_name_model(epoch)
        data_handler = TrainDatasetHandler(settings, dir_ent.folder_images, 0)
        data_handler.train = True

        for i in steps_logger:
            x ,y = data_handler[ i *settings.bs : ( i +1 )* settings.bs]

            x = x.to(device)
            y = y.to(device)

            model.train()
            optimizer.zero_grad()
            Good_Edge_Distribution = model(x)

            loss = criterion(Good_Edge_Distribution, y)

            loss.backward()
            optimizer.step()

            loss = loss.item()
            loss = loss/ settings.bs
            log_str = f'loss: {loss:.5f}'
            steps_logger.set_postfix_str(log_str)
            epoch_logger.set_postfix_str(log_str)
            if i % 250 == 0:
                val_loss = do_test_cv(i, epoch, loss, model, checkpoint_path, dir_ent, settings)

            if val_loss < 2:
                val_loss = do_test_cv(i, epoch, loss, model, checkpoint_path, dir_ent, settings)


    torch.save(model.state_dict(), dir_ent.final_model_file)
    print("model saved")

def do_test_cv(i, epoch, loss, model, checkpoint_path, dir_ent, settings):
    checkpoint_name = f'{epoch}_{i}_{loss:.5f}_state_dict.pth'
    torch.save(model.state_dict(), dir_ent.final_model_file)
    torch.save(model.state_dict(), checkpoint_path + '/' + checkpoint_name)
    val = results_on_tsplib(settings, epoch, i)
    settings.operation = 'test'
    test_the_model(settings, epoch, i)
    settings.operation = 'train'
    return val
