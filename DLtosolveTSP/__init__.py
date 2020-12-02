import argparse
from DLtosolveTSP.gym import train_the_model
from DLtosolveTSP.test.test_tsplib_files import results_on_tsplib
from DLtosolveTSP.generation_data.create_dataset import create_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--operation', type=str, help='which operation to deal', default='generation')
    parser.add_argument('--num_pixels', type=int, help='how many pixels for image', default=192)
    parser.add_argument('--ray_dot', type=int, help='ray of each dot in the image', default=2)
    parser.add_argument('--thickness_edge', type=int, help='ray of each point in the image', default=1)

    parser.add_argument('--total_number_instances_train', type=int,
                        help="number of instances to use during train",
                        default=200000)
    parser.add_argument('--num_instances_x_file', type=int,
                        help="number of instances to save in a file",
                        default= 1000)
    parser.add_argument('--bs', type=int, help="batch size for training", default=16)

    settings = parser.parse_args()

    for k, v in vars(settings).items():
        print(f'{k} = "{v}"')

    operations = {"generation": create_data,
                  "train": train_the_model,
                  "tsplib_test": results_on_tsplib }

    settings.operation = "generation"
    operations[settings.operation](settings)

    settings.operation = "train"
    operations[settings.operation](settings)

    settings.operation = "tsplib_test"
    operations[settings.operation](settings)
