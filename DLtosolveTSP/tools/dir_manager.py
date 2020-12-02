import os, shutil

def create_folder(folder_name_to_create, starting_folder="./"):
    folders_list = folder_name_to_create.split("/")
    folder_name = f"{starting_folder}"
    for i in range(len(folders_list[:-1])):
        folder_name += folders_list[i] + '/'
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
    return folder_name


class Dir_Manager:
    def __init__(self, settings):
        self.settings = settings
        self.folder_images = create_folder(f"data/train/", starting_folder="./DLtosolveTSP/")

        self.folder_model = create_folder('saved_models/', starting_folder="./DLtosolveTSP/")
        self.final_model_file = f"{self.folder_model}/final_state_dict.pth"

    def create_folder_to_save(self):
        self.folder_instances = create_folder(f"data/train/instances/", starting_folder="./DLtosolveTSP/")

    def create_name_model(self, epoch):
        name = f"{epoch}epoch"
        checkpoint_path = f'{self.folder_model}{name}/'

        folder = checkpoint_path
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        return name, checkpoint_path

    def path_to_risultati(self, dimension_problem, name_file):
        if not os.path.exists(f"./risultati_test/{dimension_problem}/"):
            os.mkdir(f"./risultati_test/{dimension_problem}/")
        return f"./risultati_test/{dimension_problem}/{name_file}.npy"

