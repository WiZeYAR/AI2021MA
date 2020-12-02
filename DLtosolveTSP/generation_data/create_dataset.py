import h5py
from tqdm import tqdm
from DLtosolveTSP.tools.dir_manager import Dir_Manager
from DLtosolveTSP.generation_data.data_manager import Generate_Images

def create_data(settings):
    dir_ent = Dir_Manager(settings)

    number_instances_per_file = settings.num_instances_x_file
    number_files = settings.total_number_instances_train // number_instances_per_file
    print(f"total number of files {number_files}")

    iterator_on_files = tqdm(range(number_files))

    generator = Generate_Images(settings)
    for file in iterator_on_files:
        seed = file * number_instances_per_file + 10000

        # to save created data
        data = generator.create_instances(number_instances_per_file, seed)

        hf = h5py.File(f"{dir_ent.folder_images}{seed}_file.h5", "w")
        generator.save(data, seed, hf, number_instances_per_file)
        hf.close()




