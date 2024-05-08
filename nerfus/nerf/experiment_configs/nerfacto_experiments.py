from nerfus.nerf.experiment_configs.utils import Argument

arguments_list_of_lists = []

output_folder = "outputs-postprocessed"

dataset_lists = [
    Argument(
        name="1",
        arg_string=f"--data dataset/1 --pipeline.nerf_checkpoint_path outputs-checkpoints/1.ckpt --output-dir {output_folder}/1",
    ),
    Argument(
        name="2",
        arg_string=f"--data dataset/2 --pipeline.nerf_checkpoint_path outputs-checkpoints/2.ckpt --output-dir {output_folder}/2",
    ),
    Argument(
        name="3",
        arg_string=f"--data dataset/3 --pipeline.nerf_checkpoint_path outputs-checkpoints/3.ckpt --output-dir {output_folder}/3",
    ),
    Argument(
        name="4",
        arg_string=f"--data dataset/4 --pipeline.nerf_checkpoint_path outputs-checkpoints/4.ckpt --output-dir {output_folder}/4",
    ),
    Argument(
        name="5",
        arg_string=f"--data dataset/5 --pipeline.nerf_checkpoint_path outputs-checkpoints/5.ckpt --output-dir {output_folder}/5",
    ),
    Argument(
        name="6",
        arg_string=f"--data dataset/6 --pipeline.nerf_checkpoint_path outputs-checkpoints/6.ckpt --output-dir {output_folder}/6",
    ),
    Argument(
        name="7",
        arg_string=f"--data dataset/7 --pipeline.nerf_checkpoint_path outputs-checkpoints/7.ckpt --output-dir {output_folder}/7",
    ),
    Argument(
        name="8",
        arg_string=f"--data dataset/8 --pipeline.nerf_checkpoint_path outputs-checkpoints/8.ckpt --output-dir {output_folder}/8",
    ),
    Argument(
        name="9",
        arg_string=f"--data dataset/9 --pipeline.nerf_checkpoint_path outputs-checkpoints/9.ckpt --output-dir {output_folder}/9",
    ),
    Argument(
        name="10",
        arg_string=f"--data dataset/10 --pipeline.nerf_checkpoint_path outputs-checkpoints/10.ckpt --output-dir {output_folder}/10",
    ),
]
arguments_list_of_lists.append(dataset_lists)

experiments_list = [
    Argument(
        name="nerfacto",
        arg_string="--pipeline.use_visibility_loss False --pipeline.use_singlestep_cube_loss False",
    ),
]
arguments_list_of_lists.append(experiments_list)
