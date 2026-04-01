import argparse
import os

import utils


def generate_configs_and_names(experiment: utils.Experiment):
    variants = experiment.get_variants()
    configs_and_names = []
    for base_config in experiment.bases:
        for variant in variants:
            config = utils.BareConfig.load(base_config)
            name = ""
            seed = None
            remove_trailing_underscore = False
            for k, v in variant.items():
                config_path = k.split(".")
                config_dict = config
                while len(config_path) > 1:
                    if config_path[0] not in config_dict:
                        raise ValueError("Experiment specified key not in config: " + str(k))
                    config_dict = config_dict[config_path[0]]
                    config_path.pop(0)
                if config_path[0] not in config_dict:
                    raise ValueError("Experiment specified key not in config: " + str(k))
                config_dict[config_path[0]] = v

                if k in utils.FOLDER_KEYS and len(experiment[k]) > 1:
                    name = os.path.join(v, name)
                elif k == "seed" and len(experiment["seed"]) > 1:
                    seed = v
                elif len(experiment[k]) > 1:
                    name += experiment.format_name(k, v) + "_"
                    remove_trailing_underscore = True

            if remove_trailing_underscore:
                name = name[:-1]
            if len(experiment.bases) > 1:
                name = os.path.join(os.path.splitext(os.path.basename(base_config))[0], name)
            name = os.path.join(experiment.name, name)
            if seed is not None:
                name = os.path.join(name, "seed-" + str(seed))
            configs_and_names.append((config, name))
    return configs_and_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, required=True, help="Path to a sweep json file.")
    parser.add_argument("--output-root", type=str, required=True, help="Base directory for experiment outputs.")
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default=None,
        help="Directory to store the generated manifest and stable config files. Defaults under slurm/manifests/<sweep_name>.",
    )
    args = parser.parse_args()

    experiment = utils.Experiment.load(args.sweep)
    configs_and_names = generate_configs_and_names(experiment)

    sweep_name = os.path.splitext(os.path.basename(args.sweep))[0]
    manifest_dir = (
        os.path.join("slurm", "manifests", sweep_name) if args.manifest_dir is None else args.manifest_dir
    )
    config_dir = os.path.join(manifest_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)

    manifest_path = os.path.join(manifest_dir, "manifest.tsv")
    with open(manifest_path, "w") as manifest_file:
        for i, (config, run_name) in enumerate(configs_and_names, start=1):
            stable_config_path = os.path.join(config_dir, f"{i:04d}.yaml")
            config.save(stable_config_path)
            run_path = os.path.join(args.output_root, run_name)
            manifest_file.write(f"{stable_config_path}\t{run_path}\n")

    print("[slurm] Wrote manifest to", manifest_path)
    print("[slurm] Number of tasks:", len(configs_and_names))
    print("[slurm] Submit with:")
    print(f"sbatch --array=1-{len(configs_and_names)} --time=24:00:00 slurm/train_array.sbatch {manifest_path}")


if __name__ == "__main__":
    main()
