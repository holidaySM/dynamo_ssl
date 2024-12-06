import hydra


@hydra.main(version_base="1.2", config_path="configs", config_name="train")
def main(cfg):
    print(cfg.env.dataset)
    dataset = hydra.utils.instantiate(cfg.env.dataset)
    print(len(dataset))


if __name__ == "__main__":
    main()
