import smooth.measures
import smooth.model


def main():
    dataset = smooth.datasets.from_params("mnist12")

    model = smooth.model.train_shallow(
        dataset,
        learning_rate=0.01,
        init_scale=0.1,
        epochs=5,
        verbose=2,
        batch_size=64,
        # path_length_d_reg_coef=1e-3,
        path_length_f_reg_coef=1e-3,
    )

    print(model.history.history)
    print(smooth.measures.get_measures(model, dataset, samples=24))


if __name__ == "__main__":
    main()
