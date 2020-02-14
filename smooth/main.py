from smooth import train_models

def main():
    hparams = train_models.Hyperparams(
        learning_rate=0.01,
        hidden_size=1000,
        epochs=10000,
        batch_size=1024,
        log_dir="logs_debug",
        iteration=None,
    )

    res = train_models.train_model(hparams, verbose=1)
    print(res)

if __name__ == "__main__":
    main()
