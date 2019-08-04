def build_training_visualization(model_name, train_metrics, losses, validation_metrics, path_to_save=None):
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(20, 30))
    figure.suptitle(f'Visualizations of {model_name} training progress', fontsize=16)

    ax1 = figure.add_subplot(3, 1, 1)
    ax1.plot(losses)
    ax1.set_title("Loss through epochs")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2 = figure.add_subplot(3, 1, 2)
    for metric, results in validation_metrics.items():
        ax2.plot(results, label=metric)
    ax2.legend(loc='upper left')
    ax2.set_title("Metrics through epochs")
    ax2.set_xlabel("Epochs")

    ax3 = figure.add_subplot(3, 1, 3)
    for metric, results in train_metrics.items():
        ax3.plot(results, label=metric)
    ax3.legend(loc='upper left')
    ax3.set_title("Results on dev set through epochs")
    ax3.set_xlabel("Epochs")

    if path_to_save:
        figure.savefig(path_to_save)




