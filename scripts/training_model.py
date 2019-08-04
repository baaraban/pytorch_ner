import torch
from scripts.utils import *
from tqdm import tqdm


def train_model(model, optimizer, loss_function,
                data_dict, batch_size, word_indexer,
                tag_indexer, training_metrics, validation_metrics, num_epochs=5):
    val_words, val_tags = data_dict['dev']

    def validate_model():
        with torch.no_grad():
            inputs = torch.tensor(word_indexer.elements_to_index(val_words), dtype=torch.long)
            true_vals = tag_indexer.elements_to_index(val_tags)
            tag_scores = model(inputs)
            prediction = get_tag_indexes_from_scores(tag_scores)
        validation_metrics.update(prediction, true_vals)
        validation_metrics.collect()
        for metric in validation_metrics.metrics_dict.keys():
            print(f"{metric} - {validation_metrics.metrics_dict[metric][-1]}")
        print()

    words, tags = data_dict['train']

    batches = list(get_batched(words, tags, batch_size))

    losses = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        running_loss = 0.0

        for sentence, tags in tqdm(batches):
            model.zero_grad()

            sentence_in = torch.tensor(word_indexer.elements_to_index(sentence), dtype=torch.long)
            targets = torch.tensor(tag_indexer.elements_to_index(tags), dtype=torch.long)

            tag_scores = model(sentence_in)

            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

            prediction = get_tag_indexes_from_scores(tag_scores.detach().numpy())
            training_metrics.update(prediction, targets)

            running_loss += loss.item() * sentence_in.size(0)

        training_metrics.collect()
        epoch_loss = running_loss / len(batches)

        losses.append(epoch_loss)
        print(f"Loss per epoch - {epoch_loss}")
        for metric in training_metrics.metrics_dict.keys():
            print(f"{metric} - {training_metrics.metrics_dict[metric][-1]}")
        print()
        print("Validating on dev test: ")
        validate_model()
        print()

    return model, training_metrics, validation_metrics, losses
