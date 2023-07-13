import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from datasets import Dataset
from pprint import pprint
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TFAutoModelForSequenceClassification

def read_csv():
    # Read CSV
    df = pd.read_csv("datasets/amazon/redmi6.csv", encoding="ISO-8859-1")

    # Print number of data (lines)
    print(len(df))

    # Histogram of stars
    star_df = df['Rating'].astype(str).str[0]
    print(star_df.head())

    ax = star_df.value_counts().plot(kind="bar", figsize=(16, 9))
    ax.set_xlabel("Stars")
    ax.set_ylabel("Count")
    plt.show()

    # Histogram of category
    ax = df["Category"].value_counts().plot(kind="bar", figsize=(16, 9))
    ax.set_xlabel("Comment Category")
    ax.set_ylabel("Count")
    plt.show()


def create_dataset():
    # Create dataset
    df = pd.read_csv("datasets/amazon/redmi6.csv", encoding="ISO-8859-1")
    # Drop useless columns
    df.drop(['Review Title', 'Customer name', 'Date', 'Category', 'Useful'], axis=1, inplace=True)
    # Get ratings and make them string into new dataset: star_df
    star_df = df['Rating'].astype(str).str[0]
    # Drop the original "Rating" column
    df.drop(['Rating'], axis=1, inplace=True)
    # Concat dataset and star_df tgt
    new_df = pd.concat([df, star_df], axis=1)
    print(new_df.head())
    # Convert the form of comments and ratings
    convert_dict = {'Comments': str, 'Rating': int }
    new_df = new_df.astype(convert_dict)

    # Get comments and labels
    texts, labels = [], []

    for comment in new_df["Comments"]:
            texts.append(comment)
    for rating in new_df["Rating"]:
        if rating <= 3:
            labels.append(0)
        else:
            labels.append(1)

    print(texts[:5])
    print(labels[:5])

    return Dataset.from_dict({ "text": texts, "label": labels })



if __name__ == '__main__':
    # Create dataset
    dataset = create_dataset()
    dataset = dataset.class_encode_column("label")
    # Train Test Split
    dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label")
    print(dataset)

    # Tokenize data
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True)
    tokenized_dataset = dataset.map(tokenize, batched=True)
    print(tokenized_dataset)
    # Collate data
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    # Create tensorflow dataset for training
    train_dataset = tokenized_dataset["train"].to_tf_dataset(
        columns=["attention_mask", "input_ids", "label"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator)
    # Create tensorflow dataset for validation
    valid_dataset = tokenized_dataset["test"].to_tf_dataset(
        columns=["attention_mask", "input_ids", "label"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator)]

    # Optimise dataset for performance
    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Download pre-trained model
    model = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english', num_labels=2)

    # Compile model
    optimizer = Adam(learning_rate=3e-5)
    model.compile(optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    # Train model
    history = model.fit(train_dataset, validation_data=valid_dataset, epochs=3)
    # Save model
    tokenizer.save_pretrained("amazon-review")
    model.save_pretrained("amazon-review")
