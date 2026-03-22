import os
import re
import glob
import csv
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

DATA_DIR = "reuters21578"
OUTPUT_DIR = "processed_reuters"

os.makedirs(OUTPUT_DIR, exist_ok=True)

documents = []
labels = []

sgm_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.sgm")))

for file in sgm_files:
    with open(file, "r", encoding="latin-1") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

        for r in soup.find_all("reuters"):

            topics = r.topics
            body = r.body

            if topics and body:

                topic_list = [d.text for d in topics.find_all("d")]
                text = body.text.strip()

                if text and topic_list:
                    documents.append(text)
                    labels.append(topic_list)

print("Total documents:", len(documents))


# 多标签集合
all_labels = sorted(set(l for sub in labels for l in sub))

print("Total labels:", len(all_labels))


# multi-hot编码
def encode_labels(label_list):

    vec = [0] * len(all_labels)

    for l in label_list:
        vec[all_labels.index(l)] = 1

    return vec


encoded_labels = [encode_labels(l) for l in labels]


# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    documents,
    encoded_labels,
    test_size=0.2,
    random_state=42
)


def save_csv(filename, texts, ys):

    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w", newline="", encoding="utf8") as f:

        writer = csv.writer(f)

        header = ["text"] + all_labels
        writer.writerow(header)

        for text, label_vec in zip(texts, ys):
            writer.writerow([text] + label_vec)


save_csv("train.csv", X_train, y_train)
save_csv("test.csv", X_test, y_test)

print("Saved to", OUTPUT_DIR)