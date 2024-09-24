from dataset_preparation import prepare_dataset, preprocess_dataset
from model_preparation import prepare_model
from torch.utils.data import DataLoader


def train(model, train_dataset, batch_size=8, learning_rate=5e-5, num_epochs=3):
    import torch.optim as optim
    # DataLoaderの作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # オプティマイザの設定
    optimizer_name = input("使用するオプティマイザを選択してください (e.g. Adam, AdamW, SGD, RMSprop): ")
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            outputs = model(**batch)
            print(outputs)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train loss: {avg_loss:.4f}")

    return model

def test(model, test_dataset, batch_size=8):
    # DataLoaderの作成
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model.eval()
    total_loss = 0
    for batch in test_loader:
        outputs = model(**batch)
        total_loss += outputs.loss.item()
    
    avg_loss = total_loss / len(test_loader)
    print(f"Test loss: {avg_loss:.4f}")

    return avg_loss


def main():
    query_dataset = input("あれば具体的なデータセット名、なければ所望のデータが得られるようなクエリを入力: ")
    query_model = input("あれば具体的なモデル名、なければ所望のモデルが得られるようなクエリを入力: ")  # 画像系は timm をクエリに含むように指示する
    dataset = prepare_dataset(query_dataset)  # Hugging Face datasets の load_dataset を使用
    model, tokenizer = prepare_model(query_model)  # Hugging Face transformers の AutoModel と AutoTokenizer を使用
    dataset = preprocess_dataset(dataset, model, tokenizer)

    # トレーニングの実行
    is_train_needed = True
    if is_train_needed:
        model = train(model, dataset['train'])

    # テストの実行
    test_loss = test(model, dataset['test'])

    # モデルの保存
    model.save_pretrained("fine_tuned_model")
    print("モデルの学習が完了し、保存されました。")

if __name__ == "__main__":
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(torch.cuda.device_count())])
    main()