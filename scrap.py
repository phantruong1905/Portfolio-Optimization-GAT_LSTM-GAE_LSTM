# Load embeddings
    embeddings_train = pd.read_csv("embeddings_train_after.csv")
    embeddings_val = pd.read_csv("embeddings_val_after.csv")
    embeddings_test = pd.read_csv("embeddings_test_after.csv")

    # Load original data for returns
    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('test.csv')
    test_df = pd.read_csv('val.csv')