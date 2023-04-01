from utils.multi_class_preprocessor import MultiClassPreprocessor
from utils.multi_class_trainer import MultiClassTrainer

if __name__ == '__main__':
    label2id = {
        'bola': 0,
        'news': 1,
        'bisnis': 2,
        'tekno': 3,
        'otomotif': 4
    }
    
    prepro = MultiClassPreprocessor(max_length = 24, 
                                    preprocessed_dir = "dataset/preprocessed",
                                    train_data_dir = "dataset/training.res",
                                    test_data_dir = "dataset/testing.res",
                                    batch_size = 20)
    
    train_dataset, validation_dataset, test_dataset = prepro.preprocessor_manual()
    
    

    mclass_trainer = MultiClassTrainer(dropout = 0.1, 
                                    lr = 2e-5,
                                    max_epoch = 10,
                                    device = "cuda",
                                    n_class= len(label2id))
    
    mclass_trainer.trainer(train_dataset, validation_dataset, test_dataset)