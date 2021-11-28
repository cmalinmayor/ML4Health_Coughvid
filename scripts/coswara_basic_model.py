from coughvid import CoswaraTrainer

if __name__ == '__main__':
    data_dir = '../data/coswara'
    trainer = CoswaraTrainer(data_dir)
    trainer.train_model(
            model_type='resnet18',
            num_epochs=50)
