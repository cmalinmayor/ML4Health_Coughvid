from coughvid import CoswaraTrainer

if __name__ == '__main__':
    data_dir = 'D://coughvid/coswara'
    trainer = CoswaraTrainer(data_dir, name='baseline')
    trainer.train_model(
            model_type='resnet18',
            num_epochs=100,
            use_wandb=False)
