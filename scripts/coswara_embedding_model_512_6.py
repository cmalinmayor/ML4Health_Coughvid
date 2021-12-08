from coughvid import CoswaraEmbeddingTrainer
import sys

if __name__ == '__main__':
    data_dir = '../data/coswara'
    name = sys.argv[1]
    trainer = CoswaraEmbeddingTrainer(data_dir, unit_sec=6.0,
                                      feature_d=512, name=name)
    trainer.train_model(
            num_epochs=100,
            use_wandb=False)
