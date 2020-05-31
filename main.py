from train.train_netG import train as trainZSL
from train.train_DAMSM import run_train as trainAttnGAN

def main():
    #first train ZSL
    #trainZSL()
    #then train AttnGAN with pretrained ZSL encoder, instead of trial nn.Embedding
    trainAttnGAN()
    

if __name__ == "__main__":
    main()
