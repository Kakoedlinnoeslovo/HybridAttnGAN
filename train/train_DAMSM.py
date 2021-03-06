from __future__ import print_function

from miscc_own.utils import mkdir_p
from miscc_own.losses import sent_loss, words_loss
from miscc_own.config import cfg, cfg_from_file

from dataset.dataset_DAMSM import TextDataset

from models.modelAttnGAN import RNN_ENCODER, CNN_ENCODER_SMALL

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)
import cfg.configs as configs


def train(dataloader, cnn_model, rnn_model, batch_size,
          labels, optimizer, epoch):
    cnn_model.train()
    rnn_model.train()
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    count = (epoch + 1) * len(dataloader)
    start_time = time.time()
    for step, data in enumerate(dataloader, 0):
        # print('step', step)
        rnn_model.zero_grad()
        cnn_model.zero_grad()


        imgs, captions, class_ids = data
        imgs = torch.squeeze(imgs, 0)
        captions = torch.squeeze(captions, 0)
        class_ids = torch.squeeze(class_ids, 0)

        words_features, sent_code = cnn_model(imgs)


        hidden = rnn_model.init_hidden(batch_size)


        words_emb, sent_emb = rnn_model(captions, hidden)

        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                                 -1, class_ids, batch_size)
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        loss = w_loss0 + w_loss1

        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data
        #
        loss.backward()
        #
        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(rnn_model.parameters(),
                                      cfg.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()

        if step % configs.UPDATE_INTERVAL == 0:
            count = epoch * len(dataloader) + step

            s_cur_loss0 = s_total_loss0.item() / configs.UPDATE_INTERVAL
            s_cur_loss1 = s_total_loss1.item() / configs.UPDATE_INTERVAL

            w_cur_loss0 = w_total_loss0.item() / configs.UPDATE_INTERVAL
            w_cur_loss1 = w_total_loss1.item() / configs.UPDATE_INTERVAL

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  's_loss {:5.2f} {:5.2f} | '
                  'w_loss {:5.2f} {:5.2f}'
                  .format(epoch, step, len(dataloader),
                          elapsed * 1000. / configs.UPDATE_INTERVAL,
                          s_cur_loss0, s_cur_loss1,
                          w_cur_loss0, w_cur_loss1))
            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            start_time = time.time()
            # attention Maps
            #img_set, _ = \
            #    build_super_images(imgs[-1].cpu(), captions,
            #                       ixtoword, attn_maps, att_sze)
            #if img_set is not None:
            #    im = Image.fromarray(img_set)
            #    fullpath = '%s/attention_maps%d.png' % (image_dir, step)
            #    im.save(fullpath)
    return count



def build_models(vocab_size, indim):
    batch_size = cfg.TRAIN.BATCH_SIZE
    # build model ############################################################
    text_encoder = RNN_ENCODER(vocab_size=vocab_size, nhidden=cfg.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER_SMALL(indim, cfg.TEXT.EMBEDDING_DIM)
    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 0
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E)
        text_encoder.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_E)
        #
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)

        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)
    if cfg.CUDA:
        text_encoder = text_encoder#.cuda()
        image_encoder = image_encoder#.cuda()
        labels = labels#.cuda()

    return text_encoder, image_encoder, labels, start_epoch


def run_train():
    import cfg.configs as configs

    if configs.cfg_file is not None:
        cfg_from_file(configs.cfg_file)


    if configs.data_dir != '':
        cfg.DATA_DIR = configs.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        configs.manualSeed = 100
    elif configs.manualSeed is None:
        configs.manualSeed = random.randint(1, 10000)
    random.seed(configs.manualSeed)
    np.random.seed(configs.manualSeed)
    torch.manual_seed(configs.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(configs.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)

    #torch.cuda.set_device(cfg.GPU_ID)
    #cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(type="train")

    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #
    dataset_val = TextDataset(type="test")

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # Train ##############################################################
    text_encoder, image_encoder, labels, start_epoch = build_models(vocab_size=dataset.dataset.text_dim, indim=dataset.dataset.feature_dim)
    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)
    # optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        lr = cfg.TRAIN.ENCODER_LR
        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
            epoch_start_time = time.time()
            count = train(dataloader, image_encoder, text_encoder,
                          batch_size, labels, optimizer, epoch)
            print('-' * 89)
            if len(dataloader_val) > 0:
                pass
                # s_loss, w_loss = evaluate(dataloader_val, image_encoder,
                #                           text_encoder, batch_size)
                # print('| end epoch {:3d} | valid loss '
                #       '{:5.2f} {:5.2f} | lr {:.5f}|'
                #       .format(epoch, s_loss, w_loss, lr))
            print('-' * 89)
            if lr > cfg.TRAIN.ENCODER_LR/10.:
                lr *= 0.98

            if (epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
                epoch == cfg.TRAIN.MAX_EPOCH):
                torch.save(image_encoder.state_dict(),
                           '%s/image_encoder%d.pth' % (model_dir, epoch))
                torch.save(text_encoder.state_dict(),
                           '%s/text_encoder%d.pth' % (model_dir, epoch))
                print('Save G/Ds models.')
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


if __name__ == "__main__":
    run_train()