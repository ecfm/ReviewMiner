import os, time, gc, json, pickle, argparse, math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.nn import DataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup, Conv1D
from tensorboardX import SummaryWriter
from tqdm import tqdm
import importlib
import logging
import copy

# from apex.optimizers import FusedAdam
# from apex import amp
# from apex.fp16_utils import FP16_Optimizer

from data.util import *
from util import *

from model import *

from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

devices = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = devices


def compute_loss(device, model, uids, pids, input_tokens, target_tokens, mask, loss_fn):
    input_tokens = input_tokens.to(device)
    target_tokens = target_tokens.to(device)
    mask = mask.to(device)

    outputs = model(input_ids=input_tokens, attention_mask=mask, uids=uids, pids=pids)
    logits = outputs[0]
    num_logits = logits.size(-1)

    # Perform masking
    if mask is not None:
        mask = mask.type(torch.bool)
        mask = mask.to(device)
        logits = logits.masked_select(mask.unsqueeze(-1))
        target_tokens = target_tokens.masked_select(mask)

    ce_loss = loss_fn(logits.view(-1, num_logits), target_tokens.view(-1))
    loss = ce_loss.mean()

    return loss, ce_loss


def train_step(device, model, optimizer, uids, pids, input_tokens, target_tokens, mask, loss_fn):
    output = []
    optimizer.zero_grad()
    loss, ce_loss = compute_loss(device, model, uids, pids, input_tokens, target_tokens, mask, loss_fn)
    # with amp.scale_loss(loss, optimizer) as scaled_loss:
    #     scaled_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.0)  # max_grad_norm=1.0
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # max_grad_norm=1.0
    optimizer.step()
    output.append((loss.item(), ce_loss.mean().item()))

    return output


def top_k_top_p_filtering(logits, top_k=100, top_p=0.95, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


def repeat_score(text, ngram=[3, 4, 5, 6]):
    ngram_list = []
    for ng in ngram:
        ngram_list.append([text[idx:idx + ng] for idx in range(len(text) - ng - 1)])

    max_occurs = []
    for ngrams in ngram_list:
        count_result = Counter([' '.join(n) for n in ngrams])
        try:
            max_occurs.append(
                max(count_result.values())
            )
        except:
            pass

    scores = [max_oc / ((len(text) / ngram[idx]) + ngram[idx]) for idx, max_oc in enumerate(max_occurs)]
    return max(scores) if len(scores) >= 1 else 1.0


def sample_sequence(model, tokenizer, length, uids, pids, input_tokens, mask, batch_size=None,
                    temperature=1, top_k=100, top_p=0.95, device='cuda', sample=True, eos_token=None, model_type='cvae'):

    with torch.no_grad():
        _, _, mem = model(input_ids=input_tokens, attention_mask=mask, uids=uids, pids=pids)
        prev = input_tokens

        output = prev
        probability = torch.FloatTensor([], device=device)
        if_end = torch.tensor([False] * batch_size, dtype=torch.bool, device=device)

        for i in range(length): #trange
            logits, _, mem = model(input_ids=prev, past=mem, attention_mask=mask, uids=uids, pids=pids)

            logits = logits[:, -1, :] / temperature
            logits = top_k_top_p_filtering(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            if sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = torch.topk(probs, k=1, dim=-1)

            probability = torch.cat((probability, probs.gather(1, next_token)), dim=1)
            output = torch.cat((output, next_token), dim=1)
            prev = next_token

            # early stopping if all sents have ended once
            if_end[next_token.view(-1).eq(eos_token)] = True
            if if_end.all(): break
    return output, probability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)

    # Default parameters are set based on single GPU training
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument('--data_type', type=str, default='t1', choices=['t' + str(i) for i in range(9)], help="t: type")
    parser.add_argument('--model_type', type=str, default='gpt2', choices=['gpt2'])
    parser.add_argument('--iterations', type=int, default=101640 * 4)  # wp 850001  wi 300001 ax 300001 yp 800001
    parser.add_argument('--dataset', type=str, default='am', choices=['am'], help="Dataset to use for training")
    parser.add_argument('--warmup', type=int, default=10000,
                        help="Amount of iterations to warmup, then decay. (-1 for no warmup and decay)")

    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[16],
                        help='batch size per GPU. Lists the schedule.')
    parser.add_argument('--seq-lens', nargs='+', type=int, default=[1024],
                        help='seq length per sample. Lists the schedule.')
    parser.add_argument('--switch-time', type=float, default=0,
                        help="Percentage of iterations to spend on short sequence training.")
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--out-dir', type=str, default='out')
    parser.add_argument('--load', type=str, help='path to load model from') # , default='out/test/'
    parser.add_argument('--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers')
    # use GPU
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--no_gpu', action="store_true")

    parser.add_argument('--fp16', action='store_true', help="Train using FP16?")
    parser.add_argument('--fp16_opt_level', default='O0', type=str, required=False)

    # KL cost annealing, increase beta from beta_0 to 1 in beta_warmup steps
    parser.add_argument('--beta_0', default=1.00, type=float)
    parser.add_argument('--beta_warmup', type=int, default=50000)
    # cyc_vae parameters
    parser.add_argument('--cycle', type=int, default=101640)

    parser.add_argument('--add_input', action="store_true")
    parser.add_argument('--add_attn', action="store_true")
    parser.add_argument('--attn_proj_vary', action="store_true")

    args = parser.parse_args('test --batch-sizes 1 --seq-lens 1024 '
                             '--add_input --fp16'.split()) # wi.12.proj_vary_beta_cvae

    # GPU
    if not torch.cuda.is_available(): args.no_gpu = True
    gpu = not args.no_gpu
    if gpu:
        print("There are ", torch.cuda.device_count(), " available GPUs!")
        # print('Setting GPUs {}'.format(args.device))
        print('Using GPU devices {}'.format(devices))
        torch.cuda.set_device(args.gpu)
        print('Current single GPU: {}'.format(torch.cuda.current_device()))
    device = torch.device(args.gpu if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # logging
    save_folder = os.path.join(args.out_dir, args.experiment)
    os.makedirs(save_folder, exist_ok=True)
    t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)
    v_writer = SummaryWriter(os.path.join(save_folder, 'val'), flush_secs=5)
    import logging

    logging.basicConfig(filename=os.path.join(save_folder, 'train.log'),
                        level=logging.INFO, format='%(asctime)s--- %(message)s')
    logging.info('\n*******************************************************************************\n')
    logging.info("the configuration:")
    logging.info(str(args).replace(',', '\n'))



    print('Loading models...')
    cache_dir = os.path.join(args.out_dir, 'model_cache')
    os.makedirs(cache_dir, exist_ok=True)
    # Load pre-trained teacher tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
    # Hack to allow tokenizing longer sequences.
    # tokenizer.max_len = int(1e12)
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=cache_dir)
    print('gpt2_params:', num_params(gpt2_model))  # gpt2: 124439808
    config = GPT2Config()

    # add special tokens
    special_tokens_dict = {
        'bos_token': '<|startoftext|>',
        'pad_token': '<|endoftext|>',
        'cls_token': '<|startofcond|>',
        'sep_token': '<|sepofcond|>',
        'mask_token': '<|endofcond|>'
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'special tokens')
    # Notice: resize_token_embeddings expect to receive the full size of the new vocab
    gpt2_model.resize_token_embeddings(len(tokenizer))
    # assert tokenizer.pad_token == '<|startoftext|>'
    print('Setup data...')
    # Batch and sequence length schedule
    assert len(args.batch_sizes) == len(args.seq_lens)
    batch_schedule = list(zip(map(int, args.batch_sizes), map(int, args.seq_lens)))
    assert len(batch_schedule) <= 2, 'Currently not supporting multiple schedule'
    cur_b_schedule = len(batch_schedule) - 1 if args.switch_time == 0 else 0
    print('Batch schedule', batch_schedule)
    train_loader, val_loader, test_loader = prepare_dataset(
        args.data_dir, args.dataset, tokenizer,
        batch_schedule[cur_b_schedule][0], batch_schedule[cur_b_schedule][1],
        batch_schedule[-1][0], batch_schedule[-1][1],
        batch_schedule[-1][0], batch_schedule[-1][1],
        num_workers=args.workers, data_type=args.data_type
    )

    model = Decoder(config, add_input=args.add_input, add_attn=args.add_attn, attn_proj_vary=args.attn_proj_vary)
    init_para_frompretrained(model, gpt2_model.transformer, share_para=True)
    model.lm_head.weight = gpt2_model.lm_head.weight
    print('model_params:', num_params(model))  # 286694400
    if args.load:
        print('Loading model weights...')
        state = torch.load(os.path.join(args.load, 'model_latest.pt'))  # , map_location='cpu' model_latest.pt
        if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attr 'module'
            state_copy = copy.copy(state)
            keys = state_copy.keys()
            for k in keys:
                state[k.replace('module.', '')] = state.pop(k)
        model.load_state_dict(state)
        gc.collect()
    print('Done.')

    # fix pre-trained parameters before certain iterations
    tuning_all_after_iters = 40000
    tuning_all = False
    for name, parameter in model.named_parameters():
        # print((name, parameter.requires_grad))
        new_pars = ['input_proj', 'attn_proj']

        if not any([True if n in name else False for n in new_pars]):
           parameter.requires_grad = False

    ###
    # val_loader = test_loader
    ###

    print('Wrapping models and optimizers...')
    # Apply linear scaling rule to increase batch size for short sequence training.
    lr_schedule = switch_schedule(linear_schedule(args), batch_schedule[cur_b_schedule][0] / batch_schedule[-1][0],
                                  int(args.iterations * args.switch_time))
    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    # model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    print('Done.')

    print('Begin training iterations')
    logging.info("Begin training iterations")
    max_val_batches = 10  # max num. of val batches
    logging.info("Total iteration: %d" % args.iterations)
    e = 0  # number of epoch
    num_iters = 0
    optimizer.zero_grad()
    beta = args.beta_0
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    def val_step(val_loader):
        model.eval()

        n_words_bpe = 0
        n_words = 0
        logp_sum = 0.0

        logging.info("Validation loop.         Batches: %d" % len(val_loader))
        logging.info("Validation loop. max_val_batches: %d" % max_val_batches)
        print('Begin validation iterations')
        # val_iter = iter(val_loader); x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask = next(val_iter)
        with tqdm(total=min(len(val_loader), max_val_batches)) as pbar:
            for i, (uids, pids, input_tokens, target_tokens, mask) in enumerate(val_loader):
                with torch.no_grad():
                    loss, ce_loss = compute_loss(device, model, uids, pids, input_tokens, target_tokens, mask, loss_fn)

                if len(target_tokens.size()) == 1:
                    target_tokens = target_tokens.unsqueeze(0)
                n, l = target_tokens.size()

                text = target_tokens[0, :].tolist()
                logprob = ce_loss.tolist()
                assert len(text) == len(logprob)

                if endoftext in text:
                    idx = text.index(endoftext)
                    text = text[:idx]
                    logprob = logprob[:idx]

                logp_sum += sum(logprob)

                n_words_bpe += len(text)

                story = [tokenizer.decode(target_tokens[i, :]) for i in range(n)]
                words = sum([len(
                    [t for t in re.split('("|\'|!|\?|\.|,|:| |\n|???|???|???|;|\(|\)|`)', s) if t != ' ' and t != '']) for
                    s in story])
                n_words += words

                if i > max_val_batches:
                    break
                pbar.update(1)

        loss_bpe = logp_sum / n_words_bpe
        ppl_bpe = round(math.exp(min(logp_sum / n_words_bpe, 100)), 3)
        ppl_word = round(math.exp(min(logp_sum / n_words, 100)), 3)

        v_writer.add_scalar('loss', loss_bpe, num_iters)
        v_writer.add_scalar('ppl_bpe', ppl_bpe, num_iters)
        v_writer.add_scalar('ppl_word', ppl_word, num_iters)
        logging.info('val loss    : %.4f' % loss_bpe)
        logging.info('val ppl_bpe : %.4f' % ppl_bpe)
        logging.info('val ppl_word: %.4f' % ppl_word)

        model.train()

    def generate(test_loader, num_iters):
        model.eval()

        n_samples = 0
        bleu4_sum = 0.0
        rouge_scores_values_sum = [0.0] * 9

        args.nsamples = 1
        args.batch_size = 1
        args.temperature = 0.95
        args.top_k = 100
        args.top_p = 0.95
        model_type = args.model_type

        # write samples to file
        samples_file = open(os.path.join(save_folder, 'generate-' + '%07d' % num_iters + '.txt'), 'w', encoding='utf8')
        print('Begin generate iterations')
        # test_iter = iter(test_loader); x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask = next(test_iter)
        with tqdm(total=len(test_loader)) as pbar:
            for i_test, (uids, pids, input_tokens, target_tokens, mask) in enumerate(
                    test_loader):

                if i_test >= 10: break


                eff_samples = []
                n, l = target_tokens.size()
                length = n
                storys = [tokenizer.decode(target_tokens[i, :]) for i in range(n)]
                storys_str = [s[:s.find("<|endoftext|>")] for s in storys]

                for _ in range(args.nsamples // args.batch_size):
                    # model, batch_size, temperature, top_k, top_p, eos_token, sample = VAE, args.batch_size, args.temperature, args.top_k, args.top_p, tokenizer.encoder['<|endoftext|>'], True
                    out, _ = sample_sequence(
                        model=model,
                        tokenizer=tokenizer,
                        length=length,
                        batch_size=args.batch_size,
                        uids=uids,
                        pids=pids,
                        input_tokens=input_tokens, 
                        mask=mask,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        device=device,
                        eos_token=tokenizer.encoder['<|endoftext|>'],
                        model_type=model_type
                    )
                    out = out.tolist()

                    # extract story, check metrics
                    for i in range(len(out)):
                        text = out[i]

                        if endoftext in text:
                            idx = text.index(endoftext)
                            text = text[:idx]

                        text = tokenizer.decode(text).strip()

                        # score for one long text, higher than 0.075 usually means repetition
                        # rep_score = repeat_score(text.split(), ngram=[3, 4, 5, 6, 7, 8])
                        # if rep_score > 0.075:
                        #     # print(rep_score)
                        #     continue

                        try:
                            # check bleu
                            bleu4 = sentence_bleu([storys_str[i].split()], text,
                                                  smoothing_function=SmoothingFunction().method7)

                            # check rouge
                            rouge = Rouge()
                            rouge_scores = rouge.get_scores(text, storys_str[i])
                            rouge_scores_values = [v for k in rouge_scores[0].keys() for v in
                                                   rouge_scores[0][k].values()]

                            bleu4_sum += bleu4
                            rouge_scores_values_sum = [v1 + v2 for v1, v2 in
                                                       zip(rouge_scores_values_sum, rouge_scores_values)]
                            n_samples += 1
                        except:
                            bleu4 = 0.0
                            rouge_scores = [{'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                                             'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                                             'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}]

                        eff_samples.append((text, bleu4, rouge_scores))

                    pbar.update(1)

                for i in range(len(eff_samples)):
                    samples_file.write("=" * 50 + " SAMPLE " + str(i_test) + " " + "=" * 50)
                    samples_file.write('\n' * 2)

                    samples_file.write("=" * 40 + " Outlines  " + "=" * 40)
                    samples_file.write('\n' * 2)
                    samples_file.write("=" * 40 + " Story " + "=" * 40)
                    samples_file.write('\n' * 2)
                    samples_file.write(storys_str[i])
                    samples_file.write('\n' * 2)

                    samples_file.write("=" * 40 + " Generated " + "=" * 40)
                    samples_file.write('\n' * 2)
                    samples_file.write(eff_samples[i][0])
                    samples_file.write('\n' * 4)
                    samples_file.flush()

        print('Test complete with %05d samples.' % n_samples)
        logging.info("Test complete with %05d samples.", n_samples)
        logging.info("Iteration completed: %d" % num_iters)

        bleu4 = round(bleu4_sum / n_samples, 3)
        rouge_scores_values = [round(r / n_samples, 3) for r in rouge_scores_values_sum]
        print(' bleu-4:', bleu4)
        print(' rouge :', rouge_scores_values)
        logging.info(' bleu-4: %f', bleu4)
        logging.info(' rouge : %s', str(rouge_scores_values))

        model.train()

    # val_step(val_loader)
    # generate(test_loader, num_iters)
    torch.save(model.state_dict(), os.path.join(save_folder, 'model_' + '{:07d}'.format(num_iters) + '.pt'))

    while num_iters < args.iterations:
        # Run epoch
        st = time.time()

        # Training
        print('Training loop. Batches:', len(train_loader))
        logging.info('\n----------------------------------------------------------------------')
        logging.info("Training loop.       Batches: %d" % len(train_loader))

        # train_iter = iter(train_loader); x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask = next(train_iter)
        with tqdm(total=len(train_loader)) as pbar:
            for i, (uids, pids, input_tokens, target_tokens, mask) in enumerate(train_loader):

                # if num_iters % args.cycle >= args.cycle - args.beta_warmup:
                #     beta = min(1.0, beta + (1. - args.beta_0) / args.beta_warmup)

                if not tuning_all and num_iters >= tuning_all_after_iters:
                    for name, parameter in model.named_parameters():
                        # print((name, parameter.requires_grad))
                        parameter.requires_grad = True
                    tuning_all = True

                output = train_step(device, model, optimizer, uids, pids, input_tokens, target_tokens, mask, loss_fn)
                loss, ce_loss = output[-1]

                lr = scheduler.get_last_lr()[0]
                # Log to Tensorboard
                t_writer.add_scalar('loss', loss, num_iters)
                t_writer.add_scalar('ppl', math.exp(min(ce_loss, 10)), num_iters)
                t_writer.add_scalar('lr', lr, num_iters)
                t_writer.add_scalar('iter_time', time.time() - st, num_iters)


                st = time.time()
                end = num_iters >= args.iterations

                if args.warmup != -1:
                    scheduler.step()

                if end: break
                num_iters += 1
                pbar.update(1)

                if num_iters % args.cycle == 0:
                    beta = args.beta_0
                    logging.info('KL annealing restart')

                if num_iters % 10 == 0:
                    # test_plot(test_loader, num_iters)
                    val_step(val_loader)
                    generate(test_loader, num_iters)

                if num_iters % 50000 == 0:
                    print('Saving model...')
                    logging.info("Iteration completed: %d, remained %d" % (num_iters, args.iterations - num_iters))
                    logging.info("Saving model...")
                    logging.info('\n------------------------------------------------------')
                    torch.save(model.state_dict(), os.path.join(save_folder, 'model_' + '{:07d}'.format(num_iters) + '.pt'))

                # if args.switch_time > 0 and num_iters == int(args.iterations * args.switch_time):
                #     print('Switch to long sequence training')
                #     logging.info("Switch to long sequence training")
                #     cur_b_schedule += 1
                #     train_loader, val_loader, test_loader = prepare_dataset(
                #         args.data_dir, args.dataset, tokenizer,
                #         batch_schedule[cur_b_schedule][0], batch_schedule[cur_b_schedule][1],
                #         batch_schedule[-1][0], batch_schedule[-1][1],
                #         batch_schedule[-1][0], batch_schedule[-1][1],
                #         make_test=True,
                #         num_workers=args.workers, data_type=args.data_type
                #     )
        if not end:
            e += 1
            logging.info("Training loop. The ith epoch completed: %d" % e)

    torch.save(model.state_dict(), os.path.join(save_folder, 'model_latest.pt'))
    print('Training complete.')
    logging.info("Training complete.")


if __name__ == "__main__":
    main()
