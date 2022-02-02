# -*- coding: utf-8 -*-
import copy
import os
from datetime import datetime, timedelta

import dill
import supar
import torch
import torch.distributed as dist
from supar.utils import Config, Dataset
from supar.utils.field import Field
from supar.utils.fn import download, get_rng_state, set_rng_state
from supar.utils.logging import init_logger, logger
from supar.utils.metric import Metric
from supar.utils.parallel import DistributedDataParallel as DDP
from supar.utils.parallel import is_master
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


class Parser(object):

    NAME = None
    MODEL = None

    def __init__(self, args, model, transform):
        self.args = args
        self.model = model
        self.transform = transform

    def train(self, train, dev, test, buckets=32, batch_size=5000, update_steps=1,
              clip=5.0, epochs=100, patience=20, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        batch_size = batch_size // update_steps
        if dist.is_initialized():
            batch_size = batch_size // dist.get_world_size()
        logger.info("Loading the data")

        '------------------------load init_labels of train predicted by existing parser--------------------'
        init_labels_train_dict = {}
        init_train = Dataset(self.transform, args.init_train, is_test=False, **args).build(batch_size, buckets, True,
                                                                                           dist.is_initialized())
        for sentence in init_train.sentences:
            annotations = sentence.annotations
            sent_id = annotations[-2]
            init_labels = sentence.transformed['labels']
            init_labels_train_dict.update({sent_id: init_labels})

        '------------------------load init_labels of test predicted by existing parser--------------------'
        init_labels_test_dict = {}
        init_test = Dataset(self.transform, args.init_test, is_test=True).build(batch_size, buckets)
        for sentence in init_test.sentences:
            annotations = sentence.annotations
            sent_id = annotations[-2]
            init_labels = sentence.transformed['labels']
            init_labels_test_dict.update({sent_id: init_labels})

        '------------------------add init_labels to train--------------------'
        train = Dataset(self.transform, args.train, is_test=False, **args).build(batch_size, buckets, True, dist.is_initialized())
        for sentence in train.sentences:
            annotations = sentence.annotations
            sent_id = annotations[-2]
            init_labels = init_labels_train_dict[sent_id]
            sentence.transformed.update({'init_labels': init_labels})

        '------------------------add init_labels to dev--------------------'
        dev = Dataset(self.transform, args.dev, is_test=False).build(batch_size, buckets)
        for sentence in dev.sentences:
            annotations = sentence.annotations
            sent_id = annotations[-2]
            init_labels = init_labels_train_dict[sent_id]
            sentence.transformed.update({'init_labels': init_labels})

        '------------------------add init_labels to test--------------------'
        test = Dataset(self.transform, args.test, is_test=True).build(batch_size, buckets)
        for sentence in test.sentences:
            annotations = sentence.annotations
            sent_id = annotations[-2]
            init_labels = init_labels_test_dict[sent_id]
            sentence.transformed.update({'init_labels': init_labels})

        test_dataset = copy.deepcopy(test).build(batch_size, buckets)
        logger.info(f"\n{'train:':6} {train}\n{'dev:':6} {dev}\n{'test:':6} {test}\n")

        if args.encoder == 'lstm':
            self.optimizer = Adam(self.model.parameters(), args.lr, (args.mu, args.nu), args.eps, args.weight_decay)
            self.scheduler = ExponentialLR(self.optimizer, args.decay**(1/args.decay_steps))
        else:
            from transformers import AdamW, get_linear_schedule_with_warmup
            steps = len(train.loader) * epochs // args.update_steps
            self.optimizer = AdamW(
                [{'params': p, 'lr': args.lr * (1 if n.startswith('encoder') else args.lr_rate)}
                 for n, p in self.model.named_parameters()],
                args.lr)
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, int(steps*args.warmup), steps)

        if dist.is_initialized():
            self.model = DDP(self.model, device_ids=[args.local_rank], find_unused_parameters=True)

        self.epoch, self.best_e, self.patience, self.best_metric, self.elapsed = 1, 1, patience, Metric(), timedelta()
        if self.args.checkpoint:
            self.optimizer.load_state_dict(self.checkpoint_state_dict.pop('optimizer_state_dict'))
            self.scheduler.load_state_dict(self.checkpoint_state_dict.pop('scheduler_state_dict'))
            set_rng_state(self.checkpoint_state_dict.pop('rng_state'))
            for k, v in self.checkpoint_state_dict.items():
                setattr(self, k, v)
            train.loader.batch_sampler.epoch = self.epoch

        for epoch in range(self.epoch, args.epochs + 1):
            start = datetime.now()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            self.args.mode = 'train'
            self._train(train.loader)
            self.args.mode = 'evaluate'
            loss, dev_metric = self._evaluate(dev.loader)
            logger.info(f"{'dev:':5} loss: {loss:.4f} - {dev_metric}")
            loss, test_metric = self._evaluate(test.loader)
            logger.info(f"{'test:':5} loss: {loss:.4f} - {test_metric}")

            # # 保存测试结果
            # preds = self._predict(test_dataset.loader)
            # for name, value in preds.items():
            #     setattr(test_dataset, name, value)
            # if self.args.test_result is not None and is_master():
            #     logger.info(f"Saving tested results to {self.args.test_result}")
            #     self.transform.save(self.args.test_result, test_dataset.sentences)

            t = datetime.now() - start
            self.epoch += 1
            self.patience -= 1
            self.elapsed += t

            if dev_metric > self.best_metric:
                self.best_e, self.patience, self.best_metric = epoch, patience, dev_metric
                if is_master():
                    self.save_checkpoint(args.path)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            if self.patience < 1:
                break
        parser = self.load(**args)
        loss, metric = parser._evaluate(test.loader)
        parser.save(args.path)

        logger.info(f"Epoch {self.best_e} saved")
        logger.info(f"{'dev:':5} {self.best_metric}")
        logger.info(f"{'test:':5} {metric}")
        logger.info(f"{self.elapsed}s elapsed, {self.elapsed / epoch}s/epoch")

    def evaluate(self, data, buckets=8, batch_size=5000, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        logger.info("Loading the data")
        '------------------------load init_labels of test predicted by existing parser--------------------'
        init_labels_test_dict = {}
        init_data = Dataset(self.transform, args.init_data, is_test=True).build(batch_size, buckets)
        for sentence in init_data.sentences:
            annotations = sentence.annotations
            sent_id = annotations[-2]
            init_labels = sentence.transformed['labels']
            init_labels_test_dict.update({sent_id: init_labels})

        dataset = Dataset(self.transform, data)
        dataset.build(batch_size, buckets)

        '------------------------add init_labels to test--------------------'
        for sentence in dataset.sentences:
            annotations = sentence.annotations
            sent_id = annotations[-2]
            init_labels = init_labels_test_dict[sent_id]
            sentence.transformed.update({'init_labels': init_labels})

        logger.info(f"\n{dataset}")

        logger.info("Evaluating the dataset")
        start = datetime.now()
        loss, metric = self._evaluate(dataset.loader)
        elapsed = datetime.now() - start
        logger.info(f"loss: {loss:.4f} - {metric}")
        logger.info(f"{elapsed}s elapsed, {len(dataset)/elapsed.total_seconds():.2f} Sents/s")

        return loss, metric

    def predict(self, data, pred=None, lang=None, buckets=8, batch_size=5000, prob=False, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        # self.transform.eval()
        self.transform.train()
        if args.prob:
            self.transform.append(Field('probs'))

        logger.info("Loading the data")
        '------------------------load init_labels of test predicted by existing parser--------------------'
        init_labels_test_dict = {}
        init_data = Dataset(self.transform, args.init_data, is_test=True).build(batch_size, buckets)
        for sentence in init_data.sentences:
            annotations = sentence.annotations
            sent_id = annotations[-2]
            init_labels = sentence.transformed['labels']
            init_labels_test_dict.update({sent_id: init_labels})

        dataset = Dataset(self.transform, data, lang=lang)
        dataset.build(batch_size, buckets)

        '------------------------add init_labels to test--------------------'
        for sentence in dataset.sentences:
            annotations = sentence.annotations
            sent_id = annotations[-2]
            init_labels = init_labels_test_dict[sent_id]
            sentence.transformed.update({'init_labels': init_labels})

        logger.info(f"\n{dataset}")

        logger.info("Making predictions on the dataset")
        start = datetime.now()
        preds = self._predict(dataset.loader)
        elapsed = datetime.now() - start

        for name, value in preds.items():
            setattr(dataset, name, value)
        if pred is not None and is_master():
            logger.info(f"Saving predicted results to {pred}")
            self.transform.save(pred, dataset.sentences)
        logger.info(f"{elapsed}s elapsed, {len(dataset) / elapsed.total_seconds():.2f} Sents/s")

        return dataset

    def _train(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _evaluate(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _predict(self, loader):
        raise NotImplementedError

    @classmethod
    def build(cls, path, **kwargs):
        raise NotImplementedError

    @classmethod
    def load(cls, path, reload=False, src='github', checkpoint=False, **kwargs):
        r"""
        Loads a parser with data fields and pretrained model parameters.

        Args:
            path (str):
                - a string with the shortcut name of a pretrained model defined in ``supar.MODEL``
                  to load from cache or download, e.g., ``'biaffine-dep-en'``.
                - a local path to a pretrained model, e.g., ``./<path>/model``.
            reload (bool):
                Whether to discard the existing cache and force a fresh download. Default: ``False``.
            src (str):
                Specifies where to download the model.
                ``'github'``: github release page.
                ``'hlt'``: hlt homepage, only accessible from 9:00 to 18:00 (UTC+8).
                Default: ``'github'``.
            checkpoint (bool):
                If ``True``, loads all checkpoint states to restore the training process. Default: ``False``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating training configs and initializing the model.

        Examples:
            >>> from supar import Parser
            >>> parser = Parser.load('biaffine-dep-en')
            >>> parser = Parser.load('./ptb.biaffine.dep.lstm.char')
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path if os.path.exists(path) else download(supar.MODEL[src].get(path, path), reload=reload))
        cls = supar.PARSER[state['name']] if cls.NAME is None else cls
        args = state['args'].update(args)
        model = cls.MODEL(**args)
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(args.device)
        transform = state['transform']
        parser = cls(args, model, transform)
        parser.checkpoint_state_dict = state['checkpoint_state_dict'] if args.checkpoint else None
        return parser

    def save(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        args = model.args
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {'name': self.NAME,
                 'args': args,
                 'state_dict': state_dict,
                 'pretrained': pretrained,
                 'transform': self.transform}
        torch.save(state, path, pickle_module=dill)

    def save_checkpoint(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        args = model.args
        checkpoint_state_dict = {k: getattr(self, k) for k in ['epoch', 'best_e', 'patience', 'best_metric', 'elapsed']}
        checkpoint_state_dict.update({'optimizer_state_dict': self.optimizer.state_dict(),
                                      'scheduler_state_dict': self.scheduler.state_dict(),
                                      'rng_state': get_rng_state()})
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {'name': self.NAME,
                 'args': args,
                 'state_dict': state_dict,
                 'pretrained': pretrained,
                 'checkpoint_state_dict': checkpoint_state_dict,
                 'transform': self.transform}
        torch.save(state, path, pickle_module=dill)
