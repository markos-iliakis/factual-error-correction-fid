#
# Copyright (c) 2019-2021 James Thorne.
#
# This file is part of factual error correction.
# See https://jamesthorne.co.uk for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import json
import os
import time
import warnings
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import List, Tuple, Dict
import torch.nn.functional as F
from torch import nn
import types
from collections import OrderedDict

from pathlib import Path

from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from error_correction.modelling.base_transformer import BaseTransformer
from error_correction.modelling.dataset.error_correction_dataset import (
    ErrorCorrectionSeq2SeqDataset,
)
from error_correction.modelling.lightning_base import add_generic_args
from error_correction.modelling.reader.mask_based_correction_reader import (
    MaskBasedCorrectionReader,
)
from error_correction.modelling.reader.supervised_correction_reader import (
    SupervisedCorrectionReader,
)
from error_correction.modelling.utils import (
    is_truthy,
    SARI_KEYS,
    use_task_specific_params,
    pickle_save,
    freeze_params,
    lmap,
    flatten_list,
    save_json,
    calculate_sari,
    post_clean,
    assert_all_frozen,
)


class ErrorCorrectionModule(BaseTransformer):
    mode = "error-correction"
    val_metric = SARI_KEYS[2]
    loss_names = ["loss"]
    metric_names = SARI_KEYS

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        use_task_specific_params(self.model, "error-correction")

        self.metrics_save_path = Path(self.output_dir) / (
            "metrics.json"
            if not self.hparams.do_predict
            else "metrics_test_{}.json".format(os.path.basename(self.hparams.test_file))
        )

        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)

        self.dataset_kwargs: dict = dict(
            max_source_length=self.hparams.max_source_length,
            mutation_source=is_truthy(self.hparams.mutation_source),
            mutation_target=is_truthy(self.hparams.mutation_target),
        )

        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {
            k: v if v >= 0 else None for k, v in n_observations_per_split.items()
        }

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }

        self.data_paths = {
            "train": self.hparams.train_file,
            "val": self.hparams.val_file,
            "test": self.hparams.test_file,
        }

        assert (
            self.target_lens["train"] <= self.target_lens["val"]
        ), f"target_lens: {self.target_lens}"
        assert (
            self.target_lens["train"] <= self.target_lens["test"]
        ), f"target_lens: {self.target_lens}"

        if self.hparams.freeze_embeds:
            self.freeze_embeds()

        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

        self.num_workers = hparams.num_workers
        self.decoder_start_token_id = None
        self.dataset_class = ErrorCorrectionSeq2SeqDataset
        self.wiki_reader = self.get_reader(self.hparams.reader, self.hparams.do_predict)
        self.wrap_encoder()

    def forward_(self, **kwargs):
        print('Forward_________________________________')
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(ErrorCorrectionModule, self).forward(
            **kwargs
        )

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    # added code and attention mask in arguments
    def forward(self, input_ids, attention_mask, **kwargs):
        print('FORWARD NORMAL !!!!!!!!!!!!!!!!!!!!!!!')
        print(input_ids.dim(), input_ids.size())
        if input_ids is not None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.model.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
            print('size changed : ')
            print(input_ids.size())
        if attention_mask is not None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

        return self.model.forward(input_ids, attention_mask, return_dict=True, **kwargs)

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, use_cache, decoder_start_token_id, max_length):
        self.model.encoder.n_passages = input_ids.size(1)

        return self.model.generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
            use_cache=use_cache,
            decoder_start_token_id=decoder_start_token_id
            # **model_kwargs
        )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.model.encoder = EncoderWrapper(self.model.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.model.encoder = self.model.encoder.encoder
        block = []
        for mod in self.model.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.model.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.model.load_state_dict(state_dict)
        self.wrap_encoder()

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores / ntokens
        return scores

    # End of added functions for T5 class

    def get_reader(self, name, test):
        labels = set()
        if self.hparams.labels == "all":
            labels.add("SUPPORTS")
            labels.add("REFUTES")
        else:
            labels.add(self.hparams.labels.upper())

        print("XXX Reader labels {}".format(self.hparams.labels))
        print("XXX labels", labels)

        if name == "supervised":
            return SupervisedCorrectionReader(labels, test)
        elif name == "mask":
            return MaskBasedCorrectionReader(labels, test)
        else:
            raise RuntimeError(f"Unknown reader {name}")

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                freeze_params(d.embed_tokens)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.tokenizer.pad_token_id
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["decoder_input_ids"]

        decoder_input_ids = self.model._shift_right(tgt_ids)

        outputs = self(
            src_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        )
        # assert 1 == -1
        lm_logits = outputs[0]

        # Same behavior as modeling_bart.py, besides ignoring pad_token_id
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

        assert lm_logits.shape[-1] == self.model.config.vocab_size
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

        return (loss,)

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)
        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        return {"loss": loss_tensors[0], "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def _generative_step(self, batch: dict) -> dict:
        pad_token_id = self.tokenizer.pad_token_id
        (
            source_ids,
            source_mask,
            y,
            original_ids,
        ) = ErrorCorrectionSeq2SeqDataset.trim_seq2seq_batch(batch, pad_token_id)
        t0 = time.time()
        generated_ids = self.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
            max_length=self.hparams.val_max_target_length,
        )
        gen_time = (time.time() - t0) / source_ids.shape[0]
        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(y)
        original = self.ids_to_clean_text(original_ids)
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        metrics: Dict = self.calc_generative_metrics(original, preds, target)
        summ_len = lmap(len, generated_ids)
        base_metrics.update(
            gen_time=[gen_time],
            summ_len=summ_len,
            preds=preds,
            target=target,
            metadata=batch["metadata"],
            **metrics,
        )
        return base_metrics

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {
            k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names
        }
        loss = losses["loss"]

        rouges = {
            k: np.array(flatten_list([x[k] for x in outputs])).mean().item()
            for k in self.metric_names + ["gen_time", "summ_len"]
        }
        rouge_tensor: torch.FloatTensor = torch.tensor(rouges[self.val_metric]).type_as(
            loss
        )
        rouges.update({k: v.item() for k, v in losses.items()})
        losses.update(rouges)
        metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        metrics["step_count"] = self.step_count

        self.save_metrics(metrics, prefix)  # writes to self.metrics_save_path

        preds = flatten_list([x["preds"] for x in outputs])
        targets = flatten_list([x["target"] for x in outputs])
        metadata = flatten_list([x["metadata"] for x in outputs])

        self.save_predictions(preds, targets, metadata, prefix)

        return {
            "log": metrics,
            "preds": preds,
            f"{prefix}_loss": loss,
            f"{prefix}_{self.val_metric}": rouge_tensor,
        }

    def save_metrics(self, latest_metrics, type_path) -> None:
        self.metrics[type_path].append(latest_metrics)
        save_json(self.metrics, self.metrics_save_path)

    def save_predictions(self, predictions, actual, metadata, type_path) -> None:
        with open(
            Path(self.output_dir)
            / (
                "predictions_set_{}_epoch_{}_steps_{}_.jsonl".format(
                    type_path, self.trainer.current_epoch, self.step_count
                )
                if not self.hparams.do_predict
                else "final_predictions_set_{}_file_{}".format(
                    type_path, os.path.basename(self.hparams.test_file)
                )
            ),
            "w+",
        ) as f:
            for p, a, m in zip(predictions, actual, metadata):
                f.write(
                    json.dumps({"prediction": p, "actual": a, "metadata": m}) + "\n"
                )

    def calc_generative_metrics(self, originals, preds, target) -> Dict:
        # return calculate_rouge([p.replace(": ","") for p in preds], [p.replace("correction: ","") for p in target])

        return calculate_sari(
            originals, [post_clean(p) for p in preds], [post_clean(p) for p in target]
        )

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path) -> ErrorCorrectionSeq2SeqDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        instance_generator = self.wiki_reader.read(self.data_paths[type_path])

        dataset = self.dataset_class(
            self.tokenizer,
            instance_generator=instance_generator,
            n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        dataset.instances = dataset.instances[:1000]
        return dataset

    def get_dataloader(
        self, type_path: str, batch_size: int, shuffle: bool = False
    ) -> DataLoader:
        dataset = self.get_dataset(type_path)
        sampler = None
        if self.hparams.sortish_sampler and type_path == "train":
            assert self.hparams.gpus <= 1  # TODO: assert earlier
            sampler = dataset.make_sortish_sampler(batch_size)
            shuffle = False

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
            sampler=sampler,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader(
            "train", batch_size=self.hparams.train_batch_size, shuffle=True
        )
        t_total = (
            (
                len(dataloader.dataset)
                // (self.hparams.train_batch_size * max(1, self.hparams.gpus))
            )
            // self.hparams.accumulate_grad_batches
            * float(self.hparams.max_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=t_total,
        )
        if max(scheduler.get_last_lr()) > 0:
            warnings.warn("All learning rates are 0")
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)

        parser.add_argument("--reader", default="wiki", type=str)
        parser.add_argument("--train_file", required=True, type=str)
        parser.add_argument("--val_file", required=True, type=str)
        parser.add_argument("--test_file", required=False, type=str)
        parser.add_argument(
            "--max_source_length",
            default=512,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=256,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=256,  # these defaults are optimized for CNNDM. For xsum, see README.md.
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=256,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--sortish_sampler", action="store_true", default=False)
        parser.add_argument(
            "--logger_name",
            type=str,
            choices=["default", "wandb", "wandb_shared"],
            default="default",
        )
        parser.add_argument(
            "--n_train",
            type=int,
            default=-1,
            required=False,
            help="# examples. -1 means use all.",
        )
        parser.add_argument(
            "--n_val",
            type=int,
            default=-1,
            required=False,
            help="# examples. -1 means use all.",
        )
        parser.add_argument(
            "--n_test",
            type=int,
            default=-1,
            required=False,
            help="# examples. -1 means use all.",
        )
        parser.add_argument("--mutation_source", required=True)
        parser.add_argument("--mutation_target", required=True)
        parser.add_argument(
            "--labels", type=str, choices=["supports", "refutes", "all"], required=True
        )

        return parser


# Added Classes and functions
class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """

    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids=None, attention_mask=None, **kwargs, ):
        print('MYENCODER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        print(input_ids.shape, self.n_passages, passage_length)
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        # assert 1 == -1

        # return (outputs[0].view(bsz, self.n_passages * passage_length, -1),) + outputs[1:]
        return BaseModelOutputWithPastAndCrossAttentions(outputs[0].view(bsz, self.n_passages * passage_length, -1), *outputs[1:])


class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """

    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output


def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block


def cross_attention_forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
):
    """
    This only works for computing cross attention over the input
    """
    assert (kv is not None)
    assert (head_mask is None)
    assert (position_bias is not None or self.has_relative_attention_bias)

    bsz, qlen, dim = input.size()
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.size(1)

    q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    else:
        k, v = past_key_value_state

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)

    if mask is not None:
        scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores

    attn = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn = F.dropout(attn, p=self.dropout, training=self.training)

    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output
