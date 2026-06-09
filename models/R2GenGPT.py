import json
import os
import shutil

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from evalcap.bleu.bleu import Bleu
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from evalcap.rouge.rouge import Rouge
from models.vision_resampler import PerceiverResampler


class R2GenGPT(pl.LightningModule):
    """
    R2GenGPT mainline upgraded to DINOv2 + Perceiver Resampler + Qwen Instruct.

    SFT batches use samples["input_text"].
    DPO batches additionally use samples["chosen_text"] and samples["rejected_text"].
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        print(f"Loading vision encoder: {args.vision_model}")
        self.visual_encoder = AutoModel.from_pretrained(args.vision_model)
        vision_dim = self.visual_encoder.config.hidden_size
        if args.freeze_vm:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            print("Vision encoder frozen")
        else:
            print("Vision encoder trainable")

        print(f"Loading language model: {args.llm_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.llm = AutoModelForCausalLM.from_pretrained(
            args.llm_model,
            torch_dtype=torch.bfloat16 if args.precision.startswith("bf16") else torch.float16,
            trust_remote_code=True,
        )
        self.embed_tokens = self.llm.get_input_embeddings()

        if args.llm_use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.llm_r,
                lora_alpha=args.llm_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.llm_lora_target_modules.split(","),
            )
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()
        elif args.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

        llm_dim = self.llm.config.hidden_size
        self.resampler = PerceiverResampler(
            input_dim=vision_dim,
            output_dim=llm_dim,
            num_queries=args.resampler_num_queries,
            num_layers=args.resampler_num_layers,
            num_heads=args.resampler_num_heads,
            dropout=args.resampler_dropout,
        )
        self.visual_norm = nn.LayerNorm(llm_dim)
        self.end_sym = args.end_sym
        self.prompt = args.prompt
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        if args.delta_file is not None:
            map_location = "cpu" if not torch.cuda.is_available() else torch.device(f"cuda:{torch.cuda.current_device()}")
            state_dict = torch.load(args.delta_file, map_location=map_location, weights_only=False)["model"]
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f"Load checkpoint from {args.delta_file}")

    def score(self, ref, hypo):
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]
        if shutil.which("java") is not None:
            scorers.insert(2, (Meteor(), "METEOR"))
        final_scores = {}
        for scorer, method in scorers:
            score, _ = scorer.compute_score(ref, hypo)
            if isinstance(score, list):
                for metric_name, metric_score in zip(method, score):
                    final_scores[metric_name] = metric_score
            else:
                final_scores[method] = score
        return final_scores

    def encode_img(self, images):
        if len(images) > 0 and isinstance(images[0], list):
            sample_tokens = []
            sample_masks = []
            max_tokens = 0
            for sample_images in images:
                pixel_values = torch.stack(sample_images, dim=0)
                outputs = self.visual_encoder(pixel_values=pixel_values)
                tokens = outputs.last_hidden_state.reshape(1, -1, outputs.last_hidden_state.shape[-1])
                sample_tokens.append(tokens)
                sample_masks.append(torch.ones(tokens.shape[:2], dtype=torch.long, device=tokens.device))
                max_tokens = max(max_tokens, tokens.shape[1])

            padded_tokens = []
            padded_masks = []
            for tokens, mask in zip(sample_tokens, sample_masks):
                pad_len = max_tokens - tokens.shape[1]
                if pad_len > 0:
                    tokens = F.pad(tokens, (0, 0, 0, pad_len), value=0)
                    mask = F.pad(mask, (0, pad_len), value=0)
                padded_tokens.append(tokens)
                padded_masks.append(mask)

            tokens = torch.cat(padded_tokens, dim=0)
            masks = torch.cat(padded_masks, dim=0)
            visual_embeds = self.visual_norm(self.resampler(tokens, masks))
            visual_atts = torch.ones(visual_embeds.shape[:2], dtype=torch.long, device=visual_embeds.device)
            return visual_embeds, visual_atts

        image_tokens = []
        image_masks = []
        for image in images:
            outputs = self.visual_encoder(pixel_values=image)
            tokens = outputs.last_hidden_state
            image_tokens.append(tokens)
            image_masks.append(torch.ones(tokens.shape[:2], dtype=torch.long, device=tokens.device))

        tokens = torch.cat(image_tokens, dim=1)
        masks = torch.cat(image_masks, dim=1)
        visual_embeds = self.visual_norm(self.resampler(tokens, masks))
        visual_atts = torch.ones(visual_embeds.shape[:2], dtype=torch.long, device=visual_embeds.device)
        return visual_embeds, visual_atts

    def _prompt_parts(self):
        prompt = (
            "<|im_start|>user\n"
            "<ImageHere>\n"
            f"{self.prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return prompt.split("<ImageHere>")

    def _prompt_after_image(self, retrieved_context=None):
        if retrieved_context:
            retrieval_instruction = getattr(
                self.hparams,
                "retrieval_instruction",
                "Use the candidate reports as retrieval hints, but write the final report for the current image only.",
            )
            return (
                "Candidate reports from visually similar prior exams:\n"
                f"{retrieved_context}\n"
                f"{retrieval_instruction}\n"
                f"{self.prompt}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        return self._prompt_parts()[1]

    def prompt_wrap(self, img_embeds, retrieved_contexts=None):
        batch_size = img_embeds.shape[0]
        p_before, _ = self._prompt_parts()
        before_tokens = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        if retrieved_contexts is None:
            after_texts = [self._prompt_after_image()] * batch_size
        else:
            after_texts = [
                self._prompt_after_image(context)
                for context in retrieved_contexts
            ]
        after_tokens = self.tokenizer(
            after_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(img_embeds.device)
        before_embeds = self.embed_tokens(before_tokens.input_ids).expand(batch_size, -1, -1)
        before_attention = torch.ones(before_embeds.shape[:2], dtype=torch.long, device=img_embeds.device)
        image_attention = torch.ones(img_embeds.shape[:2], dtype=torch.long, device=img_embeds.device)
        after_embeds = self.embed_tokens(after_tokens.input_ids)
        wrapped = torch.cat([before_embeds, img_embeds, after_embeds], dim=1)
        attention = torch.cat([before_attention, image_attention, after_tokens.attention_mask], dim=1)
        return wrapped, attention

    def _tokenize_reports(self, reports, device):
        text = [report + self.end_sym for report in reports]
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False,
        ).to(device)

    def _build_inputs(self, samples, reports):
        image = samples["image"]
        img_embeds, _ = self.encode_img(image)
        prompt_embeds, prompt_attention = self.prompt_wrap(
            img_embeds,
            samples.get("retrieved_context", None),
        )
        report_tokens = self._tokenize_reports(reports, prompt_embeds.device)
        report_embeds = self.embed_tokens(report_tokens.input_ids)

        inputs_embeds = torch.cat([prompt_embeds, report_embeds], dim=1)
        attention_mask = torch.cat([prompt_attention, report_tokens.attention_mask], dim=1)

        prompt_targets = torch.full(
            prompt_attention.shape,
            fill_value=-100,
            dtype=torch.long,
            device=prompt_attention.device,
        )
        report_targets = report_tokens.input_ids.masked_fill(
            report_tokens.attention_mask == 0,
            -100,
        )
        labels = torch.cat([prompt_targets, report_targets], dim=1)
        return inputs_embeds, attention_mask, labels

    def _sequence_logps(self, samples, reports):
        inputs_embeds, attention_mask, labels = self._build_inputs(samples, reports)
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits[:, :-1, :].float()
        shifted_labels = labels[:, 1:]
        loss_mask = shifted_labels != -100
        safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
        token_logps = torch.gather(F.log_softmax(logits, dim=-1), dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logps = token_logps * loss_mask
        if self.hparams.dpo_average_logps:
            return token_logps.sum(-1) / loss_mask.sum(-1).clamp(min=1)
        return token_logps.sum(-1)

    def forward(self, samples):
        inputs_embeds, attention_mask, labels = self._build_inputs(samples, samples["input_text"])
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
        )
        return {"loss": outputs.loss}

    def dpo_loss(self, samples):
        chosen_logps = self._sequence_logps(samples, samples["chosen_text"])
        rejected_logps = self._sequence_logps(samples, samples["rejected_text"])
        policy_logits = chosen_logps - rejected_logps
        logits = policy_logits
        result = {"dpo_policy_margin": policy_logits.mean()}

        if getattr(self.hparams, "dpo_objective", "pairwise") == "reference":
            if "ref_chosen_logp" not in samples or "ref_rejected_logp" not in samples:
                raise KeyError(
                    "Reference DPO requires ref_chosen_logp/ref_rejected_logp in the batch. "
                    "Run precompute_reference_logps.py first."
                )
            ref_chosen = torch.as_tensor(samples["ref_chosen_logp"], device=chosen_logps.device, dtype=chosen_logps.dtype)
            ref_rejected = torch.as_tensor(samples["ref_rejected_logp"], device=rejected_logps.device, dtype=rejected_logps.dtype)
            ref_logits = ref_chosen - ref_rejected
            logits = policy_logits - ref_logits
            result["dpo_ref_margin"] = ref_logits.mean()

        dpo_loss = -F.logsigmoid(self.hparams.dpo_beta * logits).mean()
        reward_acc = (logits > 0).float().mean()
        loss = dpo_loss
        result.update({"dpo_loss": dpo_loss, "dpo_reward_acc": reward_acc, "dpo_margin": logits.mean()})

        sft_weight = getattr(self.hparams, "dpo_sft_loss_weight", 0.0)
        if sft_weight > 0:
            sft_loss = self(samples)["loss"]
            loss = loss + sft_weight * sft_loss
            result["dpo_sft_loss"] = sft_loss

        result["loss"] = loss
        return result

    def training_step(self, batch, batch_idx):
        if self.hparams.stage == "dpo":
            result = self.dpo_loss(batch)
        else:
            result = self(batch)
        self.log_dict(result, prog_bar=True, sync_dist=True)
        return result

    def on_train_epoch_end(self):
        if self.hparams.limit_val_batches == 0:
            self.save_train_checkpoint()

    def save_checkpoint(self, eval_res):
        if not self.trainer.is_global_zero:
            return
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            key: value.requires_grad
            for key, value in self.named_parameters()
            if value.requires_grad
        }
        state_dict = self.state_dict()
        for key in list(state_dict.keys()):
            if key not in param_grad_dic:
                del state_dict[key]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step": global_step,
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, "checkpoints"), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path,
            "checkpoints",
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(
                current_epoch,
                global_step,
                eval_res["Bleu_4"],
                eval_res["CIDEr"],
            ),
        )
        self.print(f"Saving checkpoint at step {global_step} to {save_to}.")
        torch.save(save_obj, save_to)

    def save_train_checkpoint(self):
        if not self.trainer.is_global_zero:
            return
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            key: value.requires_grad
            for key, value in self.named_parameters()
            if value.requires_grad
        }
        state_dict = self.state_dict()
        for key in list(state_dict.keys()):
            if key not in param_grad_dic:
                del state_dict[key]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step": global_step,
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, "checkpoints"), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path,
            "checkpoints",
            f"checkpoint_epoch{current_epoch}_step{global_step}_train.pth",
        )
        self.print(f"Saving train checkpoint at step {global_step} to {save_to}.")
        torch.save(save_obj, save_to)

    def _generate_batch(self, samples):
        image = samples["image"]
        img_embeds, _ = self.encode_img(image)
        prompt_embeds, prompt_attention = self.prompt_wrap(
            img_embeds,
            samples.get("retrieved_context", None),
        )
        outputs = self.llm.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=prompt_attention,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            no_repeat_ngram_size=self.hparams.no_repeat_ngram_size,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        hypo = [self.decode(output) for output in outputs]
        return hypo

    def validation_step(self, samples, batch_idx):
        hypo = self._generate_batch(samples)
        ref = samples["input_text"]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def decode(self, output_token):
        output_text = self.tokenizer.decode(output_token, skip_special_tokens=True)
        output_text = output_text.split("<|im_start|>assistant")[-1]
        output_text = output_text.split("<|im_end|>")[0].strip()
        return output_text.replace("<unk>", "").strip()

    def on_validation_epoch_end(self):
        ref, hypo, ids = [], [], []
        for item in self.val_step_outputs:
            ref.extend(item["ref"])
            hypo.extend(item["hypo"])
            ids.extend(item["id"])

        ref = {key: [value] for key, value in zip(ids, ref)}
        hypo = {key: [value] for key, value in zip(ids, hypo)}
        eval_res = self.score(ref=ref, hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, "result")
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}.json"), "w"))
        json.dump(ref, open(os.path.join(result_folder, "refs.json"), "w"))
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0 and val_score > self.val_score:
            self.save_checkpoint(eval_res)
            self.val_score = val_score
        self.val_step_outputs.clear()

    def test_step(self, samples, batch_idx):
        hypo = self._generate_batch(samples)
        ref = samples["input_text"]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def on_test_epoch_end(self):
        ref, hypo, ids = [], [], []
        for item in self.test_step_outputs:
            ref.extend(item["ref"])
            hypo.extend(item["hypo"])
            ids.extend(item["id"])

        ref = {key: [value] for key, value in zip(ids, ref)}
        hypo = {key: [value] for key, value in zip(ids, hypo)}
        eval_res = self.score(ref=ref, hypo=hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, "result")
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, "test_result.json"), "w"))
        json.dump(ref, open(os.path.join(result_folder, "test_refs.json"), "w"))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [param for param in self.parameters() if param.requires_grad],
            lr=self.hparams.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()
