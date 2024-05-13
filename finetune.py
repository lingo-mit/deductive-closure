import gc
import os
import sys
import threading

import numpy as np
import psutil
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)
from util import eval_perf
from peft import LoraConfig, TaskType, get_peft_model
import ipdb
import fire


# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


def main(
    creak_train: str,
    creak_dev: str,
    outdir: str,
    epoch: str,
    eval_method: str,
    max_new_tokens: int = 10,
    creak_test: str = None,
    eval_modes: str = "dev",
    lr: float = 1e-4,
    model_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf",
    seed: int = 42,
    batch_size: int = 1,
    max_length: int = 256,
    text_column: str = "input_prompt",
    label_column: str = "label",
):
    print("Dev path: ", creak_dev)
    print("Test path: ", creak_test)
    print("Train path: ", creak_train)

    accelerator = Accelerator()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    num_epochs = epoch
    set_seed(seed)

    if creak_test is None:
        creak_test = creak_dev

    dataset = load_dataset(
        "csv", data_files={"train": creak_train, "dev": creak_dev, "test": creak_test}
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    def preprocess_function(examples):
        batch_size = len(examples[text_column])
        inputs = [str(x) for x in examples[text_column]]
        targets = [str(x) for x in examples[label_column]]
        model_inputs = tokenizer(inputs, return_token_type_ids=False)
        labels = tokenizer(targets, return_token_type_ids=False)

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (
                max_length - len(sample_input_ids)
            ) + model_inputs["attention_mask"][i]
            labels["input_ids"][i] = [-100] * (
                max_length - len(sample_input_ids)
            ) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(
                model_inputs["input_ids"][i][:max_length]
            )
            model_inputs["attention_mask"][i] = torch.tensor(
                model_inputs["attention_mask"][i][:max_length]
            )
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def test_preprocess_function(examples):
        batch_size = len(examples[text_column])
        # inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]] # TODO: CREAK
        inputs = [str(x) for x in examples[text_column]]
        model_inputs = tokenizer(inputs, return_token_type_ids=False)

        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (
                max_length - len(sample_input_ids)
            ) + model_inputs["attention_mask"][i]
            model_inputs["input_ids"][i] = torch.tensor(
                model_inputs["input_ids"][i][:max_length]
            )
            model_inputs["attention_mask"][i] = torch.tensor(
                model_inputs["attention_mask"][i][:max_length]
            )
        return model_inputs

    # Create weighted random sampler so that the model sees equal number of examples from each class.
    labels_c = np.array(dataset["train"][label_column])
    unique_labels = np.unique(labels_c)
    weight_map = {}
    for t in unique_labels:
        weight_map[t] = 1.0 / len(np.where(labels_c == t)[0])
    samples_weight = np.array([weight_map[t] for t in labels_c])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.WeightedRandomSampler(
        samples_weight, len(samples_weight)
    )

    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()

    train_dataset = processed_datasets["train"]

    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            test_preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
    eval_dataset = processed_datasets["dev"]
    test_dataset = processed_datasets["test"]

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        batch_size=batch_size,
        pin_memory=True,
        sampler=sampler,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=batch_size,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=default_data_collator,
        batch_size=batch_size,
        pin_memory=True,
    )

    print(next(iter(train_dataloader)))

    # creating model
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    (
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
    )
    accelerator.print(model)

    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    if type(eval_modes) is str:
        eval_modes = [eval_modes]

    # Evaluation before training.
    for eval_mode in eval_modes:
        eval_model_dataloader = (
            eval_dataloader if eval_mode == "dev" else test_dataloader
        )
        model.eval()
        eval_preds = []
        with TorchTracemalloc() as tracemalloc:
            for _, batch in enumerate(tqdm(eval_model_dataloader)):
                batch = {k: v for k, v in batch.items() if k != "labels"}
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model).generate(
                        **batch,
                        synced_gpus=is_ds_zero_3,
                        max_new_tokens=max_new_tokens,
                    )  # synced_gpus=True for DS-stage 3
                outputs = accelerator.pad_across_processes(
                    outputs, dim=1, pad_index=tokenizer.pad_token_id
                )
                preds = accelerator.gather_for_metrics(outputs)
                preds = preds[:, max_length:].detach().cpu().numpy()
                eval_preds.extend(
                    tokenizer.batch_decode(preds, skip_special_tokens=True)
                )

        assert len(eval_preds) == len(
            dataset[eval_mode][label_column]
        ), f"{len(eval_preds)} != {len(dataset[eval_mode][label_column])}"

        # Calculate accuracy.
        correct_l = eval_perf[eval_method](eval_preds, dataset[eval_mode][label_column])
        accuracy = sum(correct_l) / len(correct_l)

        accelerator.print(f"{eval_mode}_{accuracy=}")
        accelerator.print(f"{eval_mode}_{eval_preds[:10]=}")
        accelerator.print(f"{eval_mode}_{dataset[eval_mode][label_column][:10]=}")

        # Save predictions with accuracies for analysis. Denote eval_mode and epoch in the filename.
        eval_df = dataset[eval_mode].to_pandas()
        eval_df["preds"] = eval_preds
        eval_df["correct"] = correct_l
        eval_df.to_csv(f"{outdir}/eval_{eval_mode}_start.csv", index=False)

    # Training
    for epoch in range(num_epochs):
        with TorchTracemalloc() as tracemalloc:
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print(
            "GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin))
        )
        accelerator.print(
            "GPU Memory consumed at the end of the train (end-begin): {}".format(
                tracemalloc.used
            )
        )
        accelerator.print(
            "GPU Peak Memory consumed during the train (max-begin): {}".format(
                tracemalloc.peaked
            )
        )
        accelerator.print(
            "GPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print(
            "CPU Memory before entering the train : {}".format(
                b2mb(tracemalloc.cpu_begin)
            )
        )
        accelerator.print(
            "CPU Memory consumed at the end of the train (end-begin): {}".format(
                tracemalloc.cpu_used
            )
        )
        accelerator.print(
            "CPU Peak Memory consumed during the train (max-begin): {}".format(
                tracemalloc.cpu_peaked
            )
        )
        accelerator.print(
            "CPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

        for eval_mode in eval_modes:
            eval_model_dataloader = (
                eval_dataloader if eval_mode == "dev" else test_dataloader
            )
            model.eval()
            eval_preds = []
            with TorchTracemalloc() as tracemalloc:
                for _, batch in enumerate(tqdm(eval_model_dataloader)):
                    batch = {k: v for k, v in batch.items() if k != "labels"}
                    with torch.no_grad():
                        outputs = accelerator.unwrap_model(model).generate(
                            **batch,
                            synced_gpus=is_ds_zero_3,
                            max_new_tokens=max_new_tokens,
                        )  # synced_gpus=True for DS-stage 3
                    outputs = accelerator.pad_across_processes(
                        outputs, dim=1, pad_index=tokenizer.pad_token_id
                    )
                    preds = accelerator.gather_for_metrics(outputs)
                    preds = preds[:, max_length:].detach().cpu().numpy()
                    eval_preds.extend(
                        tokenizer.batch_decode(preds, skip_special_tokens=True)
                    )

            correct_l = eval_perf[eval_method](
                eval_preds, dataset[eval_mode][label_column]
            )
            accuracy = sum(correct_l) / len(correct_l)
            accelerator.print(f"{eval_mode}_{accuracy=}")
            accelerator.print(f"{eval_mode}_{eval_preds[:10]=}")
            accelerator.print(f"{eval_mode}_{dataset[eval_mode][label_column][:10]=}")

            # Save predictions with accuracies for analysis. Denote eval_mode and epoch in the filename.
            eval_df = dataset[eval_mode].to_pandas()
            eval_df["preds"] = eval_preds
            eval_df["correct"] = correct_l
            eval_df.to_csv(f"{outdir}/eval_{eval_mode}_{epoch}.csv", index=False)

    accelerator.wait_for_everyone()

    # Save the model locally.
    model_name_or_path = model_name_or_path.replace("/", "_")
    model_path = (
        f"{outdir}/{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
    )
    model.save_pretrained(model_path)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(main)
