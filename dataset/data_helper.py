
import os
import json
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor


class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)


    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0] 

    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
    def clean_report(self, report):
        # clean Iu-xray reports
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # clean MIMIC-CXR reports
        else:
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .' 
        # report = ' '.join(report.split()[:self.args.max_txt_len])
        return report


    def parse(self, features):
        to_return = {'id': features['id']}
        report = features.get("report", "")
        report = self.clean_report(report)
        to_return['input_text'] = report
        retrieved_reports = features.get("retrieved_reports", [])
        if retrieved_reports:
            cleaned = []
            for idx, candidate in enumerate(retrieved_reports, start=1):
                candidate = self.clean_report(candidate)
                if candidate:
                    cleaned.append(f"{idx}. {candidate}")
            to_return["retrieved_context"] = "\n".join(cleaned)
        # chest x-ray images
        images = []
        for image_path in features['image_path']:
            with Image.open(os.path.join(self.args.base_dir, image_path)) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                images.append(image)
        to_return["image"] = images
        return to_return


    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.meta = json.load(open(args.annotation, 'r'))
        self.meta = self.meta[split]
        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


class PreferenceDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        if args.preference_file is None:
            raise ValueError("--preference_file is required for --stage dpo")
        self.preferences = self._load_preferences(args.preference_file)
        annotation = json.load(open(args.annotation, 'r'))
        self.meta_by_id = {}
        for split_name in ["train", "val", "test"]:
            for item in annotation.get(split_name, []):
                self.meta_by_id[str(item["id"])] = item
        self.parser = FieldParser(args)

    def _load_preferences(self, path):
        if path.endswith(".jsonl"):
            with open(path, "r") as f:
                return [json.loads(line) for line in f if line.strip()]
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("pairs", data.get("data", []))
        return data

    def __len__(self):
        return len(self.preferences)

    def __getitem__(self, index):
        pref = self.preferences[index]
        study_id = str(pref["id"])
        if study_id not in self.meta_by_id:
            raise KeyError(f"Preference id {study_id} not found in annotation file")
        parsed = self.parser.transform_with_parse(self.meta_by_id[study_id])
        parsed["chosen_text"] = self.parser.clean_report(pref["chosen"])
        parsed["rejected_text"] = self.parser.clean_report(pref["rejected"])
        if "ref_chosen_logp" in pref and "ref_rejected_logp" in pref:
            parsed["ref_chosen_logp"] = float(pref["ref_chosen_logp"])
            parsed["ref_rejected_logp"] = float(pref["ref_rejected_logp"])
        elif getattr(self.args, "dpo_require_ref_logps", False):
            raise KeyError(
                "Preference row is missing ref_chosen_logp/ref_rejected_logp; "
                "run precompute_reference_logps.py first or disable --dpo_require_ref_logps."
            )
        return parsed


def create_datasets(args):
    if args.stage == "dpo":
        train_dataset = PreferenceDataset(args, 'train')
    else:
        train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset


