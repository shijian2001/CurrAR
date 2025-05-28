import base64
import io
import re
from typing import Any

import numpy as np
import torch
from openai import OpenAI
from torchvision.transforms import ToPILImage
from transformers import Pipeline

default_qa_template = "Based on the image, answer the following question by strictly selecting only one option from the given choices.\nQuestion: {question}\nAnswer:"


def is_answer_match(ans: str, should: str) -> bool:
    ans = ans.lower().strip()
    should = should.lower().strip()

    option_part = should.split(")")[0] + ")"  # "(b)"
    desc_part = should.split(") ")[1]  # "7 years"
    option_letter = option_part[1]  # "b"

    # 构造正则表达式，匹配：
    # 1. 整个 should
    # 2. 仅选项部分（如 "(b)"）
    # 3. 仅描述部分（如 "7 years"）
    # 4. 仅选项字母（如 "b"），且必须是独立字母
    pattern = rf"^({re.escape(should)}|{re.escape(option_part)}|{re.escape(desc_part)}|\b{option_letter}\b)$"
    return bool(re.fullmatch(pattern, ans))


class VQAScorer:
    def __init__(self, template: str = default_qa_template) -> None:
        self.template = template

    def calc_score(self, vqa_pipeline: Pipeline, images: torch.Tensor, prompts: tuple[str], metadata: tuple[Any]):
        batch_size = len(images)
        scores = [0.0] * len(images)
        to_pil = ToPILImage()

        vqa_samples = []

        for i, image in enumerate(images):
            all_qa: list[dict[str, str]] = metadata[i]["qa"]["relation"] + metadata[i]["qa"]["attribute"]
            if isinstance(image, torch.Tensor):
                pil_image = to_pil(image.to(torch.float))
            for each_qa in all_qa:
                vqa_samples.append(
                    (
                        [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": pil_image,
                                    },
                                    {
                                        "type": "text",
                                        "text": self.template.format(question=each_qa["question"]),
                                    },
                                ],
                            }
                        ],
                        each_qa["answer"],
                        len(all_qa),
                        i,
                    )
                )

        # calc reward in batch
        for i in range(0, len(vqa_samples), batch_size):
            q_with_image, answers, qa_lens, img_indices = zip(*vqa_samples[i : i + batch_size])
            responses = vqa_pipeline(text=q_with_image, max_new_tokens=512, return_full_text=False)  # type: ignore

            for response, answer, qa_len, img_idx in zip(responses, answers, qa_lens, img_indices):
                generated_answer = response[0]["generated_text"]
                # print(generated_answer)
                scores[img_idx] += (1 / qa_len) if is_answer_match(generated_answer, answer) else 0
            # print(scores)

        return np.array(scores), None  # type: ignore


class OpenAIVQAScorer:
    """
    OpenAI兼容API的VQA评分器，使用OpenAI SDK进行推理
    """

    def __init__(
        self,
        api_base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy-key",
        model_name: str = "llava-v1.5-7b-hf",
        template: str = default_qa_template,
        timeout: int = 30,
    ) -> None:
        """
        初始化OpenAI兼容的VQA评分器

        Args:
            api_base_url: API基础URL
            api_key: API密钥
            model_name: 模型名称
            template: 问答模板
            timeout: 请求超时时间（秒）
        """
        self.model_name = model_name
        self.template = template
        self.to_pil = ToPILImage()

        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=api_key, base_url=api_base_url, timeout=timeout)

    def _image_to_base64(self, image: torch.Tensor) -> str:
        """将torch tensor图像转换为base64编码字符串"""
        if isinstance(image, torch.Tensor):
            pil_image = self.to_pil(image.to(torch.float))
        else:
            pil_image = image

        # 将PIL图像转换为base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"

    def _make_api_request(self, messages: list[dict], max_tokens: int = 512) -> str:
        """使用OpenAI SDK发送API请求并返回生成的文本"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,  # 确保结果一致性
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"API请求失败: {e}")
            return ""

    def calc_score(self, vqa_pipeline: Pipeline, images: torch.Tensor, prompts: tuple[str], metadata: tuple[Any]):
        """
        计算VQA分数，与VQAScorer保持相同的接口
        注意：vqa_pipeline参数在这里不会被使用，但保持接口一致性
        """
        batch_size = len(images)
        scores = [0.0] * len(images)

        # 收集所有VQA样本
        vqa_samples = []
        for i, image in enumerate(images):
            all_qa: list[dict[str, str]] = metadata[i]["qa"]["relation"] + metadata[i]["qa"]["attribute"]

            # 转换图像为base64
            image_base64 = self._image_to_base64(image)

            for each_qa in all_qa:
                # 构造OpenAI格式的消息
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_base64}},
                            {"type": "text", "text": self.template.format(question=each_qa["question"])},
                        ],
                    }
                ]

                vqa_samples.append(
                    (
                        messages,
                        each_qa["answer"],
                        len(all_qa),
                        i,
                    )
                )

        # 批量处理API请求
        for messages, answer, qa_len, img_idx in vqa_samples:
            generated_answer = self._make_api_request(messages)

            # 计算分数
            if is_answer_match(generated_answer, answer):
                scores[img_idx] += 1 / qa_len

        return np.array(scores), None  # type: ignore
