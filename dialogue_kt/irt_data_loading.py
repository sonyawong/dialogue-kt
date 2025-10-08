"""
IRT Knowledge Tracing 数据加载模块

根据proposal要求，专门为IRT模型设计的数据加载器：
- 输入：问题q，历史对话{X_1, ..., X_j}，当前轮次t_{j+1}
- 输出：知识状态θ_{j+1}，问题难度d^q，轮次难度d^t，KC信息
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
import json

from dialogue_kt.kt_data_loading import apply_annotations
from dialogue_kt.utils import device
from dialogue_kt.prompting import get_dataset_desc, COMTA_DIALOGUE_DESC, MATHDIAL_DIALOGUE_DESC, EEDI_DESC


class IRTKTDataset(Dataset):
    """
    IRT知识追踪数据集
    
    根据proposal设计：
    - 每个样本包含：问题q，历史对话，当前轮次t_{j+1}，KC信息，标签
    - 支持序列化的对话历史处理
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 tokenizer, 
                 args,
                 kc_dict: Dict[str, int],
                 skip_first_turn: bool = False):
        self.data = []
        self.tokenizer = tokenizer
        self.args = args
        self.kc_dict = kc_dict
        self.num_kcs = len(kc_dict)
        
        failed = 0
        for idx, sample in data.iterrows():
            dialogue = apply_annotations(sample)
            if not dialogue:
                failed += 1
                continue
            
            # 构建序列化的对话历史
            dialogue_sequence = self._build_dialogue_sequence(dialogue, sample, skip_first_turn)
            if dialogue_sequence:
                self.data.extend(dialogue_sequence)
        
        print(f"{failed} / {len(data)} dialogues failed processing")
        print(f"Number of IRT data points: {len(self.data)}")
    

    def _build_dialogue_sequence(self, dialogue, sample, skip_first_turn: bool) -> List[Dict]:
        """
        构建序列化的对话历史
        
        Args:
            dialogue: 处理后的对话数据
            sample: 原始样本数据
            skip_first_turn: 是否跳过第一个轮次
            
        Returns:
            dialogue_sequence: 序列化的对话数据列表
        """
        sequence = []
        is_first_turn = True
        
        for turn_idx, turn in enumerate(dialogue):
            # 跳过第一个轮次（如果指定）
            if skip_first_turn and is_first_turn:
                is_first_turn = False
                continue
            
            # 检查是否有下一个轮次（学生回答）
            if turn_idx + 1 >= len(dialogue):
                break
                
            next_turn = dialogue[turn_idx + 1]
            # 确保当前轮次是教师问题，下一个轮次是学生回答且有correctness标签
            if (turn.get("teacher") and turn.get("teacher") != "" and 
                next_turn.get("correct") is not None):
                
                # 构建历史对话上下文（包含到当前轮次之前，使用学生回答的知识点）
                historical_context = self._build_historical_context_with_student_kcs(dialogue[:turn_idx])
                
                # 获取下一个轮次（学生回答）的KC信息
                student_kcs = next_turn["kcs"]
                kc_ids = self._get_kc_ids(student_kcs)
                
                # 构建输入序列：问题 + 历史对话 + 当前教师问题
                input_sequence = self._build_input_sequence(sample, historical_context, turn, student_kcs)
                
                sequence.append({
                    "dialogue_idx": sample.name if hasattr(sample, 'name') else 0,
                    "turn_idx": turn_idx,
                    "input_sequence": input_sequence,
                    "current_turn": turn,  # 当前教师问题
                    "next_turn": next_turn,  # 下一个学生回答
                    "kc_ids": kc_ids,
                    "label": next_turn["correct"],  # 学生回答的正确性
                    "question_text": self._extract_question_text(sample),
                    "turn_text": turn.get("teacher", "")
                })
            
            is_first_turn = False
        return sequence
    
    def _build_historical_context_with_student_kcs(self, dialogue_history: List[Dict]) -> str:
        """
        构建历史对话上下文
        参照get_dialogue_text的格式，不包含知识点标签
        
        Args:
            dialogue_history: 历史对话列表
            
        Returns:
            context: 格式化的历史对话文本
        """
        context_parts = []
        for turn in dialogue_history:
            if turn.get("teacher") and turn["teacher"] != "":
                context_parts.append(f"Teacher Turn {turn['turn']}: {turn['teacher']}")
            if turn.get("student") and turn["student"] != "":
                context_parts.append(f"Student Turn {turn['turn']}: {turn['student']}")
        
        return "\n".join(context_parts)
    
    def _build_input_sequence(self, sample, historical_context: str, current_turn: Dict, student_kcs: List[str]) -> str:
        """
        构建输入序列：参照prompting.py的KT模型格式
        使用Llama3对话格式
        
        Args:
            sample: 原始样本
            historical_context: 历史对话上下文
            current_turn: 当前轮次
            student_kcs: 学生回答的知识点
            
        Returns:
            input_sequence: 完整的输入序列（Llama3对话格式）
        """
        # 构建对话消息列表
        messages = []
        
        # 系统提示 - 参照KT_SYSTEM_PROMPT
        system_prompt = """You are an experienced math teacher. You are given a dialogue between a student and teacher where {desc} Your job is to predict if the student has a particular knowledge component at the current point in the dialogue. Please follow these instructions carefully when making your prediction:
- The student will need to possess this knowledge component in order to respond correctly to the teacher's most recent question.
- Use previous information in the dialogue to determine if the student has this knowledge component or not.
- Only respond with a single word, "True" or "False".""".format(desc=get_dataset_desc(self.args))

        messages.append({"role": "system", "content": system_prompt})
        
        # 构建用户消息内容 - 参照kt_user_prompt的格式
        user_content = ""
        
        # 添加问题上下文 - 参照get_eedi_context
        if hasattr(sample, 'meta_data') and 'question' in sample.meta_data:
            question_text = sample.meta_data['question'].strip()
            user_content += f"[BEGIN QUESTION]\n{question_text}\n[END QUESTION]\n\n"
            
            # 添加正确答案信息（如果存在）
            if 'correct_answer' in sample.meta_data:
                correct_answer = sample.meta_data['correct_answer'].strip()
                user_content += f"[BEGIN CORRECT ANSWER]\n{correct_answer}\n[END CORRECT ANSWER]\n\n"
        
        # 添加历史对话和当前轮次 - 参照get_dialogue_text的格式
        if historical_context.strip():
            user_content += f"[BEGIN DIALOGUE]\n{historical_context}\n[END DIALOGUE]\n\n"
        
        # 添加当前教师问题（标记为当前轮次）
        current_turn_text = current_turn.get("teacher", "")
        if current_turn_text and current_turn_text != "":
            turn_num = current_turn.get("turn", 0)
            user_content += f"[CURRENT TEACHER TURN] Teacher Turn {turn_num}: {current_turn_text} [END CURRENT TURN]\n\n"
        
        # 添加知识组件信息 - 参照kt_user_prompt的Knowledge Component格式
        if student_kcs:
            kcs_text = ", ".join(student_kcs)
            user_content += f"Knowledge Components: {kcs_text}"
        
        messages.append({"role": "user", "content": user_content})
        
        # 使用tokenizer的对话模板
        input_sequence = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return input_sequence
    
    def _extract_question_text(self, sample) -> str:
        """
        提取问题文本
        
        Args:
            sample: 原始样本
            
        Returns:
            question_text: 问题文本
        """
        # 从样本中提取问题文本
        if hasattr(sample, 'question') and sample.question:
            return sample.question
        elif hasattr(sample, 'meta_data') and 'question' in sample.meta_data:
            return sample.meta_data['question']
        # else:
        #     # 如果没有明确的问题字段，使用对话的第一部分作为问题
        #     dialogue = sample.get('dialogue', [])
        #     if dialogue and len(dialogue) > 0:
        #         first_turn = dialogue[0]
        #         return first_turn.get('teacher', '') or first_turn.get('student', '')
        #     return "Unknown question"
    
    def _get_kc_ids(self, kcs: List[str]) -> torch.Tensor:
        """
        将KC文本转换为ID
        
        Args:
            kcs: KC文本列表
            
        Returns:
            kc_ids: KC ID张量 [num_kcs]
        """
        kc_ids = torch.zeros(self.num_kcs, dtype=torch.long)
        for kc in kcs:
            if kc in self.kc_dict:
                kc_ids[self.kc_dict[kc]] = 1
        
        return kc_ids
    
    def __getitem__(self, index: int) -> Dict:
        # print(f"getting_data: {self.data[index]}")
        return self.data[index]
    
    def __len__(self) -> int:
        return len(self.data)


class IRTKTCollator:
    """
    IRT数据整理器
    
    负责将批次数据整理为模型输入格式
    """
    
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        整理批次数据
        
        Args:
            batch: 批次数据列表
            
        Returns:
            batch_data: 整理后的批次数据
        """
        # 提取输入序列
        input_sequences = [sample["input_sequence"] for sample in batch]
        
        # 提取KC IDs
        kc_ids = torch.stack([sample["kc_ids"] for sample in batch])
        
        # 提取标签
        labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.float)
        
        # 提取元数据
        meta_data = batch
        
        # 对输入序列进行tokenization
        tokenized = self.tokenizer(
            input_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        return {
            "input_ids": tokenized.input_ids.to(device),
            "attention_mask": tokenized.attention_mask.to(device),
            "kc_ids": kc_ids.to(device),
            "labels": labels.to(device),
            "meta_data": meta_data
        }


def create_irt_kt_dataloader(dataset: IRTKTDataset, 
                           collator: IRTKTCollator, 
                           batch_size: int, 
                           shuffle: bool = True) -> DataLoader:
    """
    创建IRT数据加载器
    
    Args:
        dataset: IRT数据集
        collator: 数据整理器
        batch_size: 批次大小
        shuffle: 是否打乱数据
        
    Returns:
        dataloader: 数据加载器
    """
    return DataLoader(
        dataset, 
        collate_fn=collator, 
        batch_size=batch_size, 
        shuffle=shuffle
    )


def prepare_irt_kt_data(args, fold: int) -> Tuple[IRTKTDataset, IRTKTDataset, IRTKTDataset]:
    """
    准备IRT训练数据
    
    Args:
        args: 训练参数
        fold: 数据折叠
        
    Returns:
        train_dataset, val_dataset, test_dataset: 训练、验证、测试数据集
    """
    from dialogue_kt.data_loading import load_annotated_data, load_kc_dict
    
    # 加载数据
    train_df, val_df, test_df = load_annotated_data(args, fold)
    
    # 加载KC字典
    kc_dict = load_kc_dict(args)
    
    # 创建tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建数据集
    train_dataset = IRTKTDataset(train_df, tokenizer, args, kc_dict)
    val_dataset = IRTKTDataset(val_df, tokenizer, args, kc_dict)
    test_dataset = IRTKTDataset(test_df, tokenizer, args, kc_dict, skip_first_turn=True)
    
    return train_dataset, val_dataset, test_dataset


def get_kc_dict_from_dataset(args) -> Dict[str, int]:
    """
    从数据集获取KC字典
    
    Args:
        args: 训练参数
        
    Returns:
        kc_dict: KC名称到ID的映射
    """
    from dialogue_kt.data_loading import load_kc_dict
    return load_kc_dict(args)
