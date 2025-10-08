import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import numpy as np


class IRTKnowledgeTracing(nn.Module):
    """
    基于IRT的知识追踪模型，实现图中的技术方案：
    1. LLM-based Knowledge State Estimation: h_{j+1} = LLM(q, {X_1, ..., X_j}, t_{j+1})
    2. IRT-based Prediction: θ_{j+1} = MLP^{(1)}(MeanPooling(h_{j+1}))
    3. Question-level and turn-level difficulty representations
    """
    
    def __init__(self, 
                 base_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
                 num_kcs: int = 100,
                 hidden_dim: int = 4096,
                 dropout: float = 0.1,
                 freeze_llm: bool = True):
        super(IRTKnowledgeTracing, self).__init__()
        
        self.base_model_name = base_model_name
        self.num_kcs = num_kcs
        self.hidden_dim = hidden_dim
        
        # LLM for knowledge state estimation
        self.llm = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": 0}
        )
        
        # 确保LLM的tokenizer有padding token
        if hasattr(self.llm, 'tokenizer') and self.llm.tokenizer.pad_token is None:
            self.llm.tokenizer.pad_token = self.llm.tokenizer.eos_token
        
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
        
        # 2-layer MLP for mapping knowledge state to M-dimensional vectors
        # θ_{j+1} = MLP^{(1)}(MeanPooling(h_{j+1}))
        self.knowledge_state_mapper = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # M-dimensional output
        ).to(torch.bfloat16)  # 确保与LLM数据类型一致
        
        # Question-level difficulty representation network
        self.difficulty_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Single difficulty value
        ).to(torch.bfloat16)

        self.q_weight = nn.Linear(hidden_dim, hidden_dim).to(torch.bfloat16)
        self.t_weight = nn.Linear(hidden_dim, hidden_dim).to(torch.bfloat16)
        
        
    def get_llm_hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        获取LLM的隐藏状态用于知识状态估计
        h_{j+1} = LLM(q, {X_1, ..., X_j}, t_{j+1})
        
        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            
        Returns:
            hidden_states: LLM最后一层的隐藏状态 [batch_size, seq_len, hidden_dim]
        """
        # 确保attention_mask的数据类型与LLM的查询张量匹配
        # 获取LLM的第一个参数的数据类型作为参考
        try:
            # 尝试获取LLM的第一个参数的数据类型
            llm_dtype = next(self.llm.parameters()).dtype
        except:
            # 如果无法获取，默认使用bfloat16
            llm_dtype = torch.bfloat16
        
        # 将attention_mask转换为与LLM匹配的数据类型
        if attention_mask.dtype != llm_dtype:
            attention_mask = attention_mask.to(llm_dtype)
        
        with torch.no_grad() if not self.llm.training else torch.enable_grad():
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # 返回最后一层的隐藏状态
            return outputs.hidden_states[-1]
    
    def compute_knowledge_state(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        计算知识状态表示
        θ_{j+1} = MLP^{(1)}(MeanPooling(h_{j+1}))
        
        Args:
            hidden_states: LLM隐藏状态 [batch_size, seq_len, hidden_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len] 或其他维度
            
        Returns:
            knowledge_state: 知识状态向量 [batch_size, output_dim]
        """
        # 处理不同维度的attention_mask
        if len(attention_mask.shape) > 2:
            # 如果是3D或更高维度的掩码，取第一个head或压缩维度
            if len(attention_mask.shape) == 3:
                # 3D掩码 [batch_size, seq_len, 1] -> [batch_size, seq_len]
                attention_mask = attention_mask.squeeze(-1)
            elif len(attention_mask.shape) == 5:
                # 5D掩码 [batch_size, 1, seq_len, seq_len, 1] -> [batch_size, seq_len]
                # 更安全的处理方式：先压缩维度，然后提取对角线
                try:
                    # 尝试提取对角线
                    mask_3d = attention_mask.squeeze(1).squeeze(-1)  # [batch_size, seq_len, seq_len]
                    if mask_3d.shape[1] == mask_3d.shape[2]:  # 确保是方阵
                        attention_mask = torch.diagonal(mask_3d, dim1=1, dim2=2)  # [batch_size, seq_len]
                    else:
                        # 如果不是方阵，取第一个维度
                        attention_mask = mask_3d[:, 0, :]  # [batch_size, seq_len]
                except:
                    # 如果对角线提取失败，使用第一个序列
                    attention_mask = attention_mask[:, 0, :, 0, 0]  # [batch_size, seq_len]
            else:
                # 其他高维情况，使用更安全的方法
                try:
                    # 尝试保持batch_size，压缩其他维度
                    batch_size = attention_mask.shape[0]
                    seq_len = hidden_states.shape[1]  # 从hidden_states获取正确的seq_len
                    
                    # 如果是4D，尝试提取合适的维度
                    if len(attention_mask.shape) == 4:
                        # [batch_size, seq_len, seq_len, 1] -> [batch_size, seq_len]
                        if attention_mask.shape[1] == seq_len:
                            attention_mask = attention_mask[:, :, 0, 0]  # 取第一个序列
                        else:
                            # 创建默认掩码
                            attention_mask = torch.ones(batch_size, seq_len, device=hidden_states.device, dtype=hidden_states.dtype)
                    else:
                        # 对于其他维度，创建默认掩码
                        attention_mask = torch.ones(batch_size, seq_len, device=hidden_states.device, dtype=hidden_states.dtype)
                except:
                    # 最后的回退：创建默认掩码
                    batch_size = hidden_states.shape[0]
                    seq_len = hidden_states.shape[1]
                    attention_mask = torch.ones(batch_size, seq_len, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # 确保attention_mask是2D且数据类型正确
        if len(attention_mask.shape) != 2:
            # 如果仍然不是2D，使用简单的方法
            batch_size = hidden_states.shape[0]
            seq_len = hidden_states.shape[1]
            attention_mask = torch.ones(batch_size, seq_len, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # 确保数据类型匹配
        attention_mask = attention_mask.to(hidden_states.dtype)
        
        # Mean pooling over sequence length
        # attention_mask: 1 for valid tokens, 0 for padding
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
        masked_hidden = hidden_states * mask_expanded
        pooled = masked_hidden.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-8)
        
        # Apply MLP to get M-dimensional knowledge state
        knowledge_state = self.knowledge_state_mapper(pooled)
        return knowledge_state
    
    def compute_difficulty_representations(self, question_hidden: torch.Tensor, turn_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算问题级和轮次级难度表示
        d^q = MeanPooling(LLM(q))
        
        Args:
            question_hidden: 问题对应的隐藏状态
            turn_hidden: 轮次对应的隐藏状态
            
        Returns:
            question_difficulty: 问题难度 [batch_size, 1]
            turn_difficulty: 轮次难度 [batch_size, 1]
        """
        # # 问题级难度
        # question_difficulty = self.question_difficulty_encoder(question_hidden)
        
        # # 轮次级难度
        # turn_difficulty = self.turn_difficulty_encoder(turn_hidden)

        w = torch.sigmoid(self.q_weight(question_hidden) + self.t_weight(turn_hidden)) # w = sigmoid(基本信息编码 + 时间信息编码)，每一维设置为0-1之间的数值
        difficulty = w * question_hidden + (1 - w) * turn_hidden
        final_difficulty = self.difficulty_encoder(difficulty)
        return final_difficulty
    
    def _extract_question_and_turn_hidden(self, 
                                        hidden_states: torch.Tensor, 
                                        attention_mask: torch.Tensor,
                                        input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        从隐藏状态中精确提取question和turn的表示
        使用特殊标记精确定位question和当前teacher turn部分
        
        Args:
            hidden_states: LLM隐藏状态 [batch_size, seq_len, hidden_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            input_ids: 输入token ids [batch_size, seq_len]
            
        Returns:
            question_hidden: 问题部分的隐藏状态 [batch_size, hidden_dim]
            turn_hidden: 当前轮次的隐藏状态 [batch_size, hidden_dim]
        """
        batch_size, seq_len = hidden_states.shape[:2]
        
        question_hidden_list = []
        turn_hidden_list = []
        
        # 获取tokenizer来解码token
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        for i in range(batch_size):
            # 获取当前序列的token ids（去除padding）
            seq_mask = attention_mask[i].bool()
            
            # 获取有效tokens
            if len(input_ids[i].shape) == 1 and len(seq_mask.shape) == 1:
                if seq_mask.shape[0] <= input_ids[i].shape[0]:
                    seq_tokens = input_ids[i][seq_mask]
                else:
                    seq_mask = seq_mask[:input_ids[i].shape[0]]
                    seq_tokens = input_ids[i][seq_mask]
            else:
                seq_tokens = input_ids[i]
            
            # 解码tokens
            if isinstance(seq_tokens, torch.Tensor):
                seq_tokens_cpu = seq_tokens.cpu().long()
                text = tokenizer.decode(seq_tokens_cpu, skip_special_tokens=False)
            else:
                text = tokenizer.decode(seq_tokens, skip_special_tokens=False)
            
            # 使用tokenizer编码特殊标记来精确定位token位置
            def find_token_positions(text: str, marker: str, tokenizer) -> tuple[int, int]:
                """查找标记在token序列中的精确位置"""
                if marker not in text:
                    return -1, -1
                
                # 编码完整文本
                full_tokens = tokenizer.encode(text, add_special_tokens=False)
                full_text = tokenizer.decode(full_tokens, skip_special_tokens=False)
                
                # 编码标记
                marker_tokens = tokenizer.encode(marker, add_special_tokens=False)
                marker_text = tokenizer.decode(marker_tokens, skip_special_tokens=False)
                
                # 在解码文本中查找标记位置
                marker_pos = full_text.find(marker_text)
                if marker_pos == -1:
                    return -1, -1
                
                # 通过逐token解码来精确定位token位置
                current_pos = 0
                start_token = -1
                end_token = -1
                
                for token_idx, token in enumerate(full_tokens):
                    token_text = tokenizer.decode([token], skip_special_tokens=False)
                    token_len = len(token_text)
                    
                    # 检查当前token是否包含标记的起始位置
                    if current_pos <= marker_pos < current_pos + token_len:
                        start_token = token_idx
                    
                    # 检查当前token是否包含标记的结束位置
                    if current_pos <= marker_pos + len(marker_text) <= current_pos + token_len:
                        end_token = token_idx + 1
                        break
                    
                    current_pos += token_len
                
                return start_token, end_token
            
            # 1. 精确定位question部分
            question_start_token = -1
            question_end_token = -1
            
            # 查找question标记
            if "[BEGIN QUESTION]" in text and "[END QUESTION]" in text:
                start_token, _ = find_token_positions(text, "[BEGIN QUESTION]", tokenizer)
                _, end_token = find_token_positions(text, "[END QUESTION]", tokenizer)
                if start_token != -1 and end_token != -1:
                    question_start_token = start_token
                    question_end_token = end_token
            elif "[BEGIN PROBLEM]" in text and "[END PROBLEM]" in text:
                start_token, _ = find_token_positions(text, "[BEGIN PROBLEM]", tokenizer)
                _, end_token = find_token_positions(text, "[END PROBLEM]", tokenizer)
                if start_token != -1 and end_token != -1:
                    question_start_token = start_token
                    question_end_token = end_token
            
            # 2. 精确定位当前teacher turn部分
            turn_start_token = -1
            turn_end_token = -1
            
            # 查找当前teacher turn标记（假设在数据加载时已添加特殊前缀）
            current_turn_markers = [
                "[CURRENT TEACHER TURN]"
            ]
            
            for marker in current_turn_markers:
                if marker in text:
                    start_token, end_token = find_token_positions(text, marker, tokenizer)
                    if start_token != -1:
                        turn_start_token = start_token
                        # 找到下一个标记或序列结束
                        remaining_text = text[text.find(marker) + len(marker):]
                        # 查找下一个可能的结束标记
                        end_markers = ["[END CURRENT TURN]", "\n\n"]
                        turn_end_pos = len(text)
                        for end_marker in end_markers:
                            pos = remaining_text.find(end_marker)
                            if pos != -1:
                                turn_end_pos = text.find(marker) + len(marker) + pos
                                break
                        
                        # 将结束位置转换为token位置
                        _, turn_end_token = find_token_positions(text[:turn_end_pos], "", tokenizer)
                        break
            
            # 3. 提取对应的隐藏状态
            seq_len_valid = min(seq_mask.sum().item(), seq_len)
            
            # 提取question隐藏状态 - 只使用精确标记
            if question_start_token != -1 and question_end_token != -1 and question_end_token > question_start_token:
                start_pos = max(0, min(question_start_token, seq_len_valid - 1))
                end_pos = max(start_pos + 1, min(question_end_token, seq_len_valid))
                question_hidden = hidden_states[i, start_pos:end_pos].mean(dim=0).to(hidden_states.dtype)
            else:
                # 如果没有找到question标记，跳过该批次
                print(f"Warning: Question markers not found in batch {i}, skipping...")
                continue
            
            # 提取turn隐藏状态 - 只使用精确标记
            if turn_start_token != -1:
                start_pos = max(0, min(turn_start_token, seq_len_valid - 1))
                if turn_end_token != -1:
                    end_pos = max(start_pos + 1, min(turn_end_token, seq_len_valid))
                else:
                    end_pos = seq_len_valid
                turn_hidden = hidden_states[i, start_pos:end_pos].mean(dim=0).to(hidden_states.dtype)
            else:
                # 如果没有找到turn标记，跳过该批次
                print(f"Warning: Current teacher turn markers not found in batch {i}, skipping...")
                continue
            
            question_hidden_list.append(question_hidden)
            turn_hidden_list.append(turn_hidden)
        
        # 堆叠结果
        if len(question_hidden_list) == 0:
            # 如果所有批次都被跳过，返回零张量
            print("Warning: All batches were skipped, returning zero tensors")
            batch_size, seq_len, hidden_dim = hidden_states.shape
            question_hidden = torch.zeros(batch_size, hidden_dim, dtype=hidden_states.dtype, device=hidden_states.device)
            turn_hidden = torch.zeros(batch_size, hidden_dim, dtype=hidden_states.dtype, device=hidden_states.device)
        else:
            question_hidden = torch.stack(question_hidden_list, dim=0)
            turn_hidden = torch.stack(turn_hidden_list, dim=0)
        
        return question_hidden, turn_hidden
    
    def predict_correctness(self, 
                          knowledge_state: torch.Tensor, 
                          difficulty: torch.Tensor) -> torch.Tensor:
        """
        IRT基础预测
        P(r_{j+1} = 1) = sigmoid(θ_{j+1} - d^q - d^t)
        
        Args:
            knowledge_state: 知识状态 [batch_size, output_dim] - θ_{j+1}
            question_difficulty: 问题难度 [batch_size, 1] - d^q
            turn_difficulty: 轮次难度 [batch_size, 1] - d^t
            
        Returns:
            predictions: 正确性预测概率 [batch_size, 1]
        """
        # 根据IRT公式：P(r_{j+1} = 1) = sigmoid(θ_{j+1} - d)
        
        # IRT公式：θ - d
        logits = knowledge_state - difficulty
        
        # 应用sigmoid函数
        predictions = torch.sigmoid(logits)
        
        return predictions
    
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入token ids [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            outputs: 包含各种预测结果的字典
        """
        # 确保数据类型正确
        input_ids = input_ids.long()
        # 保存原始attention_mask用于MeanPooling
        original_attention_mask = attention_mask.long()
        # 为LLM调用准备匹配LLM数据类型的attention_mask
        # 先转换为float，然后在get_llm_hidden_states中会转换为正确的类型
        llm_attention_mask = attention_mask.long().float()
        
        # 1. 获取LLM隐藏状态
        hidden_states = self.get_llm_hidden_states(input_ids, llm_attention_mask)
        
        # 2. 计算知识状态 - 平均池化
        knowledge_state = self.compute_knowledge_state(hidden_states, original_attention_mask)
        
        # 3. 计算难度表示
        # 更精确地提取question和turn的隐藏状态
        question_hidden, turn_hidden = self._extract_question_and_turn_hidden(
            hidden_states, original_attention_mask, input_ids
        )
        
        difficulty = self.compute_difficulty_representations(
            question_hidden, turn_hidden
        )
        
        # 4. 预测正确性
        correctness_pred = self.predict_correctness(
            knowledge_state, difficulty
        )
        
        outputs = {
            'correctness_prediction': correctness_pred,
            'knowledge_state': knowledge_state,
            'difficulty': difficulty,
        }
        
        return outputs


class IRTKnowledgeTracingLoss(nn.Module):
    """
    IRT知识追踪模型的损失函数
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.1):
        super(IRTKnowledgeTracingLoss, self).__init__()
        self.alpha = alpha  # 正确性预测损失权重
        self.beta = beta    # 知识组件预测损失权重
        self.bce_loss = nn.BCELoss()
        
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                labels: torch.Tensor,
                kc_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算损失
        
        Args:
            predictions: 模型预测结果
            labels: 正确性标签 [batch_size, 1]
            kc_labels: 知识组件标签 [batch_size, num_kcs]
            
        Returns:
            total_loss: 总损失
        """
        # 正确性预测损失，确保数据类型一致
        pred_tensor = predictions['correctness_prediction'].squeeze().float()
        # print(f"pred_tensor: {pred_tensor}")
        label_tensor = labels.float()
        # print(f"label_tensor: {label_tensor}")
        correctness_loss = self.bce_loss(pred_tensor, label_tensor)
        
        total_loss = correctness_loss
    
        
        return total_loss


def create_irt_kt_model(config: Dict) -> IRTKnowledgeTracing:
    """
    创建IRT知识追踪模型的工厂函数
    
    Args:
        config: 模型配置字典
        
    Returns:
        model: IRT知识追踪模型
    """

    print(f"base_model_name: {config.get('base_model_name', 'meta-llama/Meta-Llama-3.1-8B-Instruct')}")
    print(f"num_kcs: {config.get('num_kcs', 100)}")
    print(f"hidden_dim: {config.get('hidden_dim', 768)}")
    print(f"dropout: {config.get('dropout', 0.1)}")
    print(f"freeze_llm: {config.get('freeze_llm', True)}")

    return IRTKnowledgeTracing(
        base_model_name=config.get('base_model_name', 'meta-llama/Meta-Llama-3.1-8B-Instruct'),
        num_kcs=config.get('num_kcs', 100),
        hidden_dim=config.get('hidden_dim', 768),
        dropout=config.get('dropout', 0.1),
        freeze_llm=config.get('freeze_llm', True)
    )
