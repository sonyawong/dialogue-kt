#!/usr/bin/env python3
"""
EEDI数据格式转换脚本 - 生成MathDial格式
将EEDI测试数据转换为与MathDial相同格式的训练数据
"""

import pandas as pd
import ast
import json
import argparse
from typing import List, Dict, Any
import re
import numpy as np


def add_content(cur: str, new: str) -> str:
    """合并对话内容，处理重复和标点"""
    new = new.strip()
    if not cur:
        return new
    if cur == new:  # 有时turns在MathDial中会重复
        return cur
    if not cur.endswith((".", "!", "?")):
        cur += "."
    return cur + " " + new


def create_conversation_string(turns: List[dict]) -> str:
    """创建conversation字符串格式"""
    conversation_parts = []
    for i, turn in enumerate(turns):
        role = "Teacher" if turn["role"] == "tutor" else "Student"
        content = turn["content"]
        conversation_parts.append(f"{role}: {content}")
    return "|EOM|".join(conversation_parts)

def process_dialogue_with_turn_numbers(turns: List[dict]) -> List[dict]:
    """将原始turns转换为带轮次编号的dialogue格式"""
    if not turns:
        return []
    
    dialogue = []
    turn_number = 1
    
    for turn in turns:
        role = "teacher" if turn["role"] == "tutor" else "student"
        content = turn["content"]
        
        # 创建带轮次编号的对话项
        dialogue_item = {
            "turn": turn_number,
            role: content,
        }
        
        dialogue.append(dialogue_item)
        turn_number += 1
    
    return dialogue


def create_annotation(kcs_data: Dict[str, Any], correctness_data: Dict[str, Any]) -> Dict[str, Any]:
    """创建annotation字段，按对话顺序合并KCs和correctness"""
    annotation = {}
    
    # 确保数据是字典类型
    if not isinstance(kcs_data, dict):
        kcs_data = {}
    if not isinstance(correctness_data, dict):
        correctness_data = {}
    
    # 标准化键名，确保格式一致
    kcs_data = {(k if isinstance(k, str) and "turn" in k.lower() else f"turn {k}"): v for k,v in kcs_data.items()} 
    correctness_data = {(k if isinstance(k, str) and "turn" in k.lower() else f"turn {k}"): v for k,v in correctness_data.items()}
    
    # 获取所有轮次并排序
    all_turns = set(kcs_data.keys()) | set(correctness_data.keys())
    sorted_turns = sorted(all_turns, key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)
    
    # 用于跟踪最近的KCs
    last_kcs = []
    
    for turn_key in sorted_turns:
        # 从eedi_kcs中提取KCs
        kcs = kcs_data.get(turn_key, {}).get('kcs', []) if isinstance(kcs_data.get(turn_key), dict) else []
        
        # 如果当前轮次有KCs，更新last_kcs
        if kcs:
            last_kcs = kcs
        
        # 从correctness中提取正确性信息
        correct_value = None
        if isinstance(correctness_data.get(turn_key), dict):
            correct_value = correctness_data.get(turn_key, {}).get('correct', None)
        
        # 如果当前轮次有correctness但没有KCs，使用最近的KCs
        if correct_value is not None and not kcs:
            kcs = last_kcs
        
        annotation[turn_key] = {
            "correct": correct_value,
            "kcs": kcs
        }
    
    return annotation


def create_mathdial_format_row(idx: int, row: pd.Series, dialogue: List[dict], 
                              annotation: Dict[str, Any], conversation: str) -> Dict[str, Any]:
    """创建MathDial格式的数据行"""
    
    # 解析现有数据
    question_annotation = ast.literal_eval(row['question_annotation']) if pd.notna(row['question_annotation']) else {}
    
    # 提取问题信息
    subjects = ast.literal_eval(row['subjects']) if pd.notna(row['subjects']) else []
    freeform_persona = row['freeform_persona'] if pd.notna(row['freeform_persona']) else ""
    
    # 创建meta_data
    meta_data = {
        "question": row['question'],
        "correct_solution": question_annotation.get("solution", ""),
        "incorrect_solution": "",  # EEDI数据中没有学生错误解答
        "self_correctness": "Yes",  # 默认值，因为EEDI数据中大部分是正确解答
        "self_typical_confusion": 3.0,  # 默认值
        "self_typical_interactions": 3.0  # 默认值
    }
    
    # 安全解析key字段
    try:
        if pd.notna(row["key"]) and str(row["key"]).lower() != 'nan' and str(row["key"]).strip() != '':
            parsed_key = ast.literal_eval(str(row["key"]))
            if isinstance(parsed_key, (list, tuple)) and len(parsed_key) >= 2:
                qid = parsed_key[0]
                scenario = parsed_key[1]
            else:
                qid = f"eedi_{idx}"
                scenario = 1
        else:
            qid = f"eedi_{idx}"
            scenario = 1
    except:
        qid = f"eedi_{idx}"
        scenario = 1
    
    # 创建MathDial格式的行
    mathdial_row = {
        "index": idx,
        "qid": eval(row['key'])[0],  # 生成唯一的qid
        "scenario": eval(row['key'])[1],  # 默认scenario
        "question": row["question"],
        "ground_truth": question_annotation.get("correct_option", ""),
        "student_incorrect_solution": "",  # EEDI数据中没有
        "student_profile": freeform_persona,
        "teacher_described_confusion": "mathematical concept confusion",  # 默认值
        "self-correctness": "Yes",
        "self-typical-confusion": 3.0,
        "self-typical-interactions": 3.0,
        "conversation": conversation,
        "dialogue": dialogue,
        "meta_data": meta_data,
        "domain_prompt": f"[BEGIN PROBLEM]\n{row['question']}\n[END PROBLEM]",
        "domain_annotation_raw": f"### Summary of the Dialogue\nThis dialogue involves mathematical problem solving with the question: {row['question']}",
        "domain_annotation": [subj[0] for subj in subjects[:2]] if subjects else ["Mathematics"],
        "cluster_prompt": f"[BEGIN PROBLEM]\n{row['question']}\n[END PROBLEM]",
        "cluster_annotation_raw": f"**Summary of Each Turn in the Dialogue:**\nMathematical problem solving dialogue.",
        "cluster_annotation": [subj[0] for subj in subjects[:3]] if subjects else ["Mathematics"],
        "standard_prompt": f"[BEGIN PROBLEM]\n{row['question']}\n[END PROBLEM]",
        "standard_annotation_raw": f"### Summary of Each Turn in the Dialogue\nMathematical problem solving with various learning objectives.",
        "standard_annotation": {turn_key: turn_data["kcs"] for turn_key, turn_data in annotation.items()},
        "correctness_prompt": f"[BEGIN PROBLEM]\n{row['question']}\n[END PROBLEM]",
        "correctness_annotation_raw": f"### Turn-by-Turn Analysis\nAnalysis of student responses and correctness.",
        "correctness_annotation": {turn_key: turn_data["correct"] for turn_key, turn_data in annotation.items()},
        "annotation": annotation
    }
    
    return mathdial_row

def convert_eedi_to_mathdial_format(input_file: str, output_file: str):
    """主转换函数"""
    print(f"正在读取文件: {input_file}")
    df = pd.read_csv(input_file)
    print(f"读取了 {len(df)} 行数据")
    
    converted_data = []
    failed_count = 0
    
    for idx, row in df.iterrows():
        try:
            # 解析turns字段
            turns = ast.literal_eval(row['turns'])
            
            # 处理对话，添加轮次编号
            dialogue = process_dialogue_with_turn_numbers(turns)
            if not dialogue:
                failed_count += 1
                continue
            
            # 创建conversation字符串
            conversation = create_conversation_string(turns)
            
            # 解析KCs数据（从eedi_kcs字段）
            kcs_data = ast.literal_eval(row['eedi_kcs'])
            
            # 解析correctness数据
            correctness_data = ast.literal_eval(row['correctness'])
            
            # 创建annotation
            if "error" in kcs_data or "error" in correctness_data:
                failed_count += 1
                continue
            annotation = create_annotation(kcs_data, correctness_data)
            
            # 创建MathDial格式的行
            mathdial_row = create_mathdial_format_row(idx, row, dialogue, annotation, conversation)
            
            converted_data.append(mathdial_row)
            
            if (idx + 1) % 1000 == 0:
                print(f"已处理 {idx + 1} 行...")
                
        except Exception as e:
            print(f"处理第 {idx} 行时出错: {e}")
            failed_count += 1
            continue
    
    print(f"转换完成！成功转换 {len(converted_data)} 行，失败 {failed_count} 行")
    
    # 保存转换后的数据
    converted_df = pd.DataFrame(converted_data)
    converted_df.to_csv(output_file, index=False)
    print(f"已保存到: {output_file}")
    
    # 打印一些统计信息
    total_turns = sum(len(row['dialogue']) for row in converted_data)
    total_annotated_turns = sum(len([t for t in row['annotation'].values() if t['kcs']]) for row in converted_data)
    print(f"总对话轮数: {total_turns}")
    print(f"有标注的轮数: {total_annotated_turns}")
    
    return converted_df

def main():
    parser = argparse.ArgumentParser(description="将EEDI数据转换为MathDial格式")
    parser.add_argument("--input", "-i", required=True, help="输入CSV文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出CSV文件路径")
    
    args = parser.parse_args()
    
    convert_eedi_to_mathdial_format(args.input, args.output)

if __name__ == "__main__":
    main()
