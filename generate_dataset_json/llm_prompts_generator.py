from openai import OpenAI
import openai
import re
import json
import ast
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import llm_api_key

def safely_parse_dict(input_str: str) -> dict:
    """安全解析输入字符串中的字典定义，避免使用exec"""
    result = {}
    # 分割不同变量的定义（按行分割）
    lines = [line.strip() for line in input_str.split('\n') if line.strip()]
    
    for line in lines:
        if '=' in line:
            var_name, dict_str = line.split('=', 1)
            var_name = var_name.strip()
            dict_str = dict_str.strip()
            
            try:
                # 使用ast.literal_eval安全解析字典
                parsed_dict = ast.literal_eval(dict_str)
                if isinstance(parsed_dict, dict):
                    result[var_name] = parsed_dict
                else:
                    print(f"警告: {var_name} 的值不是字典，已跳过")
            except (SyntaxError, ValueError) as e:
                print(f"解析错误: {var_name} 的字典格式无效，错误信息: {e}")
    
    return result
def generate_defect_mappings(cat_dict, n=-1, api_key=llm_api_key, model="Pro/deepseek-ai/DeepSeek-V3"):
    """
    通过LLM生成工业缺陷分类映射的标准函数
    
    参数：
    n: 超类数量
    cat_dict: 原始类别字典
    api_key: OpenAI API密钥
    model: 使用的LLM模型
    
    返回：
    (global_defects_dict, local2global_id_map) 或错误信息
    """
    assert (n>2 or n ==-1) , "hyper class number should be greater than 2 or equal to -1"
    # 构造提示词模板
    prompt_template = """
        作为工业异常检测专家，请根据输入的类别字典和超类数n执行以下操作来提取超类：

        输入：
        1. 超类数n: {n}
        2. 类别字典cat_dict: {cat_dict}

        任务要求：
        1. 创建global_defects_dict字典：
        - 键：缺陷超类名（英文）
        - 值：[索引, 标准化描述]
        - 描述必须符合"an image of a/an [normal/anamaly] object with [特征]..."句式
        - 所描述的对象统一用“object”表示
        - 对于缺陷的描述尽可能详细，例如"Structural Damage":"An image showing structural damage including cracks, breaks, or deformation altering the object's original shape.", 
        - 需根据超类数合并语义相似的缺陷（如scratch/crack归为physical_damage）并生成n个超类
        - 若n=-1,则根据你的理解生成合适的超类

        2. 创建local2global_id_map字典：
        - 保持原始类别结构
        - 将每个缺陷映射到对应超类的索引
        

        输出格式要求：
        两个紧凑的Python字典（无换行）

        
    """
    postfix = """
            示例输出模式：
        global_defects_dict = {'Normal Condition':[0,...], 'Structural Breakage':[1,...], ...}
        local2global_id_map = {'candle':{'BACKGROUND':0, ...}, ...}
        """
    
    # 准备API请求
    messages = [{
        "role": "system",
        "content": "你是一个严谨的工业异常检测专家，严格按照指定格式输出Python字典和总结"
    }, {
        "role": "user",
        "content": prompt_template.format(n=n, cat_dict=json.dumps(cat_dict))+postfix
    }]

    try:
        print("正在创建OpenAI客户端...")
        client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
        print("客户端创建成功，LLM模型：{}".format(model))
        # 调用OpenAI API
        print("正在生成异常提示词..")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=2000,
            stream=False
            )
        # 提取响应内容
        content = response.choices[0].message.content.strip()
        
        # 解析输入字符串
        parsed_data = safely_parse_dict(content)

        # 提取目标字典
        global_defects_dict = parsed_data.get("global_defects_dict", {})
        local2global_id_map = parsed_data.get("local2global_id_map", {})
        

        return global_defects_dict, local2global_id_map

    except openai.AuthenticationError:
        return "Error: API密钥无效"
    except openai.InternalServerError as e:
        return f"Error:  - {str(e)}"
    except Exception as e:
        return f"Error: 未知错误 - {str(e)}"
def demo(api_key):
    client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

    response = client.chat.completions.create(
        model="Pro/deepseek-ai/DeepSeek-V3",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        temperature=0.1,
        max_tokens=2000,
        stream=False
    )
    print(response.choices[0].message.content)

def llm_prompts_generate(cat_dict,out_root,dataset,api_key=llm_api_key,n=-1):
    global_dict, local_map  = generate_defect_mappings(cat_dict=cat_dict,n=n,api_key=api_key)
    n = len(global_dict) if n == -1 else n
    all_categories = set()
    for types in cat_dict.values():
        for name in types:
            all_categories.add(name.lower())
    total_categories = len(all_categories)
    # data redundancy check
    redundancy_reduction = (total_categories - n) / total_categories
    summary = f"Total categories: {total_categories}, Hyper categories: {n}, Reduced redundancy : {redundancy_reduction:.2%}"
    print("Global Defects Dictionary:")
    print(json.dumps(global_dict, indent=2))
    print("\nLocal to Global Mapping:")
    print(json.dumps(local_map, indent=2))
    print("\nSummary:")
    print(summary)
    out_path = os.path.join(out_root, f"{dataset}_{n}cls.json")
    with open(out_path, 'w') as f:
        result_dict={
            "global_defects_dict":global_dict,
            "local2global_id_map":local_map,
            "summary":summary
        }
        json.dump(result_dict, f, indent=2)

if __name__ == "__main__":
    # 输入参数
    n = 10
    cat_dict = {'bottle': ['good', 'contamination', 'broken_small', 'broken_large'], 'cable': ['poke_insulation', 'cut_outer_insulation', 'missing_wire', 'good', 'missing_cable', 'bent_wire', 'cable_swap', 'cut_inner_insulation', 'combined'], 'capsule': ['scratch', 'poke', 'faulty_imprint', 'good', 'crack', 'squeeze'], 'carpet': ['good', 'hole', 'cut', 'thread', 'metal_contamination', 'color'], 'grid': ['good', 'broken', 'thread', 'metal_contamination', 'glue', 'bent'], 'hazelnut': ['good', 'hole', 'cut', 'print', 'crack'], 'leather': ['poke', 'good', 'fold', 'cut', 'color', 'glue'], 'metal_nut': ['scratch', 'good', 'flip', 'color', 'bent'], 'pill': ['scratch', 'faulty_imprint', 'pill_type', 'good', 'contamination', 'combined', 'color', 'crack'], 'screw': ['scratch_neck', 'good', 'scratch_head', 'manipulated_front', 'thread_side', 'thread_top'], 'tile': ['gray_stroke', 'oil', 'rough', 'good', 'glue_strip', 'crack'], 'toothbrush': ['good', 'defective'], 'transistor': ['good', 'misplaced', 'cut_lead', 'damaged_case', 'bent_lead'], 'wood': ['scratch', 'liquid', 'good', 'hole', 'combined', 'color'], 'zipper': ['broken_teeth', 'rough', 'good', 'fabric_interior', 'split_teeth', 'squeezed_teeth', 'fabric_border', 'combined']}
    # 调用函数
    result = generate_defect_mappings(n, cat_dict,llm_api_key)
    
    # 输出结果
    if isinstance(result, tuple):
        global_dict, local_map = result
        print("Global Defects Dictionary:")
        print(json.dumps(global_dict, indent=2))
        print("\nLocal to Global Mapping:")
        print(json.dumps(local_map, indent=2))
    else:
        print(result)


