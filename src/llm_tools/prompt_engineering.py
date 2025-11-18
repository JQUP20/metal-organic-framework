"""
Prompt Engineering Toolkit for MOF Design

提示工程工具集，用于优化LLM在MOF设计任务中的表现

功能：
- 预定义的提示模板
- Few-shot示例管理
- Chain-of-Thought提示
- 提示优化和评估
"""

from typing import List, Dict, Optional
import json


class PromptTemplate:
    """提示词模板管理器"""

    # MOF信息提取模板
    EXTRACTION_TEMPLATE = """
You are an expert in Metal-Organic Frameworks (MOFs).
Extract the following information from the given text:

Text: {text}

Please extract:
1. MOF names mentioned
2. Metal centers (e.g., Zn, Cu, Zr)
3. Organic linkers (e.g., BDC, BTC)
4. Synthesis conditions (solvent, temperature, time)
5. Reported properties (surface area, gas uptake, etc.)
6. Applications mentioned

Return the result as a JSON object with these keys:
- mof_names: list of strings
- metals: list of strings
- linkers: list of strings
- synthesis: dict with keys [solvent, temperature, time]
- properties: dict with property names as keys
- applications: list of strings

JSON Output:
"""

    # MOF设计建议模板
    DESIGN_TEMPLATE = """
You are an expert MOF designer. Based on the following requirements,
suggest a MOF design with detailed justification.

Requirements:
{requirements}

Please provide:
1. Recommended metal center and why
2. Recommended organic linker and why
3. Expected topology/structure
4. Predicted properties
5. Synthesis strategy
6. Potential challenges

Think step by step and provide your reasoning for each choice.

Response:
"""

    # 性质预测模板
    PROPERTY_PREDICTION_TEMPLATE = """
Given the following MOF structure information, predict its properties.

MOF Information:
- Metal: {metal}
- Linker: {linker}
- Topology: {topology}
- Surface Area: {surface_area} m²/g

Please predict:
1. CO2 uptake at 298K, 1 bar (mmol/g)
2. Water stability (low/medium/high)
3. Thermal stability up to (°C)
4. Key advantages for gas storage
5. Potential limitations

Base your predictions on chemical principles and similar known MOFs.

Predictions:
"""

    # 文献总结模板
    LITERATURE_SUMMARY_TEMPLATE = """
Summarize the key findings about MOFs from the following papers:

Papers:
{papers}

Provide a concise summary including:
1. Main trends in MOF research
2. Most commonly studied metals and linkers
3. Typical synthesis methods
4. Performance benchmarks
5. Future directions mentioned

Summary:
"""

    @classmethod
    def get_template(cls, template_name: str) -> str:
        """获取预定义模板"""
        templates = {
            'extraction': cls.EXTRACTION_TEMPLATE,
            'design': cls.DESIGN_TEMPLATE,
            'property': cls.PROPERTY_PREDICTION_TEMPLATE,
            'summary': cls.LITERATURE_SUMMARY_TEMPLATE
        }
        return templates.get(template_name, "")


class FewShotManager:
    """
    Few-shot示例管理器

    管理和使用少样本学习示例以提高LLM性能
    """

    def __init__(self):
        self.examples = {
            'mof_extraction': [
                {
                    'input': "UiO-66 was synthesized from ZrCl4 and H2BDC in DMF at 120°C.",
                    'output': {
                        'mof_names': ['UiO-66'],
                        'metals': ['Zr'],
                        'linkers': ['BDC'],
                        'synthesis': {'solvent': 'DMF', 'temperature': '120', 'time': 'N/A'},
                        'properties': {},
                        'applications': []
                    }
                },
                {
                    'input': "HKUST-1 contains Cu paddle-wheel units and BTC linkers. "
                             "It shows 1850 m²/g surface area and 6.5 mmol/g CO2 uptake.",
                    'output': {
                        'mof_names': ['HKUST-1'],
                        'metals': ['Cu'],
                        'linkers': ['BTC'],
                        'synthesis': {},
                        'properties': {'surface_area': 1850, 'co2_uptake': 6.5},
                        'applications': []
                    }
                }
            ],
            'property_prediction': [
                {
                    'input': {'metal': 'Cu', 'linker': 'BTC', 'topology': 'tbo', 'surface_area': 1850},
                    'output': {
                        'co2_uptake': 6.5,
                        'water_stability': 'low',
                        'thermal_stability': 240,
                        'advantages': ['High CO2 uptake', 'Open metal sites'],
                        'limitations': ['Moisture sensitive']
                    }
                }
            ]
        }

    def get_examples(self, task: str, num_examples: int = 2) -> List[Dict]:
        """获取指定任务的few-shot示例"""
        return self.examples.get(task, [])[:num_examples]

    def format_few_shot_prompt(self, task: str, new_input: str, num_examples: int = 2) -> str:
        """构建包含few-shot示例的完整提示"""
        examples = self.get_examples(task, num_examples)

        prompt_parts = []

        # 添加few-shot示例
        for i, example in enumerate(examples, 1):
            prompt_parts.append(f"Example {i}:")
            prompt_parts.append(f"Input: {example['input']}")
            prompt_parts.append(f"Output: {json.dumps(example['output'], indent=2)}")
            prompt_parts.append("")

        # 添加新输入
        prompt_parts.append("Now, for the following input:")
        prompt_parts.append(f"Input: {new_input}")
        prompt_parts.append("Output:")

        return "\n".join(prompt_parts)


class ChainOfThoughtPrompt:
    """
    Chain-of-Thought提示生成器

    生成引导LLM逐步推理的提示词
    """

    @staticmethod
    def create_cot_prompt(question: str, domain: str = "MOF") -> str:
        """
        创建思维链提示

        参数:
            question: 问题
            domain: 领域（MOF, chemistry, materials）

        返回:
            cot_prompt: 包含思维链指令的提示
        """
        cot_prompt = f"""
Question: {question}

Please solve this step by step:

Step 1: Understand the question
- What is being asked?
- What information is provided?
- What information is missing?

Step 2: Recall relevant knowledge
- What principles apply here?
- What similar cases exist?
- What are the key factors?

Step 3: Reasoning process
- Work through the logic systematically
- Consider multiple perspectives
- Identify cause-and-effect relationships

Step 4: Answer
- State the conclusion clearly
- Provide supporting evidence
- Acknowledge uncertainties if any

Let's think through this carefully:
"""
        return cot_prompt

    @staticmethod
    def create_mof_design_cot(requirements: str) -> str:
        """创建MOF设计的思维链提示"""
        return f"""
Design a MOF based on these requirements:
{requirements}

Let's approach this systematically:

Step 1: Analyze requirements
- What is the target application?
- What properties are needed?
- Are there any constraints?

Step 2: Select metal center
- Which metals are suitable for this application?
- Consider cost, availability, stability
- Consider coordination chemistry

Step 3: Select organic linker
- What functional groups are beneficial?
- What linker length is optimal?
- Rigidity vs. flexibility trade-offs

Step 4: Predict structure
- Expected topology based on metal-linker combination
- Estimate pore size and surface area
- Consider interpenetration possibilities

Step 5: Evaluate feasibility
- Is the synthesis likely to succeed?
- What challenges might arise?
- How stable will the framework be?

Step 6: Final recommendation
- Summarize the design
- Justify each choice
- Suggest experimental validation approach

Now, let's work through each step:
"""


class PromptOptimizer:
    """
    提示词优化器

    自动优化和评估提示词效果
    """

    def __init__(self, llm_model=None):
        self.llm = llm_model
        self.prompt_variants = []
        self.performance_scores = []

    def generate_variants(self, base_prompt: str, num_variants: int = 3) -> List[str]:
        """生成提示词变体"""
        variants = [base_prompt]  # 原始版本

        # 变体1：添加角色设定
        variants.append(
            f"You are an expert MOF researcher with 20 years of experience.\n\n{base_prompt}"
        )

        # 变体2：添加详细指令
        variants.append(
            f"{base_prompt}\n\nPlease provide a detailed, well-structured response."
        )

        # 变体3：添加few-shot示例前缀
        variants.append(
            f"Here are examples of good responses:\n[Examples would go here]\n\n{base_prompt}"
        )

        return variants[:num_variants + 1]

    def evaluate_prompt(self, prompt: str, test_cases: List[Dict]) -> float:
        """
        评估提示词质量

        参数:
            prompt: 提示词
            test_cases: 测试用例 [{'input': ..., 'expected': ...}, ...]

        返回:
            score: 0-1之间的分数
        """
        if self.llm is None:
            # 模拟评分
            return 0.75 + (len(prompt) % 100) / 400

        # 实际评估逻辑
        correct = 0
        for case in test_cases:
            full_prompt = prompt.format(**case['input'])
            response = self.llm.generate(full_prompt)

            # 比较响应与期望输出
            if self._compare_responses(response, case['expected']):
                correct += 1

        return correct / len(test_cases) if test_cases else 0.0

    def _compare_responses(self, response: str, expected: str) -> bool:
        """比较LLM响应与期望输出"""
        # 简化的比较（实际应该更复杂）
        return expected.lower() in response.lower()

    def optimize(self, base_prompt: str, test_cases: List[Dict], iterations: int = 3):
        """
        优化提示词

        返回最佳提示词和其评分
        """
        best_prompt = base_prompt
        best_score = self.evaluate_prompt(base_prompt, test_cases)

        print(f"Initial prompt score: {best_score:.3f}")

        for i in range(iterations):
            variants = self.generate_variants(best_prompt, num_variants=2)

            for variant in variants:
                score = self.evaluate_prompt(variant, test_cases)
                print(f"  Variant {i+1} score: {score:.3f}")

                if score > best_score:
                    best_score = score
                    best_prompt = variant

        return best_prompt, best_score


# ===== 使用示例 =====

def example_prompt_templates():
    """示例：使用提示模板"""
    print("=" * 70)
    print("Prompt Templates Example")
    print("=" * 70)

    # 获取提取模板
    extraction_prompt = PromptTemplate.get_template('extraction')
    text = "MOF-5 was made from Zn and BDC in DEF at 100°C. Surface area: 3800 m²/g"

    prompt = extraction_prompt.format(text=text)
    print("\n提取信息的提示词:\n")
    print(prompt[:300] + "...")

    # 获取设计模板
    design_prompt = PromptTemplate.get_template('design')
    requirements = "High CO2 uptake, water stable, low cost"

    prompt = design_prompt.format(requirements=requirements)
    print("\n\nMOF设计的提示词:\n")
    print(prompt[:300] + "...")


def example_few_shot():
    """示例：Few-shot学习"""
    print("\n" + "=" * 70)
    print("Few-Shot Learning Example")
    print("=" * 70)

    manager = FewShotManager()

    # 新的输入
    new_input = "ZIF-8 is synthesized from Zn(NO3)2 and 2-methylimidazole in methanol."

    # 生成few-shot提示
    few_shot_prompt = manager.format_few_shot_prompt(
        task='mof_extraction',
        new_input=new_input,
        num_examples=2
    )

    print("\nFew-Shot提示词:\n")
    print(few_shot_prompt)


def example_chain_of_thought():
    """示例：Chain-of-Thought推理"""
    print("\n" + "=" * 70)
    print("Chain-of-Thought Example")
    print("=" * 70)

    cot = ChainOfThoughtPrompt()

    question = "Why is UiO-66 more water-stable than MOF-5?"

    cot_prompt = cot.create_cot_prompt(question)

    print("\nCoT提示词:\n")
    print(cot_prompt)


def example_prompt_optimization():
    """示例：提示词优化"""
    print("\n" + "=" * 70)
    print("Prompt Optimization Example")
    print("=" * 70)

    optimizer = PromptOptimizer()

    base_prompt = "Describe the MOF {mof_name}."

    test_cases = [
        {
            'input': {'mof_name': 'UiO-66'},
            'expected': 'zirconium-based'
        },
        {
            'input': {'mof_name': 'HKUST-1'},
            'expected': 'copper'
        }
    ]

    print("\n优化提示词...\n")
    best_prompt, best_score = optimizer.optimize(base_prompt, test_cases, iterations=2)

    print(f"\n最佳提示词 (score={best_score:.3f}):\n")
    print(best_prompt)


if __name__ == "__main__":
    example_prompt_templates()
    example_few_shot()
    example_chain_of_thought()
    example_prompt_optimization()

    print("\n" + "=" * 70)
    print("✓ Prompt engineering tools demonstrated!")
    print("=" * 70)
