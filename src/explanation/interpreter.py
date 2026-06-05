"""
模板化 NLG 人格解读生成
======================
将模型预测的概率值和关键词归因转化为自然语言人格描述。

输出结构参照 README §4.4:
{
    "interpretation": { "EI": "...", "SN": "...", ... },
    "summary": "..."
}

Usage:
    >>> interpreter = MBTIInterpreter()
    >>> text = interpreter.interpret(probabilities, keywords, mbti_type="INFP")
    >>> print(text["summary"])
"""

from __future__ import annotations


class MBTIInterpreter:
    """
    基于模板的人格解读生成器。
    将概率 → 人格描述文本，支持中英文。
    """

    LANG = "zh"  # 默认中文，可切换 "en"

    # ================================================================
    # 人格维度描述库
    # ================================================================

    DIM_TEMPLATES = {
        "EI": {
            "E": {
                "zh": (
                    "你在外向维度上倾向明显（E: {pct:.0f}%）。你的文本展现出对外部世界的关注，"
                    "喜欢与人交流、分享想法，在社交互动中获得能量。你更倾向于通过交谈来理清思路，"
                    "在群体环境中感到自在。"
                ),
                "en": (
                    "You show a clear preference for Extraversion (E: {pct:.0f}%). "
                    "Your writing reflects an outward focus — you engage with the external world, "
                    "enjoy sharing ideas, and draw energy from social interaction."
                ),
            },
            "I": {
                "zh": (
                    "你在内向维度上倾向明显（I: {pct:.0f}%）。你的文本显示出对内心世界的关注，"
                    "偏好独处、深度思考，在安静环境中恢复能量。你倾向于先思考再表达，"
                    "在文字中流露出内省的特质。"
                ),
                "en": (
                    "You show a clear preference for Introversion (I: {pct:.0f}%). "
                    "Your writing reflects an inward focus — you value solitude, deep reflection, "
                    "and recharge through quiet introspection."
                ),
            },
        },
        "SN": {
            "S": {
                "zh": (
                    "你在感觉维度上倾向明显（S: {pct:.0f}%）。你关注具体事实和实际细节，"
                    "偏好可验证的信息和实际经验，对当下的现实世界有敏锐的观察力。"
                ),
                "en": (
                    "You show a clear preference for Sensing (S: {pct:.0f}%). "
                    "You focus on concrete facts and practical details, preferring verifiable "
                    "information and hands-on experience."
                ),
            },
            "N": {
                "zh": (
                    "你在直觉维度上倾向明显（N: {pct:.0f}%）。你偏好抽象概念和未来可能性，"
                    "善于发现事物间的模式和联系，对理论和想象的世界充满兴趣。"
                ),
                "en": (
                    "You show a clear preference for Intuition (N: {pct:.0f}%). "
                    "You gravitate toward abstract concepts and future possibilities, "
                    "excelling at recognizing patterns and connections between ideas."
                ),
            },
        },
        "TF": {
            "T": {
                "zh": (
                    "你在思考维度上倾向明显（T: {pct:.0f}%）。你倾向于用逻辑和客观标准做决策，"
                    "重视公平和一致性，善于分析问题的因果关系。"
                ),
                "en": (
                    "You show a clear preference for Thinking (T: {pct:.0f}%). "
                    "You make decisions based on logic and objective criteria, "
                    "valuing fairness and consistency in your analysis."
                ),
            },
            "F": {
                "zh": (
                    "你在情感维度上倾向明显（F: {pct:.0f}%）。你倾向于用价值观和人际和谐做决策，"
                    "重视他人的感受，在做选择时考虑对人的影响。"
                ),
                "en": (
                    "You show a clear preference for Feeling (F: {pct:.0f}%). "
                    "You make decisions based on values and interpersonal harmony, "
                    "considering how choices affect the people involved."
                ),
            },
        },
        "JP": {
            "J": {
                "zh": (
                    "你在判断维度上倾向明显（J: {pct:.0f}%）。你喜欢计划和组织，"
                    "倾向于提前安排、按部就班地完成任务，对结构和确定性有较高的需求。"
                ),
                "en": (
                    "You show a clear preference for Judging (J: {pct:.0f}%). "
                    "You prefer planning and organization, working methodically toward "
                    "goals with a need for structure and closure."
                ),
            },
            "P": {
                "zh": (
                    "你在感知维度上倾向明显（P: {pct:.0f}%）。你偏好灵活和开放的生活方式，"
                    "喜欢保留选择余地，适应性强，乐于接受变化和新的可能性。"
                ),
                "en": (
                    "You show a clear preference for Perceiving (P: {pct:.0f}%). "
                    "You prefer a flexible, open-ended approach to life, keeping options "
                    "available and adapting easily to change."
                ),
            },
        },
    }

    # ================================================================
    # 16 种 MBTI 类型的完整描述
    # ================================================================

    TYPE_SUMMARIES = {
        "INTJ": {
            "zh": "你是一个 INTJ（建筑师）—— 具有战略眼光的独立思考者。你以全局视角规划未来，"
                  "用逻辑分析一切，追求知识和能力的持续提升。",
            "en": "You are an INTJ (Architect) — a strategic, independent thinker who plans "
                  "the future with a big-picture perspective, analyzes everything logically, "
                  "and constantly pursues knowledge and competence.",
        },
        "INTP": {
            "zh": "你是一个 INTP（逻辑学家）—— 充满好奇心的理论探索者。你享受拆解复杂问题，"
                  "追求逻辑自洽的知识体系，对未知领域有无穷的探索欲。",
            "en": "You are an INTP (Logician) — a curious theorist who enjoys deconstructing "
                  "complex problems and building logically coherent knowledge systems.",
        },
        "INFJ": {
            "zh": "你是一个 INFJ（提倡者）—— 富有同理心的理想主义者。你深刻理解他人情感，"
                  "以坚定的价值观为导向，希望为世界带来积极的改变。",
            "en": "You are an INFJ (Advocate) — an empathetic idealist with deep insight into "
                  "others' emotions, guided by strong values toward meaningful change.",
        },
        "INFP": {
            "zh": "你是一个 INFP（调停者）—— 充满诗意的理想主义者。你内心丰富、情感细腻，"
                  "忠于自己的价值观，在追寻生命意义和美的过程中找到满足。",
            "en": "You are an INFP (Mediator) — a poetic idealist with a rich inner world, "
                  "loyal to your values and fulfilled by the pursuit of meaning and beauty.",
        },
        "ENTJ": {
            "zh": "你是一个 ENTJ（指挥官）—— 果断坚定、目标驱动的领导者。你擅长组织资源、"
                  "制定战略，以强大的执行力推动团队走向成功。",
            "en": "You are an ENTJ (Commander) — a decisive, goal-oriented leader who excels "
                  "at organizing resources and executing strategies with strong determination.",
        },
        "ENTP": {
            "zh": "你是一个 ENTP（辩论家）—— 思维敏捷的创新者。你享受智力挑战和思想碰撞，"
                  "善于从多角度看问题，不断产生新的想法和解决方案。",
            "en": "You are an ENTP (Debater) — a quick-witted innovator who enjoys intellectual "
                  "challenges and exploring ideas from multiple perspectives.",
        },
        "ENFJ": {
            "zh": "你是一个 ENFJ（主人公）—— 富有魅力和感染力的引导者。你善于激励他人，"
                  "以真诚和热情凝聚共识，带领团队朝着共同目标前进。",
            "en": "You are an ENFJ (Protagonist) — a charismatic leader who inspires others "
                  "with authenticity and enthusiasm, uniting people toward shared goals.",
        },
        "ENFP": {
            "zh": "你是一个 ENFP（竞选者）—— 充满热情和创造力的自由灵魂。你对生活充满好奇，"
                  "善于发现可能性，以乐观和感染力影响身边的人。",
            "en": "You are an ENFP (Campaigner) — an enthusiastic, creative free spirit who "
                  "discovers possibilities everywhere and influences others with optimism.",
        },
        "ISTJ": {
            "zh": "你是一个 ISTJ（物流师）—— 可靠、务实的执行者。你重视规则和责任，"
                  "以系统化和一丝不苟的方式完成任务，是值得信赖的基石。",
            "en": "You are an ISTJ (Logistician) — a reliable, practical executor who values "
                  "rules and responsibility, completing tasks systematically and meticulously.",
        },
        "ISFJ": {
            "zh": "你是一个 ISFJ（守卫者）—— 温暖、细致的守护者。你以无私的奉献照顾身边的人，"
                  "在默默付出中获得满足，是团队中不可或缺的稳定力量。",
            "en": "You are an ISFJ (Defender) — a warm, detail-oriented guardian who cares "
                  "for those around you with quiet dedication.",
        },
        "ISTP": {
            "zh": "你是一个 ISTP（鉴赏家）—— 冷静、务实的实践者。你擅长动手解决问题，"
                  "对工具和技术有天然的敏感度，在危机中保持沉着。",
            "en": "You are an ISTP (Virtuoso) — a calm, practical experimenter who excels "
                  "at hands-on problem solving with a natural affinity for tools and technology.",
        },
        "ISFP": {
            "zh": "你是一个 ISFP（探险家）—— 温柔、敏感的艺术家。你以独特的美学视角探索世界，"
                  "在创造和体验中表达自我，珍视当下的美好。",
            "en": "You are an ISFP (Adventurer) — a gentle, sensitive artist who explores "
                  "the world through a unique aesthetic lens, expressing yourself through creation.",
        },
        "ESTJ": {
            "zh": "你是一个 ESTJ（总经理）—— 高效、果断的管理者。你擅长制定规则、"
                  "组织团队、推动执行，以结果为导向实现既定目标。",
            "en": "You are an ESTJ (Executive) — an efficient, decisive manager who excels "
                  "at setting rules, organizing teams, and driving execution toward results.",
        },
        "ESFJ": {
            "zh": "你是一个 ESFJ（执政官）—— 热心、尽责的服务者。你关注他人的需求，"
                  "以温暖和责任感维护和谐的社交环境，是社区的中坚力量。",
            "en": "You are an ESFJ (Consul) — a caring, responsible provider who attends "
                  "to others' needs and maintains harmonious social environments.",
        },
        "ESTP": {
            "zh": "你是一个 ESTP（企业家）—— 机智、大胆的行动派。你享受冒险和即兴发挥，"
                  "善于抓住机会，以务实的态度和快速的反应解决眼前的问题。",
            "en": "You are an ESTP (Entrepreneur) — a clever, bold doer who enjoys risk "
                  "and improvisation, seizing opportunities with practical, quick responses.",
        },
        "ESFP": {
            "zh": "你是一个 ESFP（表演者）—— 热情、自发的体验者。你热爱生活、享受当下，"
                  "以活力和幽默感染周围的人，是天生的氛围创造者。",
            "en": "You are an ESFP (Entertainer) — an enthusiastic, spontaneous experiencer "
                  "who loves life, enjoys the moment, and lifts spirits with energy and humor.",
        },
    }

    DIM_LABEL_MAP = {
        "EI": {"E": "外向 (Extraversion)", "I": "内向 (Introversion)"},
        "SN": {"S": "感觉 (Sensing)", "N": "直觉 (Intuition)"},
        "TF": {"T": "思考 (Thinking)", "F": "情感 (Feeling)"},
        "JP": {"J": "判断 (Judging)", "P": "感知 (Perceiving)"},
    }

    # ================================================================
    # 公开接口
    # ================================================================

    def interpret(
        self,
        probabilities: dict[str, dict[str, float]],
        keywords: dict[str, list[dict]] | None = None,
        mbti_type: str | None = None,
        lang: str = "zh",
    ) -> dict:
        """
        生成完整的人格解读。

        Args:
            probabilities: {"EI": {"positive": 0.32, "negative": 0.68}, ...}
                           positive = E/S/T/J 的概率
            keywords: token 归因结果 {"EI": [{"token": "...", "score": ...}], ...}
            mbti_type: MBTI 类型字符串 (e.g. "INFP")，None 则从 probs 推导
            lang: 语言 "zh" / "en"

        Returns:
            {
                "interpretation": {"EI": "...", "SN": "...", "TF": "...", "JP": "..."},
                "summary": "...",
                "keywords": {...},
                "mbti_type": "INFP"
            }
        """
        self.LANG = lang

        # 从概率推导 MBTI 类型
        if mbti_type is None:
            mbti_type = self._prob_to_type(probabilities)

        # 每维度解读
        interpretation = {}
        for dim in ["EI", "SN", "TF", "JP"]:
            prob = probabilities[dim]
            # positive = E/S/T/J 的概率 (polar 1)
            p_pos = prob.get("positive", 0.5)
            p_neg = prob.get("negative", 0.5)

            if p_pos >= 0.5:
                polar = list(self.DIM_LABEL_MAP[dim].keys())[0]  # E/S/T/J
                pct = p_pos * 100
            else:
                polar = list(self.DIM_LABEL_MAP[dim].keys())[1]  # I/N/F/P
                pct = p_neg * 100

            template = self.DIM_TEMPLATES[dim][polar].get(lang,
                          self.DIM_TEMPLATES[dim][polar]["en"])
            interpretation[dim] = template.format(pct=pct)

        # 综合总结
        summary_template = self.TYPE_SUMMARIES.get(
            mbti_type,
            {"zh": f"你的 MBTI 类型为 {mbti_type}。",
             "en": f"Your MBTI type is {mbti_type}."},
        )
        summary = summary_template.get(lang, summary_template["en"])

        result: dict = {
            "mbti_type": mbti_type,
            "interpretation": interpretation,
            "summary": summary,
        }

        if keywords:
            result["keywords"] = keywords

        return result

    # ================================================================
    # 辅助
    # ================================================================

    @staticmethod
    def _prob_to_type(probabilities: dict) -> str:
        """从概率字典推导 MBTI 类型字符串。"""
        threshold = 0.5
        ei = "E" if probabilities["EI"].get("positive", 0) >= threshold else "I"
        sn = "S" if probabilities["SN"].get("positive", 0) >= threshold else "N"
        tf = "T" if probabilities["TF"].get("positive", 0) >= threshold else "F"
        jp = "J" if probabilities["JP"].get("positive", 0) >= threshold else "P"
        return f"{ei}{sn}{tf}{jp}"
