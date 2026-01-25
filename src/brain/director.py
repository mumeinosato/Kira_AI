"""
Director - Decides what action the AI should take based on current state.

The Director is the "brain" that looks at the AI's internal state,
external context (comments, time, etc.), and decides the next action.
"""

import random
import time
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

from .ai_state import AIState, Mood


class ActionMode(Enum):
    """The type of action to take."""
    REACT = "react"           # コメントに反応（短く鋭く）
    MONOLOGUE = "monologue"   # 独白（自分から話す）
    BOKE = "boke"             # ボケる（突拍子もないこと）
    TEASE = "tease"           # 視聴者をいじる
    WAIT = "wait"             # 何もしない（静かに待つ）


@dataclass
class ActionPlan:
    """Director が決定したアクションプラン。"""
    mode: ActionMode
    directive: str              # LLMに渡す指示
    temperature: float = 0.7    # LLMのtemperature
    max_tokens: int = 150       # 最大トークン数
    priority: int = 0           # 優先度（高いほど優先）
    
    # オプション: 特定のコンテキスト
    comment_to_react: Optional[str] = None
    topic_hint: Optional[str] = None


class Director:
    """
    AIの行動を決定するディレクター。
    状態とコンテキストを見て、次に何をすべきかを決める。
    """
    
    # ボケのパターン（これらをヒントとしてLLMに渡す）
    BOKE_PATTERNS = [
        "突然全く関係ない疑問を口に出す",
        "さっき言ったことと矛盾することを自信満々に言う",
        "明らかに間違った豆知識を披露する",
        "急に哲学的なことを言い出す",
        "唐突に自分の失敗談を思い出す",
        "何かを思い出しかけて忘れる",
    ]
    
    # 煽りパターン
    TEASE_PATTERNS = [
        "視聴者が少ないことをいじる",
        "コメントの内容に突っ込む",
        "視聴者に無茶振りする",
        "自分の方が詳しいアピールをする（でも間違ってる）",
    ]
    
    # 独白のきっかけ（話題の種）
    MONOLOGUE_SEEDS = [
        "最近ハマってること",
        "昨日見た夢の話",
        "今欲しいものの話",
        "最近気づいたこと",
        "ふと思い出した昔の話",
        "今の気分",
        "さっき食べたもの/これから食べたいもの",
        "推しの話",
        "最近見たコンテンツの感想",
    ]
    
    def __init__(self):
        self.last_action_time = time.time()
        self.recent_actions: List[ActionMode] = []
    
    def decide_action(self, state: AIState, context: dict) -> ActionPlan:
        """
        状態とコンテキストから次の行動を決定。
        
        Args:
            state: 現在のAI内部状態
            context: 外部コンテキスト
                - has_comments: bool - コメントがあるか
                - comment_count: int - コメント数
                - idle_time: float - 最後の発話からの経過時間
                - recent_topics: list - 最近話した話題
        
        Returns:
            ActionPlan - 次のアクション
        """
        has_comments = context.get("has_comments", False)
        idle_time = context.get("idle_time", 0)
        
        # --- Priority 1: コメントへの反応 ---
        if has_comments:
            # 毎回反応するわけではない（sassが高いと無視することも）
            if random.random() > state.sass * 0.3:
                return self._plan_react(state, context)
        
        # --- Priority 2: 暇すぎる → ボケる or 話題変える ---
        if state.should_boke():
            return self._plan_boke(state)
        
        # --- Priority 3: sassが高い → いじる ---
        if state.mood == Mood.SASSY and random.random() < 0.4:
            return self._plan_tease(state, context)
        
        # --- Priority 4: アイドル時間が短い → 待つ ---
        if idle_time < 5:
            return ActionPlan(
                mode=ActionMode.WAIT,
                directive="",
                priority=-1
            )
        
        # --- Priority 5: 話題を変えるべき ---
        if state.should_change_topic():
            return self._plan_topic_change(state)
        
        # --- Default: 独白 ---
        return self._plan_monologue(state, context)
    
    def _plan_react(self, state: AIState, context: dict) -> ActionPlan:
        """コメントへの反応を計画。"""
        
        # Moodに応じて反応の仕方を変える
        if state.mood == Mood.SASSY:
            directive = "コメントに反応。ちょっとだけいじりつつ返事して。"
            temperature = 0.8
        elif state.mood == Mood.BORED:
            directive = "コメントに反応。ダルそうに、でも嬉しそうに。"
            temperature = 0.7
        elif state.mood == Mood.ENERGETIC:
            directive = "コメントに反応！テンション高めでリアクション！"
            temperature = 0.8
        else:
            directive = "コメントに自然に反応して。"
            temperature = 0.7
        
        return ActionPlan(
            mode=ActionMode.REACT,
            directive=directive,
            temperature=temperature + state.get_temperature_modifier(),
            max_tokens=100,  # 反応は短めに
            priority=10
        )
    
    def _plan_boke(self, state: AIState) -> ActionPlan:
        """ボケを計画。"""
        
        pattern = random.choice(self.BOKE_PATTERNS)
        
        directive = f"[ボケモード] {pattern}。脈絡なくていい。"
        
        return ActionPlan(
            mode=ActionMode.BOKE,
            directive=directive,
            temperature=0.95,  # ボケは高めのtemperatureで
            max_tokens=80,     # 短くインパクト重視
            priority=5
        )
    
    def _plan_tease(self, state: AIState, context: dict) -> ActionPlan:
        """視聴者いじりを計画。"""
        
        pattern = random.choice(self.TEASE_PATTERNS)
        
        return ActionPlan(
            mode=ActionMode.TEASE,
            directive=f"[いじりモード] {pattern}。でも愛を持って。",
            temperature=0.85,
            max_tokens=100,
            priority=3
        )
    
    def _plan_topic_change(self, state: AIState) -> ActionPlan:
        """話題変更を計画。"""
        
        # 最近話してない話題を選ぶ
        available_seeds = [s for s in self.MONOLOGUE_SEEDS 
                          if s not in state.topics_discussed[-3:]]
        
        if not available_seeds:
            available_seeds = self.MONOLOGUE_SEEDS
        
        seed = random.choice(available_seeds)
        
        return ActionPlan(
            mode=ActionMode.MONOLOGUE,
            directive=f"「{seed}」について急に話し始める。前の話題からの繋がりは不要。",
            temperature=0.75 + state.get_temperature_modifier(),
            max_tokens=150,
            priority=4,
            topic_hint=seed
        )
    
    def _plan_monologue(self, state: AIState, context: dict) -> ActionPlan:
        """通常の独白を計画。"""
        
        # 状態に応じたヒント
        state_hint = state.to_prompt_hint()
        
        # 基本的な独白指示
        if state.mood == Mood.BORED:
            base = "何か話したいけど特に話題がない。暇を持て余してる感じで。"
        elif state.mood == Mood.ENERGETIC:
            base = "何か楽しいことを話したい！テンション高めで。"
        elif state.mood == Mood.CURIOUS:
            base = "何か気になることを深掘りする。"
        else:
            base = "自然に何か話す。"
        
        if state.current_topic:
            directive = f"{base}今の話題「{state.current_topic}」に関連して。"
        else:
            directive = base
        
        if state_hint:
            directive += f" (内心: {state_hint})"
        
        return ActionPlan(
            mode=ActionMode.MONOLOGUE,
            directive=directive,
            temperature=0.7 + state.get_temperature_modifier(),
            max_tokens=150,
            priority=1
        )
    
    def record_action(self, action: ActionPlan):
        """実行したアクションを記録。"""
        self.recent_actions.append(action.mode)
        if len(self.recent_actions) > 10:
            self.recent_actions.pop(0)
        self.last_action_time = time.time()
    
    def get_action_variety_score(self) -> float:
        """最近のアクションの多様性スコアを返す（0-1）。"""
        if len(self.recent_actions) < 3:
            return 1.0
        
        unique = len(set(self.recent_actions[-5:]))
        return unique / 5.0
