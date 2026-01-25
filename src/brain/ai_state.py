"""
AIState - Internal state management for AI VTuber personality.

The AI's behavior changes based on these internal states,
creating a more dynamic and unpredictable personality.
"""

import time
import random
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Mood(Enum):
    """Current mood affecting speech patterns."""
    ENERGETIC = "energetic"     # テンション高め
    CHILL = "chill"             # まったり
    SASSY = "sassy"             # 煽りモード入ってる
    BORED = "bored"             # 暇すぎてダルい
    CURIOUS = "curious"         # 何かに興味持ってる


@dataclass
class AIState:
    """
    AI内部状態を管理するクラス。
    時間経過やイベントで状態が変化し、それに応じて振る舞いが変わる。
    """
    
    # Core states (0.0 ~ 1.0)
    boredom: float = 0.0        # 暇度。高いと脱線・ボケが増える
    energy: float = 0.7         # テンション。低いとダルそうに話す
    sass: float = 0.3           # 煽り度。高いと視聴者をいじる
    focus: float = 0.5          # 集中度。低いと話題がコロコロ変わる
    
    # Current mood (derived from core states)
    mood: Mood = Mood.CHILL
    
    # Topic tracking
    current_topic: str = ""
    topic_start_time: float = field(default_factory=time.time)
    topics_discussed: list = field(default_factory=list)
    
    # Timing
    last_boke_time: float = 0.0         # 最後にボケた時間
    last_interaction_time: float = field(default_factory=time.time)
    last_comment_reaction: float = 0.0  # 最後にコメントに反応した時間
    
    # Counters
    consecutive_monologues: int = 0     # 連続で独り言を言った回数
    
    def update(self, elapsed_seconds: float, event: Optional[str] = None):
        """
        時間経過やイベントで状態を更新。
        
        Args:
            elapsed_seconds: 前回の更新からの経過秒数
            event: 発生したイベント (comment_received, spoke, boke, etc.)
        """
        # --- 時間経過による自然な変化 ---
        
        # 暇度は時間とともに上昇
        self.boredom = min(1.0, self.boredom + elapsed_seconds * 0.005)
        
        # テンションは徐々に下がる（何もしないと）
        self.energy = max(0.2, self.energy - elapsed_seconds * 0.002)
        
        # 集中力も徐々に下がる
        self.focus = max(0.1, self.focus - elapsed_seconds * 0.003)
        
        # 同じ話題が続くと飽きる
        topic_duration = time.time() - self.topic_start_time
        if topic_duration > 120:  # 2分以上同じ話題
            self.boredom = min(1.0, self.boredom + 0.1)
            self.focus = max(0.1, self.focus - 0.1)
        
        # --- イベントによる状態変化 ---
        if event:
            self._handle_event(event)
        
        # --- ムードを更新 ---
        self._update_mood()
    
    def _handle_event(self, event: str):
        """イベントに応じて状態を変更。"""
        
        if event == "comment_received":
            # コメントが来るとテンション上がる
            self.energy = min(1.0, self.energy + 0.15)
            self.boredom = max(0.0, self.boredom - 0.2)
            self.last_comment_reaction = time.time()
            self.consecutive_monologues = 0
        
        elif event == "spoke":
            # 話すとちょっとスッキリ
            self.boredom = max(0.0, self.boredom - 0.05)
            self.last_interaction_time = time.time()
        
        elif event == "boke":
            # ボケるとテンション上がる、暇度リセット
            self.energy = min(1.0, self.energy + 0.2)
            self.boredom = max(0.0, self.boredom - 0.3)
            self.last_boke_time = time.time()
            self.sass = min(1.0, self.sass + 0.1)
        
        elif event == "topic_change":
            # 話題を変えると集中力回復
            self.focus = min(1.0, self.focus + 0.3)
            self.boredom = max(0.0, self.boredom - 0.15)
            self.topic_start_time = time.time()
        
        elif event == "got_reaction":
            # 視聴者からリアクションあると嬉しい
            self.energy = min(1.0, self.energy + 0.1)
            self.sass = max(0.0, self.sass - 0.05)  # ちょっと素直になる
        
        elif event == "monologue":
            # 独り言が続くと
            self.consecutive_monologues += 1
            if self.consecutive_monologues > 3:
                self.boredom = min(1.0, self.boredom + 0.2)
    
    def _update_mood(self):
        """core statesからmoodを決定。"""
        
        if self.boredom > 0.7:
            self.mood = Mood.BORED
        elif self.sass > 0.6:
            self.mood = Mood.SASSY
        elif self.energy > 0.7:
            self.mood = Mood.ENERGETIC
        elif self.focus > 0.6:
            self.mood = Mood.CURIOUS
        else:
            self.mood = Mood.CHILL
    
    def change_topic(self, new_topic: str):
        """話題を変更。"""
        if self.current_topic and self.current_topic != new_topic:
            self.topics_discussed.append(self.current_topic)
            # 古い話題は忘れていく
            if len(self.topics_discussed) > 10:
                self.topics_discussed.pop(0)
        
        self.current_topic = new_topic
        self.topic_start_time = time.time()
        self._handle_event("topic_change")
    
    def should_boke(self) -> bool:
        """ボケるべきかどうかを判定。"""
        # 最後のボケから30秒以上経過している
        time_since_boke = time.time() - self.last_boke_time
        if time_since_boke < 30:
            return False
        
        # 暇度が高いほどボケやすい
        boke_chance = self.boredom * 0.4 + (1 - self.focus) * 0.3
        
        # ランダム要素
        return random.random() < boke_chance
    
    def should_change_topic(self) -> bool:
        """話題を変えるべきかどうかを判定。"""
        topic_duration = time.time() - self.topic_start_time
        
        # 2分以上同じ話題
        if topic_duration > 120:
            return True
        
        # 集中力が低い + 暇度が高い
        if self.focus < 0.3 and self.boredom > 0.5:
            return random.random() < 0.3
        
        return False
    
    def get_temperature_modifier(self) -> float:
        """LLMのtemperatureの補正値を返す。"""
        # 基本: 0.7
        # 暇度が高いと上がる（ボケやすくなる）
        # 集中力が高いと下がる（安定する）
        modifier = 0.0
        modifier += self.boredom * 0.2      # max +0.2
        modifier -= self.focus * 0.1         # max -0.1
        modifier += (1 - self.energy) * 0.1  # max +0.1
        
        return modifier
    
    def get_state_summary(self) -> str:
        """デバッグ用に状態をサマリー。"""
        return (
            f"[State] mood={self.mood.value}, boredom={self.boredom:.2f}, "
            f"energy={self.energy:.2f}, sass={self.sass:.2f}, focus={self.focus:.2f}"
        )
    
    def to_prompt_hint(self) -> str:
        """LLMに渡すためのヒント文字列を生成。"""
        hints = []
        
        if self.mood == Mood.BORED:
            hints.append("今かなり暇。何か面白いことしたい")
        elif self.mood == Mood.SASSY:
            hints.append("ちょっとイジワル気分")
        elif self.mood == Mood.ENERGETIC:
            hints.append("テンション高め")
        elif self.mood == Mood.CURIOUS:
            hints.append("何かに興味津々")
        
        if self.consecutive_monologues > 2:
            hints.append("独り言が続いてる。誰かと話したい")
        
        if self.boredom > 0.6:
            hints.append("同じ話題に飽きてきた")
        
        return "、".join(hints) if hints else ""
