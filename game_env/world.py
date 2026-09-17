# -*- coding: utf-8 -*-
"""轻量文本游戏引擎：语言 → 决策 → 游戏动作 → 世界状态。

设计目标（对齐米哈游「NPC 看起来会说、实际上不会玩」的失败模式）：
  * 观测 = 自然语言描述，动作 = 受限 token（便于小模型可靠输出）
  * 每步动作都真实改变一个可验证的世界状态
  * 任务可自动判定成功/失败，且保证有解，因此适合做 RL 环境与 Agent 评测

动作空间刻意做窄（9 个动词），因为 0.5B 小模型在自由文本上容易崩，
而在受限动作集上才能把「会不会玩」与「会不会说」分开测。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

DIRS = {"north": "北", "south": "南", "east": "东", "west": "西"}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}

ROOM_NAMES = ["门厅", "书房", "厨房", "花园", "阁楼", "地窖", "走廊"]
ITEMS = ["铜钥匙", "铁钥匙", "油灯", "火柴", "旧书", "银烛台", "水桶", "地图"]
DECOR = ["地毯", "挂画", "木箱", "书架", "壁炉"]
NPCS = {"管事": "这里的东西都有它的位置，别乱翻。",
        "侍女": "阁楼很久没人打扫了，小心灰尘。"}


@dataclass
class Room:
    name: str
    desc: str
    links: Dict[str, str] = field(default_factory=dict)   # dir -> room name
    locks: Dict[str, str] = field(default_factory=dict)   # dir -> key item
    items: List[str] = field(default_factory=list)
    hidden: List[str] = field(default_factory=list)       # 需 search 才发现
    npcs: List[str] = field(default_factory=list)
    seen: bool = False
    searched: bool = False
    lit: bool = True


@dataclass
class Quest:
    """任务定义。success 用 lambda 判定，保证可自动 rice."""
    desc: str
    kind: str
    target_item: Optional[str] = None
    target_room: Optional[str] = None
    need_lit: bool = False


class World:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.step_count = 0
        self.rooms: Dict[str, Room] = {}
        self.inventory: List[str] = []
        self.done = False
        self.failed = False
        self.messages: List[str] = []
        self.history: List[Tuple[str, str]] = []  # (action, observation)
        self._build()
        self.quest = self._make_quest()

    # ---------------- 世界构建 ----------------
    def _build(self):
        names = self.rng.sample(ROOM_NAMES, 5)
        layout = {
            names[0]: {"east": names[1], "south": names[2]},
            names[1]: {"west": names[0], "north": names[3]},
            names[2]: {"north": names[0], "east": names[4]},
            names[3]: {"south": names[1]},
            names[4]: {"west": names[2]},
        }
        for n in names:
            self.rooms[n] = Room(name=n, desc=f"这里是{n}，光线{"明亮" if True else "昏暗"}。")
        for a, d in layout.items():
            self.rooms[a].links = dict(d)

        # 一把钥匙 + 一扇上锁的门（多步依赖）
        key = self.rng.choice(["铜钥匙", "铁钥匙"])
        locked_room = self.rooms[names[3]]
        container_room = self.rng.choice([names[2], names[4]])
        self.rooms[container_room].hidden.append(key)
        self.rooms[container_room].hidden.append("地图")
        # 上锁方向 = 通往阁楼的 north
        for d, r in list(self.rooms[names[1]].links.items()):
            if r == names[3]:
                self.rooms[names[1]].locks[d] = key
        self.key_item = key
        self.locked_owner = names[1]

        # 其余道具散落
        pool = [i for i in ITEMS if i != key]
        for it in self.rng.sample(pool, 3):
            self.rooms[self.rng.choice(names)].items.append(it)
        self.rooms[names[0]].npcs = ["管事"]
        self.rooms[names[2]].npcs = ["侍女"] if self.rng.random() < 0.7 else []
        self.start = names[0]
        self.goal_room = names[3]  # 阁楼
        self.cur = self.start

    def _make_quest(self) -> Quest:
        # 关键：只从「非上锁房间且非隐藏」的物品里挑任务目标，保证任务必然有解。
        # 若挑到锁门后的物品，任务直接变成死局，RL 奖励会永久为 0（无学习信号）。
        carry = [i for name, room in self.rooms.items()
                 if name != self.goal_room for i in room.items]
        if not carry:
            carry = ["旧书"]
            self.rooms[self.start].items.append("旧书")
        item = self.rng.choice(carry)
        if self.rng.random() < 0.5:
            return Quest(desc=f"把「{item}」送到{self.goal_room}。", kind="deliver",
                         target_item=item, target_room=self.goal_room)
        return Quest(desc=f"找到「{self.key_item}」并进入{self.goal_room}。",
                     kind="unlock", target_room=self.goal_room)

    # ---------------- 观测 ----------------
    def observe(self) -> str:
        r = self.rooms[self.cur]
        parts = [f"[位置] {r.name}", r.desc]
        if r.items:
            parts.append("这里放着：" + "、".join(r.items))
        if r.npcs:
            parts.append("有人在场：" + "、".join(r.npcs))
        exits = []
        for d, tgt in sorted(r.links.items()):
            tag = f"{DIRS[d]}(锁住，需要{r.locks[d]})" if d in r.locks else f"{DIRS[d]}"
            exits.append(f"{tag}→{tgt}")
        parts.append("出口：" + "，".join(exits))
        parts.append("随身物品：" + ("、".join(self.inventory) if self.inventory else "空"))
        return "\n".join(parts)

    # ---------------- 动作执行 ----------------
    def step(self, action: str) -> str:
        """执行一个动作，返回观测文本。动作语法受限，解析失败视为无效步。"""
        self.step_count += 1
        a = (action or "").strip()
        obs = self._dispatch(a)
        self.history.append((a, obs))
        self._update_done()
        return obs

    def _dispatch(self, a: str) -> str:
        verb = a.split()[0] if a.split() else ""
        arg = " ".join(a.split()[1:]).strip()
        handlers = {
            "go": self._go, "take": self._take, "drop": self._drop,
            "search": self._search, "unlock": self._unlock,
            "talk": self._talk, "use": self._use, "look": self._look,
            "inventory": self._inv,
        }
        h = handlers.get(verb)
        if h is None:
            self.messages.append(f"无法识别的动作：{a}")
            return f"无效动作「{a}」。可用动作：go / take / drop / search / unlock / talk / use / look / inventory。"
        return h(arg)

    def _rev_dir(self, target: str) -> Optional[str]:
        r = self.rooms[self.cur]
        for d, t in r.links.items():
            if t == target:
                return d
        return None

    def _go(self, arg: str) -> str:
        # 支持 go north / go 书房
        r = self.rooms[self.cur]
        d = None
        rev = {v: k for k, v in DIRS.items()}
        if arg in DIRS or arg in rev:
            d = rev.get(arg, arg)
        else:
            d = self._rev_dir(arg)
        if d is None or d not in r.links:
            return f"那边过不去。当前出口：{', '.join(DIRS[x] for x in r.links)}。"
        if d in r.locks:
            return f"{DIRS[d]}边的门锁着，需要「{r.locks[d]}」才能打开。"
        self.cur = r.links[d]
        self.rooms[self.cur].seen = True
        return f"你向北走去。\n" + self.observe() if False else f"你向{DIRS[d]}走去。\n{self.observe()}"

    def _take(self, arg: str) -> str:
        r = self.rooms[self.cur]
        it = arg.strip("「」《》 ")
        if it in r.items:
            r.items.remove(it)
            self.inventory.append(it)
            return f"你拿起了{it}。"
        if it in r.hidden:
            return f"这里没有{it}，也许该仔细搜查一下（search）。"
        return f"这里没有可拿的{it}。"

    def _drop(self, arg: str) -> str:
        it = arg.strip("「」《》 ")
        if it in self.inventory:
            self.inventory.remove(it)
            self.rooms[self.cur].items.append(it)
            return f"你把{it}放下了。"
        return f"你身上没有{it}。"

    def _search(self, arg: str = "") -> str:
        r = self.rooms[self.cur]
        if r.searched and not r.hidden:
            return "你已经把这里翻遍了，没有新发现。"
        if r.hidden:
            found = list(r.hidden)
            r.items.extend(found)
            r.hidden = []
            r.searched = True
            return "你仔细搜查，发现：" + "、".join(found) + "。\n" + self.observe()
        r.searched = True
        return "你搜了一遍，没有发现藏着的东西。"

    def _unlock(self, arg: str) -> str:
        # unlock north with 铜钥匙  或  unlock 北 with 铜钥匙
        r = self.rooms[self.cur]
        rev = {v: k for k, v in DIRS.items()}
        tok = arg.replace(" with ", " ").split()
        d = None
        key = None
        for t in tok:
            if t in DIRS or t in rev:
                d = rev.get(t, t)
            elif t.startswith("钥匙") or "钥匙" in t:
                key = t
        if d is None:
            for dd in r.links:
                if dd in r.locks:
                    d = dd
                    break
        if d is None or d not in r.locks:
            return "这个方向的门没有上锁。"
        need = r.locks[d]
        if key and key != need:
            return f"「{key}」打不开这扇门，需要「{need}」。"
        if need not in self.inventory:
            return f"你身上没有「{need}」，无法打开{DIRS[d]}边的门。"
        del r.locks[d]
        return f"「{need}」转动了两下，{DIRS[d]}边的门开了。"

    def _talk(self, arg: str) -> str:
        r = self.rooms[self.cur]
        who = arg.strip()
        if who not in r.npcs:
            return f"这里没有{who}可以交谈。" if who else "没有人可以交谈。"
        return f"{who}说：“{NPCS.get(who, '……')}”"

    def _use(self, arg: str) -> str:
        it = arg.split()[0] if arg.split() else ""
        if it not in self.inventory:
            return f"你身上没有{it}，无法使用。"
        if "地图" in it:
            others = [r for r in self.rooms if r != self.cur]
            return "你展开地图，看到连通的地点：" + "、".join(others)
        return f"你试着使用了{it}，但没有发生什么特别的事。"

    def _look(self, arg: str = "") -> str:
        return self.observe()

    def _inv(self, arg: str = "") -> str:
        return "随身物品：" + ("、".join(self.inventory) if self.inventory else "空")

    # ---------------- 任务判定 ----------------
    def _update_done(self):
        q = self.quest
        if q.kind == "deliver":
            if q.target_item in self.rooms[q.target_room].items and self.cur == q.target_room:
                self.done = True
        elif q.kind == "unlock":
            if self.cur == q.target_room:
                self.done = True

    def success(self) -> bool:
        return self.done

    def prompt(self) -> str:
        """给 Agent 的完整提示：任务 + 当前观测 + 可用动作。"""
        acts = "go <方向/房间> | take <物品> | drop <物品> | search | unlock <方向> with <钥匙> | talk <人名> | use <物品> | look | inventory"
        obs = self.observe()
        tail = ""
        if self.history:
            tail = "\n".join(f"> {a}\n{o.splitlines()[0]}" for a, o in self.history[-6:])
        return (
            f"任务：{self.quest.desc}\n"
            f"可用动作：{acts}\n"
            f"{obs}\n"
            + (f"最近几步：\n{tail}\n" if tail else "")
        )


# ---------------- 用于评测的批量任务生成 ----------------
def make_tasks(n: int, seed: int = 0) -> List[dict]:
    """生成 n 个保证有解的任务（世界种子不同 => 布局/道具不同）。"""
    out = []
    for i in range(n):
        w = World(seed=seed * 1000 + i)
        out.append({
            "id": f"g{i:03d}",
            "seed": seed * 1000 + i,
            "quest": w.quest.desc,
            "kind": w.quest.kind,
            "target_item": w.quest.target_item,
            "target_room": w.quest.target_room,
        })
    return out


if __name__ == "__main__":
    # 自洽性检查：随机走会不会把游戏搞崩
    import random as _r
    ok = 0
    for s in range(200):
        w = World(seed=s)
        rr = _r.Random(s)
        for _ in range(30):
            act = rr.choice(["look", "search", "inventory", "go 北", "go 东", "take 旧书",
                             "drop 旧书", "use 地图", "unlock 北 with 铜钥匙", "talk 管事"])
            w.step(act)
            if w.success():
                break
        ok += 1
    print(f"world self-test OK: {ok}/200 episodes ran without exception")
    w = World(seed=1)
    print("--- sample prompt ---")
    print(w.prompt())
