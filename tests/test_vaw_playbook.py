"""Playbook composition, gen-0 golden equality, and contract hash."""

from __future__ import annotations

import hashlib

from vaw.context_runtime.playbook import (
    compose_default_main_prompt,
    compose_prompt,
    default_playbook_dir,
    load_playbooks,
)
from vaw.context_runtime.protocol import (
    CONTRACT_COORDS,
    CONTRACT_PREAMBLE,
    SYSTEM_PROMPT,
)

# Frozen hash of CONTRACT_PREAMBLE + CONTRACT_COORDS.  Bump only when the
# kernel legend / mechanism text is intentionally revised.
_CONTRACT_SHA256 = "8a710d135ac64d91ac20a40bbb6ce1f2fd12cdb3af718c848b9ccb8a90504e71"


def test_gen0_compose_matches_golden_system_prompt() -> None:
    golden = (default_playbook_dir() / "gen0.golden.txt").read_text(encoding="utf-8")
    composed = compose_default_main_prompt(injection="all")
    assert composed == golden
    assert SYSTEM_PROMPT == golden


def test_contract_prompt_hash_is_frozen() -> None:
    payload = CONTRACT_PREAMBLE + CONTRACT_COORDS
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert digest == _CONTRACT_SHA256


def test_playbook_library_has_six_phase_files() -> None:
    playbooks = load_playbooks()
    assert set(playbooks) == {
        "grasp",
        "transport",
        "align",
        "place",
        "recover",
        "routing",
    }
    assert "select" in playbooks["routing"]
    assert "amber" in playbooks["routing"]
    assert "grounding" in playbooks["recover"]
    assert "restore" in playbooks["recover"]
    assert "close" in playbooks["grasp"]
    assert "geometry" in playbooks["grasp"]


def test_phase_injection_uses_one_playbook_plus_contract() -> None:
    playbooks = load_playbooks()
    text = compose_prompt(
        playbooks,
        contract_preamble=CONTRACT_PREAMBLE,
        contract_coords=CONTRACT_COORDS,
        injection="phase",
        phase="place",
    )
    assert CONTRACT_PREAMBLE in text
    assert CONTRACT_COORDS in text
    assert "放置目标不要求完美居中" in text
    assert "call_imagination" not in text


def test_strategy_sentences_live_in_playbooks() -> None:
    playbooks = load_playbooks()
    routing = "".join(playbooks["routing"].values())
    place = "".join(playbooks["place"].values())
    align = "".join(playbooks["align"].values())
    recover = "".join(playbooks["recover"].values())
    grasp = "".join(playbooks["grasp"].values())
    transport = "".join(playbooks["transport"].values())

    assert "以下情形默认\n先委派" in routing
    assert "这些是路由建议而非门控" in routing
    assert "action_proposal.executable 是能否 commit 的权威标志" in routing
    assert "只有回滚 Action 的 executable=true 才可直接 commit" in routing
    assert "不要重新 detection/propose 生成等价集合" in routing
    assert "薄/扁物体" in routing
    assert "琥珀色 carried-volume 是可选" in routing
    for reason in ("geometry_unresolved", "plan_unavailable", "turn_limit", "subagent_error"):
        assert reason in routing
    assert "不是 commit 的前置" in routing

    assert "二维重叠" in place
    assert "物体下部已经越过开口/边沿平面" in place
    assert "而非停在边沿上" in place

    assert "优先保留该 metric anchor" in align
    assert "被禁止的是继续下降和释放，而不是修正本身" in align
    assert "而不是重复同一感知循环或按次数切换策略" in align

    assert "region/point 会自动对照新画面复验" in recover
    assert "status=occluded" in recover
    assert "world change check" in recover
    assert "seed 与 ActionProposal 仍是 revision-local" in recover
    assert "只有目标离开局部视野、身份不确定或必须重新生成抓取方向时" in recover
    assert "不要仅因蓝轮廓已经消失而重新 detection" in recover

    assert "较大的 GRIP 可能表示物体阻挡" in grasp
    assert "不得让已闭合的手指朝支撑面" in grasp
    assert "指尖低于物体顶面可以是正常抓取状态" in grasp
    assert "不得把" in grasp and "指尖—顶面距离" in grasp

    assert "一次很短的抬升中物体瞬时随动" in transport


def test_contract_legend_sentences_stay_in_preamble() -> None:
    assert "Main ReAct Agent" in CONTRACT_PREAMBLE
    assert "Task Memory" in CONTRACT_PREAMBLE
    assert "executed 只表示命令完成" in CONTRACT_PREAMBLE
    assert "物理接触只看当前 Canvas，不看已经发出过多少次同类命令" in CONTRACT_PREAMBLE
    assert "region/point 仍 verified 只表示画面里还能认出同一物体" in CONTRACT_PREAMBLE
    assert "容器已不再是可用开口" in CONTRACT_PREAMBLE
    assert 'CONTACT SIDE 会抬高为斜俯视，标题标出 "OBLIQUE <角度>° DOWN"' in CONTRACT_PREAMBLE
    assert "横向对齐以带 OBLIQUE 标记的那一幅为准" in CONTRACT_PREAMBLE
    assert "竖直方向同时混合了高度与进深" in CONTRACT_PREAMBLE
    assert "掌部底面标为一条深青色窄带" in CONTRACT_PREAMBLE
    assert "窄带一旦贴到当前正下方的顶面或沿口" in CONTRACT_PREAMBLE
    assert "Action 执行后蓝轮廓" in CONTRACT_PREAMBLE
    assert "VIRTUAL TOP" not in CONTRACT_PREAMBLE
    assert "descending delta_move #" not in CONTRACT_PREAMBLE
    assert "Current Function Event" in CONTRACT_PREAMBLE
